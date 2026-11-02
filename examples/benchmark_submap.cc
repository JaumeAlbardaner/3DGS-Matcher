/**
 * benchmark_submap.cc
 *
 * Global registration benchmark on pairs of 3DGS submap PLY files.
 * Compares four methods starting from ±max_rot / ±max_trans initial perturbation:
 *   vanilla_kiss  — KISS-Matcher (geometry-only FPFH)
 *   color_cpfh    — 3DGS-Matcher: color + ColorPFH (best config from noise study)
 *   gicp          — small_gicp GICP starting from identity (local baseline, expected to fail)
 *   color_gicp    — small_gicp Color-GICP starting from identity
 *
 * Usage:
 *   benchmark_submap src.ply tgt.ply voxel_size max_rot_deg max_trans_m n_iters seed_start output.csv
 *
 * Output CSV columns:
 *   method, scene, split, rot_error_deg, trans_error_m, success, time_ms
 */

#include <3dgs_matcher/ColorKISSMatcher.hpp>
#include <kiss_matcher/KISSMatcher.hpp>

// small_gicp (standard GICP)
#include <small_gicp/registration/registration_helper.hpp>
#include <small_gicp/util/downsampling.hpp>

// small_gicp Color ICP/GICP (header-only, in 3dgs_matcher tree)
#include <3dgs_matcher/small_gicp_color/color_registration.hpp>
#include <3dgs_matcher/small_gicp_color/multimodal_icp.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// ── Data ────────────────────────────────────────────────────────────────────

struct GaussianSplat {
  Eigen::Vector3f position;
  Eigen::Vector4f color_rgba;
};

// ── PLY loader (same as align_two_splats.cc) ─────────────────────────────────

namespace ply {
struct Property { std::string type, name; bool is_list = false;
                  std::string list_count_type, list_value_type; };

inline std::string toLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
  return s;
}

template <typename T> T readBin(std::ifstream& f) { T v{}; f.read(reinterpret_cast<char*>(&v), sizeof(T)); return v; }

double readScalar(std::ifstream& f, const std::string& type) {
  const std::string t = toLower(type);
  if (t == "float"  || t == "float32") return static_cast<double>(readBin<float>(f));
  if (t == "double" || t == "float64") return readBin<double>(f);
  if (t == "int"    || t == "int32")   return static_cast<double>(readBin<int32_t>(f));
  if (t == "uint"   || t == "uint32")  return static_cast<double>(readBin<uint32_t>(f));
  if (t == "short"  || t == "int16")   return static_cast<double>(readBin<int16_t>(f));
  if (t == "ushort" || t == "uint16")  return static_cast<double>(readBin<uint16_t>(f));
  if (t == "char"   || t == "int8")    return static_cast<double>(readBin<int8_t>(f));
  if (t == "uchar"  || t == "uint8")   return static_cast<double>(readBin<uint8_t>(f));
  throw std::runtime_error("Unknown PLY type: " + type);
}
}  // namespace ply

bool loadPLY(const std::string& filename, std::vector<GaussianSplat>& splats) {
  std::ifstream file(filename, std::ios::binary);
  if (!file) { std::cerr << "Cannot open: " << filename << "\n"; return false; }

  std::string line;
  std::getline(file, line);
  if (ply::toLower(line) != "ply") { std::cerr << "Not a PLY file: " << filename << "\n"; return false; }

  std::vector<ply::Property> props;
  size_t vertex_count = 0;
  bool ascii = false, binary_le = true, in_vertex = false;

  while (std::getline(file, line)) {
    std::istringstream ss(line);
    std::string tok; ss >> tok; tok = ply::toLower(tok);
    if (tok == "end_header") break;
    if (tok == "format") {
      std::string fmt; ss >> fmt;
      ascii     = (ply::toLower(fmt) == "ascii");
      binary_le = (ply::toLower(fmt) == "binary_little_endian");
    } else if (tok == "element") {
      std::string name; ss >> name;
      in_vertex = (ply::toLower(name) == "vertex");
      if (in_vertex) ss >> vertex_count;
    } else if (tok == "property" && in_vertex) {
      ply::Property p;
      std::string next; ss >> next;
      if (ply::toLower(next) == "list") {
        p.is_list = true; ss >> p.list_count_type >> p.list_value_type >> p.name;
      } else { p.type = next; ss >> p.name; }
      p.name = ply::toLower(p.name);
      props.push_back(p);
    }
  }
  if (vertex_count == 0 || props.empty()) return false;

  std::unordered_map<std::string, size_t> idx;
  for (size_t i = 0; i < props.size(); ++i) idx[props[i].name] = i;
  auto find = [&](const std::string& n) -> int {
    auto it = idx.find(n); return it == idx.end() ? -1 : static_cast<int>(it->second);
  };

  const int ix = find("x"), iy = find("y"), iz = find("z");
  if (ix < 0 || iy < 0 || iz < 0) return false;
  const int ir = find("red"), ig = find("green"), ib = find("blue"), ia_rgb = find("alpha");
  const int idc0 = find("f_dc_0"), idc1 = find("f_dc_1"), idc2 = find("f_dc_2");
  const int iop  = find("opacity");

  constexpr double kSHC0 = 0.28209479177387814;
  auto clamp01 = [](float v){ return std::max(0.0f, std::min(1.0f, v)); };
  auto sigmoid  = [](float v){ return 1.0f / (1.0f + std::exp(-v)); };

  splats.clear(); splats.reserve(vertex_count);

  for (size_t vi = 0; vi < vertex_count; ++vi) {
    std::vector<double> vals(props.size(), 0.0);
    if (ascii) {
      if (!std::getline(file, line)) break;
      std::istringstream iss(line);
      for (size_t pi = 0; pi < props.size(); ++pi) {
        if (props[pi].is_list) { int cnt=0; iss>>cnt; for(int k=0;k<cnt;++k){double d;iss>>d;} }
        else iss >> vals[pi];
      }
    } else {
      for (size_t pi = 0; pi < props.size(); ++pi) {
        if (props[pi].is_list) {
          const int n = static_cast<int>(ply::readScalar(file, props[pi].list_count_type));
          for (int k = 0; k < n; ++k) ply::readScalar(file, props[pi].list_value_type);
        } else {
          vals[pi] = ply::readScalar(file, props[pi].type);
        }
      }
    }

    GaussianSplat s;
    s.position = Eigen::Vector3f(static_cast<float>(vals[ix]),
                                 static_cast<float>(vals[iy]),
                                 static_cast<float>(vals[iz]));
    float r = 0.5f, g = 0.5f, b = 0.5f;
    if (ir >= 0 && ig >= 0 && ib >= 0) {
      r = clamp01(static_cast<float>(vals[ir] > 1.0 ? vals[ir]/255.0 : vals[ir]));
      g = clamp01(static_cast<float>(vals[ig] > 1.0 ? vals[ig]/255.0 : vals[ig]));
      b = clamp01(static_cast<float>(vals[ib] > 1.0 ? vals[ib]/255.0 : vals[ib]));
    } else if (idc0 >= 0 && idc1 >= 0 && idc2 >= 0) {
      r = clamp01(static_cast<float>(kSHC0 * vals[idc0] + 0.5));
      g = clamp01(static_cast<float>(kSHC0 * vals[idc1] + 0.5));
      b = clamp01(static_cast<float>(kSHC0 * vals[idc2] + 0.5));
    }
    float a = 1.0f;
    if (ia_rgb >= 0)
      a = clamp01(static_cast<float>(vals[ia_rgb] > 1.0 ? vals[ia_rgb]/255.0 : vals[ia_rgb]));
    else if (iop >= 0)
      a = sigmoid(static_cast<float>(vals[iop]));

    s.color_rgba = Eigen::Vector4f(r, g, b, a);
    splats.push_back(s);
  }
  return !splats.empty();
}

// ── Helpers ──────────────────────────────────────────────────────────────────

void extractPointsColors(const std::vector<GaussianSplat>& splats,
                         std::vector<Eigen::Vector3f>& pts,
                         std::vector<Eigen::Vector4f>& cols) {
  pts.clear(); cols.clear();
  pts.reserve(splats.size()); cols.reserve(splats.size());
  for (const auto& s : splats) { pts.push_back(s.position); cols.push_back(s.color_rgba); }
}

// Random rotation uniformly distributed over SO(3), capped at max_angle_deg.
Eigen::Matrix3f randomRotation(std::mt19937& gen, float max_angle_deg) {
  std::uniform_real_distribution<float> u01(0.0f, 1.0f);
  // Random axis via normal distribution
  std::normal_distribution<float> nd(0.0f, 1.0f);
  Eigen::Vector3f axis(nd(gen), nd(gen), nd(gen));
  if (axis.norm() < 1e-8f) axis = Eigen::Vector3f::UnitX();
  axis.normalize();

  float angle = u01(gen) * max_angle_deg * static_cast<float>(M_PI) / 180.0f;
  // Random sign
  if (u01(gen) < 0.5f) angle = -angle;

  return Eigen::AngleAxisf(angle, axis).toRotationMatrix();
}

// Per-axis uniform translation in [-max_t, max_t].
Eigen::Vector3f randomTranslation(std::mt19937& gen, float max_t) {
  std::uniform_real_distribution<float> ut(-max_t, max_t);
  return Eigen::Vector3f(ut(gen), ut(gen), ut(gen));
}

// Angle between two rotation matrices [degrees].
float rotationError(const Eigen::Matrix3f& R_est, const Eigen::Matrix3f& R_gt) {
  const Eigen::Matrix3f dR = R_est * R_gt.transpose();
  const float trace = std::max(-1.0f, std::min(3.0f, static_cast<float>(dR.trace())));
  return std::acos((trace - 1.0f) / 2.0f) * 180.0f / static_cast<float>(M_PI);
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
  // ONE seed per process invocation — avoids TBB/Eigen heap corruption across iterations.
  // The shell wrapper calls this binary in a loop over seeds.
  if (argc < 8) {
    std::cerr << "Usage: " << argv[0]
              << " src.ply tgt.ply voxel_size max_rot_deg max_trans_m seed output.csv [color_weight_icp] [run_icp=1]\n";
    return 1;
  }

  const std::string src_path       = argv[1];
  const std::string tgt_path       = argv[2];
  const float voxel_size           = std::stof(argv[3]);
  const float max_rot              = std::stof(argv[4]);
  const float max_trans            = std::stof(argv[5]);
  const int   seed                 = std::stoi(argv[6]);
  const std::string output_csv     = argv[7];
  const float color_weight_icp     = (argc >= 9) ? std::stof(argv[8]) : 1.0f;
  const bool  run_icp              = (argc >= 10) ? (std::stoi(argv[9]) != 0) : true;
  // Joint-5D vs centroid keypoint selection: runtime override via env var
  // USE_JOINT_5D=0/1 (default 1 = production joint-5D). Avoids recompiling
  // to switch downsampling mode between benchmark runs.
  const char* env_j5d              = std::getenv("USE_JOINT_5D");
  const bool  use_joint_5d         = env_j5d ? (std::atoi(env_j5d) != 0) : true;
  // USE_OPACITY=0/1 (default 0 = production opacity-off). Adds the alpha channel
  // to the descriptor without recompiling.
  const char* env_op               = std::getenv("USE_OPACITY");
  const bool  use_opacity          = env_op ? (std::atoi(env_op) != 0) : false;
  // SKIP_VANILLA=1 / SKIP_GEOM_GICP=1: drop the vanilla-KISS baseline and/or the
  // geometric-GICP stage, for winning-config-only runs at large seed counts.
  const char* env_sv               = std::getenv("SKIP_VANILLA");
  const bool  skip_vanilla         = env_sv && (std::atoi(env_sv) != 0);
  const char* env_sg               = std::getenv("SKIP_GEOM_GICP");
  const bool  skip_geom_gicp       = env_sg && (std::atoi(env_sg) != 0);
  const int   n_iters              = 1;
  const int   seed_start           = seed;

  // Extract <scene> and <split> from path.
  // Expected: .../submap_gaussians/<scene>/<split>/point_cloud/iteration_*/point_cloud.ply
  // From the end: [filename, iteration_dir, point_cloud_dir, split, scene]
  auto extractSceneAndSplit = [](const std::string& p) -> std::pair<std::string,std::string> {
    std::vector<size_t> sl;
    for (size_t i = 0; i < p.size(); ++i) if (p[i] == '/') sl.push_back(i);
    if (sl.size() < 5) return {"unknown", "unknown"};
    const size_t n = sl.size();
    return {
      p.substr(sl[n-5] + 1, sl[n-4] - sl[n-5] - 1),  // <scene>
      p.substr(sl[n-4] + 1, sl[n-3] - sl[n-4] - 1)   // <split>
    };
  };
  const auto [scene_name, split_src] = extractSceneAndSplit(src_path);
  const auto [scene_name2, split_tgt] = extractSceneAndSplit(tgt_path);
  (void)scene_name2;

  // Load clouds
  std::vector<GaussianSplat> src_splats, tgt_splats;
  std::cerr << "Loading " << src_path << " ... ";
  if (!loadPLY(src_path, src_splats)) { std::cerr << "FAILED\n"; return 1; }
  std::cerr << src_splats.size() << " pts\n";

  std::cerr << "Loading " << tgt_path << " ... ";
  if (!loadPLY(tgt_path, tgt_splats)) { std::cerr << "FAILED\n"; return 1; }
  std::cerr << tgt_splats.size() << " pts\n";

  std::vector<Eigen::Vector3f> src_pts, tgt_pts_orig;
  std::vector<Eigen::Vector4f> src_cols, tgt_cols_orig;
  extractPointsColors(src_splats, src_pts, src_cols);
  extractPointsColors(tgt_splats, tgt_pts_orig, tgt_cols_orig);

  // Open output CSV
  std::ofstream csv(output_csv, std::ios::app);
  if (!csv) { std::cerr << "Cannot open output: " << output_csv << "\n"; return 1; }

  const float ROT_THRESH   = 5.0f;   // degrees
  const float TRANS_THRESH = 1.0f;   // metres

  // Config for 3DGS-Matcher: color + colorPFH (no FPFH)
  gs3d_matcher::ColorKISSMatcherConfig color_cfg(
      voxel_size, true, false, 1.0f, 5000,
      /*color_weight=*/1.0f,
      /*use_f1=*/false, /*use_f2=*/false, /*use_f3=*/false,
      /*use_color=*/true, /*use_opacity=*/use_opacity, /*num_bins=*/11,
      /*use_centroid_voxel=*/true,
      /*use_joint_5d=*/use_joint_5d, /*voxel_size_ab=*/5.0f, /*min_color_bins=*/2,
      /*use_cross_color_filter=*/false, /*cross_color_min_overlap=*/0.1f,
      /*use_delta_e_hist=*/false, /*delta_e_radius=*/1.0f, /*delta_e_bins=*/11, /*delta_e_max=*/50.0f,
      /*use_color_pfh=*/true, /*color_pfh_radius=*/1.0f, /*color_pfh_bins=*/11, /*color_pfh_max_de=*/100.0f);

  // Config for vanilla KISS-Matcher
  kiss_matcher::KISSMatcherConfig kiss_cfg(voxel_size, true, false, 1.0f, 5000);

  // Count occupied voxels in src as a rough estimate of downsampled size.
  size_t n_ds_src = 0;
  {
    std::unordered_map<int64_t, int> vox;
    const float inv_v = 1.0f / voxel_size;
    for (const auto& p : src_pts) {
      int64_t k = (int64_t(static_cast<int>(p.x()*inv_v)) * 100003LL +
                   int64_t(static_cast<int>(p.y()*inv_v))) * 100003LL +
                  int64_t(static_cast<int>(p.z()*inv_v));
      vox[k]++;
    }
    n_ds_src = vox.size();
  }
  const size_t MIN_PTS = 200;
  std::cerr << "Estimated src downsampled size: ~" << n_ds_src << " pts\n";
  if (n_ds_src < MIN_PTS) {
    std::cerr << "ERROR: scene too sparse at voxel_size=" << voxel_size
              << " (~" << n_ds_src << " pts < " << MIN_PTS << " minimum).\n"
              << "  Use a smaller --voxel-size for this scene. Aborting.\n";
    return 2;
  }

  std::cerr << "Running " << n_iters << " iterations per method...\n";

  for (int iter = 0; iter < n_iters; ++iter) {
    const uint32_t seed = static_cast<uint32_t>(seed_start + iter);
    std::mt19937 gen(seed);

    // Generate random perturbation
    const Eigen::Matrix3f R_true = randomRotation(gen, max_rot);
    const Eigen::Vector3f t_true = randomTranslation(gen, max_trans);

    // Apply perturbation to tgt
    std::vector<Eigen::Vector3f> tgt_pts(tgt_pts_orig.size());
    for (size_t i = 0; i < tgt_pts_orig.size(); ++i)
      tgt_pts[i] = R_true * tgt_pts_orig[i] + t_true;

    struct MethodResult {
      float re = -1.f, te = -1.f;
      int success = -1, solver_ok = -1;
      Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    };

    auto run_method = [&](const std::string& name, auto fn) {
      auto t0 = std::chrono::steady_clock::now();
      try {
        MethodResult r = fn();
        double ms = std::chrono::duration<double,std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
        const Eigen::Quaterniond q(r.T.rotation());
        const Eigen::Vector3d    t = r.T.translation();
        csv << name << "," << scene_name << "," << split_src << "," << split_tgt
            << "," << iter << "," << seed
            << "," << r.re << "," << r.te << "," << r.success << "," << r.solver_ok
            << "," << t.x() << "," << t.y() << "," << t.z()
            << "," << q.w() << "," << q.x() << "," << q.y() << "," << q.z()
            << "," << ms << "\n";
        csv.flush();
      } catch (const std::exception& e) {
        double ms = std::chrono::duration<double,std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
        std::cerr << "  [" << name << " iter=" << iter << "] exception: " << e.what() << "\n";
        csv << name << "," << scene_name << "," << split_src << "," << split_tgt
            << "," << iter << "," << seed
            << ",-1,-1,-1,-1"
            << ",0,0,0,1,0,0,0"
            << "," << ms << "\n";
        csv.flush();
      }
    };

    // ── Method 1: vanilla KISS-Matcher ──────────────────────────────────────
    // Heap-allocate and intentionally leak: kiss_matcher::KISSMatcher's destructor
    // tries to free TBB-managed Eigen::VectorXf, corrupting the allocator for
    // subsequent calls. Same root cause as ColorKISSMatcher (which uses .release()).
    if (!skip_vanilla)
    run_method("vanilla_kiss", [&]() -> MethodResult {
      auto* km = new kiss_matcher::KISSMatcher(kiss_cfg);  // intentionally leaked
      auto sol = km->estimate(src_pts, tgt_pts);
      const int solver_ok = (km->getNumFinalInliers() > 0) ? 1 : 0;
      Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
      T.linear() = sol.rotation; T.translation() = sol.translation;
      Eigen::Matrix3f R_est = sol.rotation.cast<float>();
      Eigen::Vector3f t_est = sol.translation.cast<float>();
      float re = rotationError(R_est, R_true);
      float te = (t_est - t_true).norm();
      return {re, te, (re < ROT_THRESH && te < TRANS_THRESH) ? 1 : 0, solver_ok, T};
    });

    // ── Method 2: 3DGS-Matcher (color + colorPFH) ───────────────────────────
    // Capture the result so ICP methods can warm-start from it.
    Eigen::Isometry3d color_cpfh_T = Eigen::Isometry3d::Identity();
    run_method("color_cpfh", [&]() -> MethodResult {
      gs3d_matcher::ColorKISSMatcher cm(color_cfg);
      auto sol = cm.estimate(src_pts, tgt_pts, src_cols, tgt_cols_orig);
      color_cpfh_T.linear()      = sol.rotation;
      color_cpfh_T.translation() = sol.translation;
      const int solver_ok = (cm.getNumInliers() > 0) ? 1 : 0;
      Eigen::Matrix3f R_est = sol.rotation.cast<float>();
      Eigen::Vector3f t_est = sol.translation.cast<float>();
      float re = rotationError(R_est, R_true);
      float te = (t_est - t_true).norm();
      return {re, te, (re < ROT_THRESH && te < TRANS_THRESH) ? 1 : 0, solver_ok, color_cpfh_T};
    });

    // ── Method 3: small_gicp GICP warm-started from color_cpfh ──────────────
    if (!run_icp) continue;
    if (!skip_geom_gicp)
    run_method("gicp_cpfh", [&]() -> MethodResult {
      small_gicp::RegistrationSetting gicp_cfg;
      gicp_cfg.type = small_gicp::RegistrationSetting::GICP;
      gicp_cfg.downsampling_resolution = static_cast<double>(voxel_size);
      gicp_cfg.max_correspondence_distance = static_cast<double>(voxel_size) * 3.0;
      gicp_cfg.num_threads = 4;
      gicp_cfg.max_iterations = 50;
      auto result = small_gicp::align(tgt_pts, src_pts, color_cpfh_T, gicp_cfg);
      Eigen::Matrix3f R_est = result.T_target_source.rotation().cast<float>();
      Eigen::Vector3f t_est = result.T_target_source.translation().cast<float>();
      float re = rotationError(R_est, R_true);
      float te = (t_est - t_true).norm();
      return {re, te, (re < ROT_THRESH && te < TRANS_THRESH) ? 1 : 0,
              result.converged ? 1 : 0, result.T_target_source};
    });

    // ── Method 4: ColorPFH-weighted GICP (correct color fusion) ─────────────
    // Geometric correspondences (Kd-tree NN), but each match's information
    // matrix is weighted by color agreement w = exp(-||dc||^2 / 2 sigma^2), so
    // color enters H and b (fixing the no-op bug of the old color_gicp_cpfh).
    // sigma is the RGB color scale (0.15); tune via ColorWeightedGICPFactor.
    run_method("color_gicp_de", [&]() -> MethodResult {
      auto result = small_gicp::align_colored_simple(
          tgt_pts, tgt_cols_orig,
          src_pts, src_cols,
          static_cast<double>(voxel_size),
          /*color_weight=sigma; <=0 => auto from color std=*/-1.0,
          /*registration_type=*/"GICP_DEW",
          /*init_T=*/color_cpfh_T);
      Eigen::Matrix3f R_est = result.T_target_source.rotation().cast<float>();
      Eigen::Vector3f t_est = result.T_target_source.translation().cast<float>();
      float re = rotationError(R_est, R_true);
      float te = (t_est - t_true).norm();
      return {re, te, (re < ROT_THRESH && te < TRANS_THRESH) ? 1 : 0,
              result.converged ? 1 : 0, result.T_target_source};
    });

    // ── Dropped methods (kept for reference) ────────────────────────────────
    // color_gicp_cpfh: color only in scalar loss -> no-op (== coarse matcher).
    // color_corr_icp / color_full_icp: color in hard kNN selection -> diverges
    // at trained_gaussians scale (5-6% success, ~3 s/run). See sweep analysis.
    // Re-enable by uncommenting if needed for the paper's 3DGS-ICP comparison.
#if 0
    run_method("color_gicp_cpfh", [&]() -> MethodResult {
      auto result = small_gicp::align_colored_simple(
          tgt_pts, tgt_cols_orig,
          src_pts, src_cols,
          static_cast<double>(voxel_size),
          static_cast<double>(color_weight_icp),
          /*registration_type=*/"GICP",
          /*init_T=*/color_cpfh_T);
      Eigen::Matrix3f R_est = result.T_target_source.rotation().cast<float>();
      Eigen::Vector3f t_est = result.T_target_source.translation().cast<float>();
      float re = rotationError(R_est, R_true);
      float te = (t_est - t_true).norm();
      return {re, te, (re < ROT_THRESH && te < TRANS_THRESH) ? 1 : 0,
              result.converged ? 1 : 0, result.T_target_source};
    });
    run_method("color_corr_icp", [&]() -> MethodResult {
      gs3d_matcher::MultimodalICPConfig cfg;
      cfg.color_scale             = static_cast<double>(color_weight_icp);
      cfg.downsampling_resolution = static_cast<double>(voxel_size);
      auto res = gs3d_matcher::align_color_correspondence(
          src_pts, src_cols, tgt_pts, tgt_cols_orig, cfg, color_cpfh_T);
      Eigen::Matrix3f R_est = res.T.rotation().cast<float>();
      Eigen::Vector3f t_est = res.T.translation().cast<float>();
      float re = rotationError(R_est, R_true);
      float te = (t_est - t_true).norm();
      return {re, te, (re < ROT_THRESH && te < TRANS_THRESH) ? 1 : 0,
              res.converged ? 1 : 0, res.T};
    });
    run_method("color_full_icp", [&]() -> MethodResult {
      gs3d_matcher::MultimodalICPConfig cfg;
      cfg.color_scale             = static_cast<double>(color_weight_icp);
      cfg.alpha                   = 5.0;
      cfg.downsampling_resolution = static_cast<double>(voxel_size);
      auto res = gs3d_matcher::align_color_full(
          src_pts, src_cols, tgt_pts, tgt_cols_orig, cfg, color_cpfh_T);
      Eigen::Matrix3f R_est = res.T.rotation().cast<float>();
      Eigen::Vector3f t_est = res.T.translation().cast<float>();
      float re = rotationError(R_est, R_true);
      float te = (t_est - t_true).norm();
      return {re, te, (re < ROT_THRESH && te < TRANS_THRESH) ? 1 : 0,
              res.converged ? 1 : 0, res.T};
    });
#endif

    if ((iter + 1) % 10 == 0)
      std::cerr << "  iter " << (iter+1) << "/" << n_iters << "\n";
  }

  csv.flush();
  std::cerr << "Done. Output: " << output_csv << "\n";
  return 0;
}

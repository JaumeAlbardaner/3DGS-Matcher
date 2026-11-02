// Budget-sensitivity control for Table II's refinement rows.
//
// Re-runs geometric GICP (the exact small_gicp::align path benchmark_submap
// uses) warm-started from the coarse poses STORED in a benchmark CSV, under a
// configurable (max_corr, max_iter) budget. Purpose: quantify how much of the
// GICP-vs-3DGS-ICP gap is attributable to their differing budgets
// (GICP: 3x voxel / 50 iters vs 3DGS-ICP library defaults: 1 m / 20 iters).
//
// The per-seed perturbation replays benchmark_submap's RNG bit-exactly and is
// validated against the CSV's stored coarse errors before running.
//
// Usage:
//   gicp_budget_control <src.ply> <tgt.ply> <benchmark.csv> <scene>
//                       <src_split> <tgt_split> <voxel> <max_corr_m>
//                       <max_iter> <out.csv>
// Only CSV rows whose split_src/split_tgt match are used, so each of a
// scene's two pairs must be run against its own clouds.
#include <small_gicp/registration/registration_helper.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// ── minimal binary-LE PLY xyz loader (same layout as benchmark_submap) ──────
static bool loadPlyXYZ(const std::string& path, std::vector<Eigen::Vector3f>& pts) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::string line;
  std::getline(f, line);
  if (line != "ply") return false;
  size_t n = 0;
  bool in_vertex = false;
  std::vector<std::pair<std::string, int>> props;  // (name, byte size)
  auto tsize = [](const std::string& t) -> int {
    if (t == "float" || t == "float32" || t == "int" || t == "int32" || t == "uint" || t == "uint32") return 4;
    if (t == "double" || t == "float64") return 8;
    if (t == "short" || t == "int16" || t == "ushort" || t == "uint16") return 2;
    return 1;
  };
  while (std::getline(f, line)) {
    std::istringstream ss(line);
    std::string tok; ss >> tok;
    if (tok == "end_header") break;
    if (tok == "element") { std::string name; ss >> name; in_vertex = (name == "vertex"); if (in_vertex) ss >> n; }
    else if (tok == "property" && in_vertex) { std::string t, name; ss >> t >> name; props.push_back({name, tsize(t)}); }
  }
  int stride = 0, ox = -1, oy = -1, oz = -1;
  for (auto& [name, sz] : props) {
    if (name == "x") ox = stride; if (name == "y") oy = stride; if (name == "z") oz = stride;
    stride += sz;
  }
  if (ox < 0 || oy < 0 || oz < 0) return false;
  std::vector<char> buf(stride);
  pts.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    f.read(buf.data(), stride);
    float x, y, z;
    std::memcpy(&x, buf.data() + ox, 4); std::memcpy(&y, buf.data() + oy, 4); std::memcpy(&z, buf.data() + oz, 4);
    pts.emplace_back(x, y, z);
  }
  return !pts.empty();
}

// ── perturbation replay: MUST stay identical to benchmark_submap.cc ─────────
static Eigen::Matrix3f randomRotation(std::mt19937& gen, float max_angle_deg) {
  std::uniform_real_distribution<float> u01(0.0f, 1.0f);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  Eigen::Vector3f axis(nd(gen), nd(gen), nd(gen));
  if (axis.norm() < 1e-8f) axis = Eigen::Vector3f::UnitX();
  axis.normalize();
  float angle = u01(gen) * max_angle_deg * static_cast<float>(M_PI) / 180.0f;
  if (u01(gen) < 0.5f) angle = -angle;
  return Eigen::AngleAxisf(angle, axis).toRotationMatrix();
}
static Eigen::Vector3f randomTranslation(std::mt19937& gen, float max_t) {
  std::uniform_real_distribution<float> ut(-max_t, max_t);
  return Eigen::Vector3f(ut(gen), ut(gen), ut(gen));
}
static float rotationError(const Eigen::Matrix3f& A, const Eigen::Matrix3f& B) {
  const float tr = std::max(-1.0f, std::min(3.0f, (A * B.transpose()).trace()));
  return std::acos((tr - 1.0f) / 2.0f) * 180.0f / static_cast<float>(M_PI);
}

struct CoarseRow { int seed; Eigen::Matrix3f R; Eigen::Vector3f t; float re, te; };

int main(int argc, char** argv) {
  if (argc != 11) {
    std::cerr << "Usage: " << argv[0]
              << " src.ply tgt.ply benchmark.csv scene src_split tgt_split voxel max_corr_m max_iter out.csv\n";
    return 1;
  }
  const std::string src_path = argv[1], tgt_path = argv[2], csv_path = argv[3], scene = argv[4];
  const std::string src_split = argv[5], tgt_split = argv[6];
  const float voxel = std::stof(argv[7]);
  const double max_corr = std::stod(argv[8]);
  const int max_iter = std::stoi(argv[9]);
  const std::string out_path = argv[10];

  std::vector<Eigen::Vector3f> src, tgt;
  if (!loadPlyXYZ(src_path, src)) { std::cerr << "bad src\n"; return 1; }
  if (!loadPlyXYZ(tgt_path, tgt)) { std::cerr << "bad tgt\n"; return 1; }
  std::cerr << "[" << scene << ":" << src_split << "->" << tgt_split
            << "] src=" << src.size() << " tgt=" << tgt.size() << " pts\n";

  // coarse (color_cpfh) rows for this scene
  std::vector<CoarseRow> rows;
  {
    std::ifstream cf(csv_path);
    std::string line;
    std::getline(cf, line);  // header
    while (std::getline(cf, line)) {
      std::vector<std::string> c;
      std::stringstream ss(line);
      std::string tok;
      while (std::getline(ss, tok, ',')) c.push_back(tok);
      if (c.size() < 18 || c[0] != "color_cpfh" || c[1] != scene ||
          c[2] != src_split || c[3] != tgt_split) continue;
      CoarseRow r;
      r.seed = std::stoi(c[5]);
      r.re = std::stof(c[6]); r.te = std::stof(c[7]);
      const float qw = std::stof(c[13]), qx = std::stof(c[14]), qy = std::stof(c[15]), qz = std::stof(c[16]);
      r.R = Eigen::Quaternionf(qw, qx, qy, qz).toRotationMatrix();
      r.t = Eigen::Vector3f(std::stof(c[10]), std::stof(c[11]), std::stof(c[12]));
      rows.push_back(r);
    }
  }
  if (rows.empty()) { std::cerr << "no color_cpfh rows for scene " << scene << "\n"; return 1; }

  std::ofstream out(out_path, std::ios::app);
  const bool fresh = out.tellp() == 0;
  if (fresh)
    out << "method,scene,split_src,split_tgt,corr_m,max_iter,seed,rot_error_deg,trans_error_m,success,converged,time_ms\n";

  small_gicp::RegistrationSetting cfg;
  cfg.type = small_gicp::RegistrationSetting::GICP;
  cfg.downsampling_resolution = voxel;
  cfg.max_correspondence_distance = max_corr;
  cfg.num_threads = 4;
  cfg.max_iterations = max_iter;

  int n_ok = 0, n_val = 0;
  for (const auto& r : rows) {
    std::mt19937 gen(static_cast<uint32_t>(r.seed));
    const Eigen::Matrix3f R_true = randomRotation(gen, 180.0f);
    const Eigen::Vector3f t_true = randomTranslation(gen, 500.0f);

    // protocol validation on the first rows: stored coarse errors must match
    if (n_val < 3) {
      const float re = rotationError(r.R, R_true), te = (r.t - t_true).norm();
      if (std::abs(re - r.re) > 0.05f || std::abs(te - r.te) > 0.05f) {
        std::cerr << "PROTOCOL MISMATCH seed " << r.seed << ": " << re << "/" << te
                  << " vs stored " << r.re << "/" << r.te << "\n";
        return 2;
      }
      ++n_val;
    }

    std::vector<Eigen::Vector3f> tgt_p(tgt.size());
    for (size_t i = 0; i < tgt.size(); ++i) tgt_p[i] = R_true * tgt[i] + t_true;

    Eigen::Isometry3d init = Eigen::Isometry3d::Identity();
    init.linear() = r.R.cast<double>();
    init.translation() = r.t.cast<double>();

    const auto t0 = std::chrono::steady_clock::now();
    const auto res = small_gicp::align(tgt_p, src, init, cfg);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    const Eigen::Matrix3f R_est = res.T_target_source.rotation().cast<float>();
    const Eigen::Vector3f t_est = res.T_target_source.translation().cast<float>();
    const float re = rotationError(R_est, R_true), te = (t_est - t_true).norm();
    const int succ = (re < 5.0f && te < 0.5f) ? 1 : 0;
    n_ok += succ;
    out << "gicp_budget," << scene << "," << src_split << "," << tgt_split << ","
        << max_corr << "," << max_iter << "," << r.seed << ","
        << re << "," << te << "," << succ << "," << (res.converged ? 1 : 0) << "," << ms << "\n";
    out.flush();
  }
  std::cerr << "[" << scene << ":" << src_split << "->" << tgt_split
            << "] corr=" << max_corr << " iter=" << max_iter
            << ": " << n_ok << "/" << rows.size() << " success\n";
  return 0;
}

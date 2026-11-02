#include <3dgs_matcher/ColorKISSMatcher.hpp>
#ifdef COLOR_MATCHER_HAS_GICP
#include <small_gicp/registration/registration_helper.hpp>
#include <3dgs_matcher/small_gicp_color/color_registration.hpp>
#endif

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef COLOR_MATCHER_USE_PCL_VIZ
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#endif

namespace {

struct GaussianSplat {
  Eigen::Vector3f position;
  Eigen::Vector4f color_rgba;  // normalized [0,1]
};

// ── .splat loader ────────────────────────────────────────────────────────────

bool loadSplatFile(const std::string& filename, std::vector<GaussianSplat>& splats) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filename << "\n";
    return false;
  }

  file.seekg(0, std::ios::end);
  const size_t file_size = static_cast<size_t>(file.tellg());
  file.seekg(0, std::ios::beg);

  if (file_size % 32 != 0) {
    std::cerr << "Unsupported .splat format in: " << filename << "\n";
    return false;
  }

  const size_t num_splats = file_size / 32;
  splats.clear();
  splats.reserve(num_splats);

  for (size_t i = 0; i < num_splats; ++i) {
    float data[6];
    file.read(reinterpret_cast<char*>(data), sizeof(float) * 6);
    if (!file) return false;

    uint8_t rgba[4];
    file.read(reinterpret_cast<char*>(rgba), 4);
    if (!file) return false;

    uint8_t rot[4];
    file.read(reinterpret_cast<char*>(rot), 4);
    if (!file) return false;

    GaussianSplat s;
    s.position = Eigen::Vector3f(data[0], data[1], data[2]);
    s.color_rgba = Eigen::Vector4f(
        rgba[0] / 255.0f, rgba[1] / 255.0f, rgba[2] / 255.0f, rgba[3] / 255.0f);
    splats.push_back(s);
  }

  return true;
}

// ── .ply loader (3DGS format: f_dc_0/1/2 + opacity) ─────────────────────────

namespace ply {

struct Property {
  std::string type;
  std::string name;
  bool is_list = false;
  std::string list_count_type;
  std::string list_value_type;
};

inline std::string toLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

template <typename T>
T readBin(std::ifstream& f) {
  T v{};
  f.read(reinterpret_cast<char*>(&v), sizeof(T));
  return v;
}

double readScalar(std::ifstream& f, const std::string& type) {
  const std::string t = toLower(type);
  if (t == "char"   || t == "int8")    return readBin<int8_t>(f);
  if (t == "uchar"  || t == "uint8")   return readBin<uint8_t>(f);
  if (t == "short"  || t == "int16")   return readBin<int16_t>(f);
  if (t == "ushort" || t == "uint16")  return readBin<uint16_t>(f);
  if (t == "int"    || t == "int32")   return readBin<int32_t>(f);
  if (t == "uint"   || t == "uint32")  return readBin<uint32_t>(f);
  if (t == "float"  || t == "float32") return readBin<float>(f);
  if (t == "double" || t == "float64") return readBin<double>(f);
  throw std::runtime_error("Unknown PLY type: " + type);
}

}  // namespace ply

bool loadPlyFile(const std::string& filename, std::vector<GaussianSplat>& splats) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filename << "\n";
    return false;
  }

  std::string line;
  if (!std::getline(file, line) || ply::toLower(line) != "ply") {
    std::cerr << "Not a valid PLY file: " << filename << "\n";
    return false;
  }

  bool ascii = false, binary_le = false;
  bool in_vertex = false;
  size_t vertex_count = 0;
  std::vector<ply::Property> props;

  while (std::getline(file, line)) {
    if (line == "end_header") break;
    std::istringstream iss(line);
    std::string tok;
    iss >> tok;
    tok = ply::toLower(tok);
    if (tok == "format") {
      std::string fmt; iss >> fmt;
      ascii     = (ply::toLower(fmt) == "ascii");
      binary_le = (ply::toLower(fmt) == "binary_little_endian");
    } else if (tok == "element") {
      std::string name; size_t cnt = 0;
      iss >> name >> cnt;
      in_vertex = (ply::toLower(name) == "vertex");
      if (in_vertex) vertex_count = cnt;
    } else if (tok == "property" && in_vertex) {
      ply::Property p;
      std::string next; iss >> next;
      if (ply::toLower(next) == "list") {
        p.is_list = true;
        iss >> p.list_count_type >> p.list_value_type >> p.name;
      } else {
        p.type = next;
        iss >> p.name;
      }
      p.name = ply::toLower(p.name);
      props.push_back(p);
    }
  }

  if (vertex_count == 0 || props.empty()) {
    std::cerr << "PLY file has no vertex data: " << filename << "\n";
    return false;
  }

  std::unordered_map<std::string, size_t> idx;
  for (size_t i = 0; i < props.size(); ++i) idx[props[i].name] = i;

  auto find = [&](const std::string& n) -> int {
    auto it = idx.find(n);
    return it == idx.end() ? -1 : static_cast<int>(it->second);
  };

  const int ix = find("x"), iy = find("y"), iz = find("z");
  if (ix < 0 || iy < 0 || iz < 0) {
    std::cerr << "PLY missing x/y/z: " << filename << "\n";
    return false;
  }

  const int ir = find("red"), ig = find("green"), ib = find("blue"), ia = find("alpha");
  const int idc0 = find("f_dc_0"), idc1 = find("f_dc_1"), idc2 = find("f_dc_2");
  const int iop  = find("opacity");

  constexpr double kSHC0 = 0.28209479177387814;
  auto clamp01 = [](float v) { return std::max(0.0f, std::min(1.0f, v)); };

  splats.clear();
  splats.reserve(vertex_count);

  for (size_t vi = 0; vi < vertex_count; ++vi) {
    std::vector<double> vals(props.size(), 0.0);

    if (ascii) {
      if (!std::getline(file, line)) break;
      std::istringstream iss(line);
      for (size_t pi = 0; pi < props.size(); ++pi) {
        if (props[pi].is_list) {
          int cnt = 0; iss >> cnt;
          for (int k = 0; k < cnt; ++k) { double d; iss >> d; }
        } else {
          iss >> vals[pi];
        }
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
    s.position = Eigen::Vector3f(
        static_cast<float>(vals[ix]),
        static_cast<float>(vals[iy]),
        static_cast<float>(vals[iz]));

    float r = 0.5f, g = 0.5f, b = 0.5f;
    if (ir >= 0 && ig >= 0 && ib >= 0) {
      // Plain RGB — normalize from [0,255] if needed
      r = clamp01(static_cast<float>(ir >= 0 ? (vals[ir] > 1.0 ? vals[ir]/255.0 : vals[ir]) : 0.5));
      g = clamp01(static_cast<float>(ig >= 0 ? (vals[ig] > 1.0 ? vals[ig]/255.0 : vals[ig]) : 0.5));
      b = clamp01(static_cast<float>(ib >= 0 ? (vals[ib] > 1.0 ? vals[ib]/255.0 : vals[ib]) : 0.5));
    } else if (idc0 >= 0 && idc1 >= 0 && idc2 >= 0) {
      // 3DGS spherical-harmonics DC component → RGB
      r = clamp01(static_cast<float>(kSHC0 * vals[idc0] + 0.5));
      g = clamp01(static_cast<float>(kSHC0 * vals[idc1] + 0.5));
      b = clamp01(static_cast<float>(kSHC0 * vals[idc2] + 0.5));
    }

    float a = 1.0f;
    if (ia >= 0) {
      a = clamp01(static_cast<float>(vals[ia] > 1.0 ? vals[ia]/255.0 : vals[ia]));
    } else if (iop >= 0) {
      // Opacity is stored as logit in 3DGS PLY
      const double v = vals[iop];
      a = (v >= 0.0 && v <= 1.0) ? static_cast<float>(v)
                                  : clamp01(static_cast<float>(1.0 / (1.0 + std::exp(-v))));
    }

    s.color_rgba = Eigen::Vector4f(r, g, b, a);
    splats.push_back(s);
  }

  return !splats.empty();
}

bool loadFile(const std::string& filename, std::vector<GaussianSplat>& splats) {
  const size_t dot = filename.rfind('.');
  const std::string ext = (dot == std::string::npos) ? "" : filename.substr(dot);
  std::string ext_lower = ext;
  std::transform(ext_lower.begin(), ext_lower.end(), ext_lower.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (ext_lower == ".ply") return loadPlyFile(filename, splats);
  return loadSplatFile(filename, splats);
}

// ── helpers ──────────────────────────────────────────────────────────────────

void extractPointsAndColors(const std::vector<GaussianSplat>& splats,
                            float min_alpha,
                            std::vector<Eigen::Vector3f>& points,
                            std::vector<Eigen::Vector4f>& colors) {
  points.clear();
  colors.clear();
  points.reserve(splats.size());
  colors.reserve(splats.size());

  for (const auto& s : splats) {
    if (s.color_rgba.w() < min_alpha) continue;
    points.push_back(s.position);
    colors.push_back(s.color_rgba);
  }
}

void printUsage(const char* prog) {
  std::cout
      << "Usage:\n"
      << "  " << prog << " <source.splat|source.ply> <target.splat|target.ply> [options]\n\n"
      << "Defaults reproduce the benchmark's best pipeline: 3DGS-Matcher coarse\n"
      << "(ColorPFH + raw colour, FPFH off, joint-5D) at 0.5 m + colour-weighted GICP.\n\n"
      << "Options:\n"
      << "  --voxel-size <float>    Voxel size (default: 0.5)\n"
      << "  --refine <mode>         Fine registration: none | gicp | color-gicp (default: color-gicp)\n"
      << "  --no-refine             Coarse only (shorthand for --refine none)\n"
      << "  --num-max-corr <int>    Max ROBIN correspondences (default: 5000)\n"
      << "  --min-alpha <0..1>      Keep splats with alpha >= threshold (default: 0.0)\n"
      << "  --color-weight <float>  Color weight (default: 1.0)\n"
      << "  --f1 <0|1>              Use FPFH sub-feature 1 (default: 0)\n"
      << "  --f2 <0|1>              Use FPFH sub-feature 2 (default: 0)\n"
      << "  --f3 <0|1>              Use FPFH sub-feature 3 (default: 0)\n"
      << "  --color <0|1>           Use raw colour feature (default: 1)\n"
      << "  --color-pfh             Use ColorPFH (relative CIELAB) descriptor (default: ON)\n"
      << "  --opacity <0|1>         Use alpha feature (default: 0)\n"
      << "  --bins <int>            FPFH bins (default: 11)\n"
      << "  --centroid <0|1>        Centroid voxel sampling (default: 1)\n"
      << "  --no-downsample         Disable voxel downsampling (use all points)\n"
      << "  --joint-5d              Joint spatial+color voxelization (default: ON)\n"
      << "  --no-joint-5d           Disable joint-5D (use plain/centroid voxelization)\n"
      << "  --voxel-ab <float>      Color voxel size in LAB ab units (default: 5.0)\n"
      << "  --min-color-bins <int>  Min distinct colors per spatial voxel (default: 2)\n"
      << "  --viz                   Show split before/after viewer (requires PCL viz)\n"
      << "  --animate               Animate source converging onto target (KISS-Matcher style)\n"
      << "  --animate-frames <int>  Frames in the convergence ramp (default: 150)\n"
      << "  --animate-hold <int>    Frames held at start/end when recording (default: 30)\n"
      << "  --animate-save <dir>    Record one pass to PNG frames in <dir> (press 'r' in viewer)\n"
      << "  --animate-auto          Record a pass immediately and exit (headless; needs --animate-save)\n"
      << "  --animate-save-real <dir>  With --animate-auto: also record the same result in real colours\n"
      << "  --animate-real-colors   Use real 3DGS colors instead of orange/cyan per-cloud\n"
      << "  --animate-bg-white      White background (default: dark grey)\n"
      << "  --animate-zoom <float>  Auto-frame zoom factor, >1 = closer (default: 1.0)\n"
      << "  --animate-cam <9 floats>  Explicit camera: posX posY posZ  lookX lookY lookZ  upX upY upZ\n"
      << "  --animate-fovy <rad>    Vertical field of view in radians (with --animate-cam)\n"
      << "  --animate-max-points <int>  Cap displayed points per cloud for clarity (0 = all)\n"
      << "  --animate-frame-target  Frame the target cloud so the source flies into view\n"
      << "  --cross-color-filter    Keep only points in color bins shared by both clouds\n"
      << "  --cross-color-min-overlap <f> Min overlap ratio per color bin (default: 0.1)\n"
      << "  --delta-e-hist          Append ΔEab neighborhood histogram to each descriptor\n"
      << "  --delta-e-radius <f>    Spatial search radius for ΔEab neighbors (default: 1.0)\n"
      << "  --delta-e-bins <int>    Histogram bins for ΔEab (default: 11)\n"
      << "  --delta-e-max <f>       Max ΔEab value mapped to last bin (default: 50.0)\n"
      << "  --retry <int>           Re-run coarse up to N times until valid (default: 1)\n"
      << "  --perturb-rotation <deg>  Apply random rotation +/-deg to source before matching\n"
      << "  --perturb-translation <m> Apply random translation +/-m to source before matching\n"
      << "  --seed <int>            RNG seed for perturbation (default: 42)\n";
}

// ── visualization ─────────────────────────────────────────────────────────────

#ifdef COLOR_MATCHER_USE_PCL_VIZ
void visualizeBeforeAfter(const std::vector<Eigen::Vector3f>& src_points,
                          const std::vector<Eigen::Vector3f>& tgt_points,
                          const std::vector<Eigen::Vector3f>& src_aligned,
                          const std::vector<Eigen::Vector3f>& src_selected,
                          const std::vector<Eigen::Vector4f>& src_selected_colors,
                          const std::vector<Eigen::Vector3f>& tgt_selected,
                          const std::vector<Eigen::Vector4f>& tgt_selected_colors) {
  using CloudXYZ  = pcl::PointCloud<pcl::PointXYZ>;
  using CloudRGB  = pcl::PointCloud<pcl::PointXYZRGB>;

  auto toXYZ = [](const std::vector<Eigen::Vector3f>& pts) {
    CloudXYZ::Ptr c(new CloudXYZ);
    c->reserve(pts.size());
    for (const auto& p : pts) c->push_back(pcl::PointXYZ(p.x(), p.y(), p.z()));
    return c;
  };

  auto toRGB = [](const std::vector<Eigen::Vector3f>& pts,
                  const std::vector<Eigen::Vector4f>& cols) {
    CloudRGB::Ptr c(new CloudRGB);
    c->reserve(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
      pcl::PointXYZRGB p;
      p.x = pts[i].x(); p.y = pts[i].y(); p.z = pts[i].z();
      p.r = static_cast<uint8_t>(std::min(1.0f, cols[i][0]) * 255.0f);
      p.g = static_cast<uint8_t>(std::min(1.0f, cols[i][1]) * 255.0f);
      p.b = static_cast<uint8_t>(std::min(1.0f, cols[i][2]) * 255.0f);
      c->push_back(p);
    }
    return c;
  };

  auto src_cloud = toXYZ(src_points);
  auto tgt_cloud = toXYZ(tgt_points);
  auto aln_cloud = toXYZ(src_aligned);
  auto src_sel   = toRGB(src_selected, src_selected_colors);
  auto tgt_sel   = toRGB(tgt_selected, tgt_selected_colors);

  pcl::visualization::PCLVisualizer::Ptr viewer(
      new pcl::visualization::PCLVisualizer("Before | After | Selected points"));
  viewer->setBackgroundColor(0.07, 0.07, 0.07);

  int vp_before = 0, vp_after = 1, vp_sel = 2;
  viewer->createViewPort(0.0,       0.0, 0.3333, 1.0, vp_before);
  viewer->createViewPort(0.3333,    0.0, 0.6667, 1.0, vp_after);
  viewer->createViewPort(0.6667,    0.0, 1.0,    1.0, vp_sel);
  viewer->setBackgroundColor(0.07, 0.07, 0.07, vp_before);
  viewer->setBackgroundColor(0.07, 0.07, 0.07, vp_after);
  viewer->setBackgroundColor(0.07, 0.07, 0.07, vp_sel);

  viewer->addText("Before alignment",  10, 30, 14, 1.0, 1.0, 1.0, "lbl_b",  vp_before);
  viewer->addText("After alignment",   10, 30, 14, 1.0, 1.0, 1.0, "lbl_a",  vp_after);
  viewer->addText("Selected points",   10, 30, 14, 1.0, 1.0, 1.0, "lbl_s",  vp_sel);
  viewer->addText("red=source  green=target",              10, 10, 11, 0.7, 0.7, 0.7, "leg_b", vp_before);
  viewer->addText("blue=aligned src  green=target",        10, 10, 11, 0.7, 0.7, 0.7, "leg_a", vp_after);
  viewer->addText("real colors  src=large  tgt=small",     10, 10, 11, 0.7, 0.7, 0.7, "leg_s", vp_sel);

  viewer->addCoordinateSystem(1.0, "cs_b", vp_before);
  viewer->addCoordinateSystem(1.0, "cs_a", vp_after);
  viewer->addCoordinateSystem(1.0, "cs_s", vp_sel);

  // ── Before: source red + target green ────────────────────────────────────
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
      h_src_b(src_cloud, 220, 70, 70),
      h_tgt_b(tgt_cloud,  70, 220, 70);
  viewer->addPointCloud<pcl::PointXYZ>(src_cloud, h_src_b, "src_b", vp_before);
  viewer->addPointCloud<pcl::PointXYZ>(tgt_cloud, h_tgt_b, "tgt_b", vp_before);
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "src_b");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "tgt_b");

  // ── After: aligned source blue + target green ─────────────────────────────
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
      h_aln(aln_cloud, 80, 140, 255),
      h_tgt_a(tgt_cloud, 70, 220, 70);
  viewer->addPointCloud<pcl::PointXYZ>(aln_cloud, h_aln,   "aln_a", vp_after);
  viewer->addPointCloud<pcl::PointXYZ>(tgt_cloud, h_tgt_a, "tgt_a", vp_after);
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "aln_a");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "tgt_a");

  // ── Selected: real colors, src larger so both are distinguishable ─────────
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>
      h_src_sel(src_sel), h_tgt_sel(tgt_sel);
  viewer->addPointCloud<pcl::PointXYZRGB>(src_sel, h_src_sel, "src_s", vp_sel);
  viewer->addPointCloud<pcl::PointXYZRGB>(tgt_sel, h_tgt_sel, "tgt_s", vp_sel);
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "src_s");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "tgt_s");

  std::cout << "Viewer: left=before | centre=after | right=selected points (real colors).\n"
            << "src selected: " << src_selected.size()
            << "  tgt selected: " << tgt_selected.size() << "\n"
            << "Press 'q' to quit.\n";

  while (!viewer->wasStopped()) {
    viewer->spinOnce(16);
  }
}

// ── animation: source converges onto target (KISS-Matcher style) ──────────────

struct AnimState {
  bool playing = true;       // toggled with space
  bool record_request = false;  // set with 'r'
};

void animKeyCb(const pcl::visualization::KeyboardEvent& e, void* cookie) {
  if (!e.keyDown()) return;
  auto* st = static_cast<AnimState*>(cookie);
  const std::string k = e.getKeySym();
  if (k == "space") st->playing = !st->playing;
  else if (k == "r" || k == "R") st->record_request = true;
}

void animateAlignment(const std::vector<Eigen::Vector3f>& src_points,
                      const std::vector<Eigen::Vector4f>& src_colors,
                      const std::vector<Eigen::Vector3f>& tgt_points,
                      const std::vector<Eigen::Vector4f>& tgt_colors,
                      const Eigen::Matrix3f& R_final,
                      const Eigen::Vector3f& t_final,
                      int ramp_frames, int hold_frames,
                      const std::string& save_dir,
                      bool auto_record, bool real_colors,
                      bool white_bg, float zoom,
                      const std::vector<double>& camera, float fovy,
                      bool frame_target,
                      const std::string& save_dir_real = "") {
  using CloudRGB = pcl::PointCloud<pcl::PointXYZRGB>;
  namespace fs = std::filesystem;

  // Either real 3DGS colors, or a flat per-cloud color so the two clouds stay
  // visually separable as they converge (KISS-Matcher style).
  auto toRGB = [&](const std::vector<Eigen::Vector3f>& pts,
                   const std::vector<Eigen::Vector4f>& cols,
                   uint8_t fr, uint8_t fg, uint8_t fb) {
    CloudRGB::Ptr c(new CloudRGB);
    c->resize(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
      auto& p = (*c)[i];
      p.x = pts[i].x(); p.y = pts[i].y(); p.z = pts[i].z();
      if (real_colors) {
        p.r = static_cast<uint8_t>(std::min(1.0f, cols[i][0]) * 255.0f);
        p.g = static_cast<uint8_t>(std::min(1.0f, cols[i][1]) * 255.0f);
        p.b = static_cast<uint8_t>(std::min(1.0f, cols[i][2]) * 255.0f);
      } else {
        p.r = fr; p.g = fg; p.b = fb;
      }
    }
    return c;
  };

  CloudRGB::Ptr tgt = toRGB(tgt_points, tgt_colors,  60, 200, 255);  // cyan
  CloudRGB::Ptr src = toRGB(src_points, src_colors, 255, 140,  30);  // orange

  // Rotate about the source centroid so it spins in place rather than orbiting
  // the world origin, while its centroid travels straight to the aligned spot.
  Eigen::Vector3f c0 = Eigen::Vector3f::Zero();
  for (const auto& p : src_points) c0 += p;
  if (!src_points.empty()) c0 /= static_cast<float>(src_points.size());
  const Eigen::Vector3f c1 = R_final * c0 + t_final;

  const Eigen::Quaternionf q0 = Eigen::Quaternionf::Identity();
  Eigen::Quaternionf q1(R_final);
  q1.normalize();

  auto smoothstep = [](float s) { return s * s * (3.0f - 2.0f * s); };

  auto set_frame = [&](float s_raw) {
    const float s = smoothstep(std::max(0.0f, std::min(1.0f, s_raw)));
    const Eigen::Matrix3f R = q0.slerp(s, q1).toRotationMatrix();
    const Eigen::Vector3f cen = (1.0f - s) * c0 + s * c1;
    for (size_t i = 0; i < src_points.size(); ++i) {
      const Eigen::Vector3f p = R * (src_points[i] - c0) + cen;
      (*src)[i].x = p.x(); (*src)[i].y = p.y(); (*src)[i].z = p.z();
    }
  };

  pcl::visualization::PCLVisualizer::Ptr viewer(
      new pcl::visualization::PCLVisualizer("Alignment animation"));
  const double bg = white_bg ? 1.0 : 0.05;
  viewer->setBackgroundColor(bg, bg, bg);
  viewer->setSize(1280, 960);
  const double tx = white_bg ? 0.1 : 0.85;  // text grey for contrast on bg
  viewer->addText("space=pause  r=record pass  q=quit", 10, 30, 12,
                  tx, tx, tx, "help");
  if (!real_colors)
    viewer->addText("orange=source  cyan=target", 10, 10, 12,
                    white_bg ? 0.85 : 0.95, white_bg ? 0.45 : 0.75,
                    white_bg ? 0.1 : 0.45, "leg");

  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>
      h_tgt(tgt), h_src(src);
  viewer->addPointCloud<pcl::PointXYZRGB>(tgt, h_tgt, "tgt");
  viewer->addPointCloud<pcl::PointXYZRGB>(src, h_src, "src");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "tgt");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "src");

  set_frame(0.0f);
  viewer->updatePointCloud(src, "src");

  if (camera.size() == 9 && !frame_target) {
    // Explicit viewpoint: pos(xyz), look-at(xyz), up(xyz).
    viewer->setCameraPosition(camera[0], camera[1], camera[2],
                              camera[3], camera[4], camera[5],
                              camera[6], camera[7], camera[8]);
    std::vector<pcl::visualization::Camera> cams;
    viewer->getCameras(cams);
    if (!cams.empty()) {
      auto cam = cams[0];
      const double fy = (fovy > 0.0f) ? fovy : cam.fovy;
      cam.fovy = fy / std::max(0.05f, zoom);  // zoom>1 narrows FOV = zoom in
      viewer->setCameraParameters(cam);
    }
  } else {
    // Robust auto-framing: median centre + percentile radius (ignore outliers),
    // keep resetCamera's view direction but move in for a tighter shot.
    // Frame the target only (source flies into view) or both clouds.
    std::vector<Eigen::Vector3f> all;
    all.reserve(src_points.size() + tgt_points.size());
    if (!frame_target)
      all.insert(all.end(), src_points.begin(), src_points.end());
    all.insert(all.end(), tgt_points.begin(), tgt_points.end());

    auto median_axis = [&](int ax) {
      std::vector<float> v(all.size());
      for (size_t i = 0; i < all.size(); ++i) v[i] = all[i][ax];
      const size_t m = v.size() / 2;
      std::nth_element(v.begin(), v.begin() + m, v.end());
      return v[m];
    };
    const Eigen::Vector3f c(median_axis(0), median_axis(1), median_axis(2));

    std::vector<float> dist(all.size());
    for (size_t i = 0; i < all.size(); ++i) dist[i] = (all[i] - c).norm();
    size_t k = static_cast<size_t>(0.85 * dist.size());
    if (k >= dist.size()) k = dist.empty() ? 0 : dist.size() - 1;
    if (!dist.empty()) std::nth_element(dist.begin(), dist.begin() + k, dist.end());
    const float r = dist.empty() ? 1.0f : std::max(1e-3f, dist[k]);

    viewer->resetCamera();
    std::vector<pcl::visualization::Camera> cams;
    viewer->getCameras(cams);
    if (!cams.empty()) {
      auto cam = cams[0];
      const Eigen::Vector3f pos_old(cam.pos[0], cam.pos[1], cam.pos[2]);
      const Eigen::Vector3f foc_old(cam.focal[0], cam.focal[1], cam.focal[2]);
      Eigen::Vector3f dir = pos_old - foc_old;
      dir = dir.norm() > 1e-6f ? dir.normalized() : Eigen::Vector3f(0, 0, 1);
      const float d = r * 2.0f / std::max(0.05f, zoom);
      const Eigen::Vector3f np = c + dir * d;
      cam.focal[0] = c.x(); cam.focal[1] = c.y(); cam.focal[2] = c.z();
      cam.pos[0] = np.x();  cam.pos[1] = np.y();  cam.pos[2] = np.z();
      cam.clip[0] = std::max(0.01f, d - 4.0f * r);
      cam.clip[1] = d + 4.0f * r;
      if (fovy > 0.0f) cam.fovy = fovy;
      viewer->setCameraParameters(cam);
    }
  }

  AnimState st;
  viewer->registerKeyboardCallback(animKeyCb, &st);

  const int ramp = std::max(1, ramp_frames);
  const int hold = std::max(0, hold_frames);

  auto record_pass = [&](const std::string& save_dir) {
    fs::create_directories(save_dir);
    int idx = 0;
    auto save = [&]() {
      char name[32];
      std::snprintf(name, sizeof(name), "/frame_%05d.png", idx++);
      viewer->saveScreenshot(save_dir + name);
    };
    for (int h = 0; h < hold; ++h) { set_frame(0.0f); viewer->updatePointCloud(src, "src"); viewer->spinOnce(1); save(); }
    for (int f = 0; f <= ramp; ++f) { set_frame(static_cast<float>(f) / ramp); viewer->updatePointCloud(src, "src"); viewer->spinOnce(1); save(); }
    for (int h = 0; h < hold; ++h) { set_frame(1.0f); viewer->updatePointCloud(src, "src"); viewer->spinOnce(1); save(); }
    std::cout << "Wrote " << idx << " frames to " << save_dir << "/\n"
              << "Make a video with:\n  ffmpeg -framerate 30 -i "
              << save_dir << "/frame_%05d.png -c:v libx264 -pix_fmt yuv420p "
              << "alignment.mp4\n";
  };

  // Auto-record one pass and exit (non-interactive / headless use).
  if (auto_record && !save_dir.empty()) {
    for (int w = 0; w < 10; ++w) viewer->spinOnce(16);  // let the window settle
    record_pass(save_dir);
    // Same estimate again in real colours (one registration, two recordings).
    if (!save_dir_real.empty()) {
      auto recolor = [](CloudRGB& c, const std::vector<Eigen::Vector4f>& cols) {
        for (size_t i = 0; i < cols.size(); ++i) {
          c[i].r = static_cast<uint8_t>(std::min(1.0f, cols[i][0]) * 255.0f);
          c[i].g = static_cast<uint8_t>(std::min(1.0f, cols[i][1]) * 255.0f);
          c[i].b = static_cast<uint8_t>(std::min(1.0f, cols[i][2]) * 255.0f);
        }
      };
      recolor(*src, src_colors);
      recolor(*tgt, tgt_colors);
      viewer->updatePointCloud(tgt, "tgt");
      viewer->removeShape("leg");
      record_pass(save_dir_real);
    }
    viewer->close();
    return;
  }

  std::cout << "Animation: source -> target. space=pause, q=quit.\n";
  if (!save_dir.empty())
    std::cout << "Frame the view, then press 'r' to record a pass to " << save_dir << "/\n";

  float s = 0.0f, dir = 1.0f;
  const float step = 1.0f / static_cast<float>(ramp);

  while (!viewer->wasStopped()) {
    if (st.record_request && !save_dir.empty()) {
      st.record_request = false;
      record_pass(save_dir);
      s = 1.0f; dir = -1.0f;
      continue;
    }

    if (st.playing) {
      s += dir * step;
      if (s >= 1.0f) { s = 1.0f; dir = -1.0f; }
      else if (s <= 0.0f) { s = 0.0f; dir = 1.0f; }
      set_frame(s);
      viewer->updatePointCloud(src, "src");
    }
    viewer->spinOnce(16);
  }
}
#endif

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    printUsage(argv[0]);
    return 1;
  }

  std::string src_path = argv[1];
  std::string tgt_path = argv[2];

  // ── Defaults = benchmark winner: 3DGS-Matcher coarse (ColorPFH + raw colour,
  //    FPFH off, joint-5D) + colour-weighted GICP refinement, at 0.5 m. ──
  float voxel_size = 0.5f;
  int num_max_corr = 5000;
  float min_alpha = 0.0f;
  float color_weight = 1.0f;
  bool f1 = false, f2 = false, f3 = false;   // FPFH geometric sub-features OFF
  bool use_color = true, use_opacity = false;
  int bins = 11;
  bool centroid = true;
  bool use_voxel_sampling = true;
  bool use_joint_5d = true;
  std::string refine = "color-gicp";          // none | gicp | color-gicp
  float voxel_size_ab = 5.0f;
  int min_color_bins = 2;
  bool viz = false;
  bool animate = false;
  int animate_frames = 150;
  int animate_hold = 30;
  std::string animate_save;
  std::string animate_save_real;       // second pass in real colours, same run
  bool animate_auto = false;
  bool animate_real_colors = false;
  bool animate_bg_white = false;
  float animate_zoom = 1.0f;
  std::vector<double> animate_camera;  // 9 values: pos, look-at, up
  float animate_fovy = 0.0f;           // vertical FOV in radians (0 = leave default)
  int animate_max_points = 0;          // cap displayed points per cloud (0 = all)
  bool animate_frame_target = false;   // frame the target cloud (source flies in)
  bool use_cross_color_filter = false;
  float cross_color_min_overlap = 0.1f;
  bool use_delta_e_hist = false;
  float delta_e_radius = 1.0f;
  int delta_e_bins = 11;
  float delta_e_max = 50.0f;
  bool use_color_pfh = true;
  float color_pfh_radius = 1.0f;
  int color_pfh_bins = 11;
  float color_pfh_max_de = 100.0f;
  float perturb_rotation_deg = 0.0f;
  float perturb_translation = 0.0f;
  uint32_t perturb_seed = 42;
  int max_attempts = 1;  // re-run coarse until valid (RANSAC is stochastic)

  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    auto need = [&](int n) {
      if (i + n >= argc) {
        std::cerr << "Missing value for " << a << "\n";
        std::exit(1);
      }
    };
    if (a == "--voxel-size") {
      need(1); voxel_size = std::stof(argv[++i]);
    } else if (a == "--num-max-corr") {
      need(1); num_max_corr = std::stoi(argv[++i]);
    } else if (a == "--min-alpha") {
      need(1); min_alpha = std::stof(argv[++i]);
    } else if (a == "--color-weight") {
      need(1); color_weight = std::stof(argv[++i]);
    } else if (a == "--f1") {
      need(1); f1 = std::stoi(argv[++i]) != 0;
    } else if (a == "--f2") {
      need(1); f2 = std::stoi(argv[++i]) != 0;
    } else if (a == "--f3") {
      need(1); f3 = std::stoi(argv[++i]) != 0;
    } else if (a == "--color") {
      need(1); use_color = std::stoi(argv[++i]) != 0;
    } else if (a == "--opacity") {
      need(1); use_opacity = std::stoi(argv[++i]) != 0;
    } else if (a == "--bins") {
      need(1); bins = std::stoi(argv[++i]);
    } else if (a == "--centroid") {
      need(1); centroid = std::stoi(argv[++i]) != 0;
    } else if (a == "--no-downsample") {
      use_voxel_sampling = false;
    } else if (a == "--joint-5d") {
      use_joint_5d = true;
    } else if (a == "--no-joint-5d") {
      use_joint_5d = false;
    } else if (a == "--refine") {
      need(1); refine = argv[++i];   // none | gicp | color-gicp
    } else if (a == "--no-refine") {
      refine = "none";
    } else if (a == "--voxel-ab") {
      need(1); voxel_size_ab = std::stof(argv[++i]);
    } else if (a == "--min-color-bins") {
      need(1); min_color_bins = std::stoi(argv[++i]);
    } else if (a == "--viz") {
      viz = true;
    } else if (a == "--animate") {
      animate = true;
    } else if (a == "--animate-frames") {
      need(1); animate_frames = std::stoi(argv[++i]);
    } else if (a == "--animate-hold") {
      need(1); animate_hold = std::stoi(argv[++i]);
    } else if (a == "--animate-save") {
      need(1); animate_save = argv[++i]; animate = true;
    } else if (a == "--animate-save-real") {
      need(1); animate_save_real = argv[++i];
    } else if (a == "--animate-auto") {
      animate_auto = true; animate = true;
    } else if (a == "--animate-real-colors") {
      animate_real_colors = true;
    } else if (a == "--animate-bg-white") {
      animate_bg_white = true;
    } else if (a == "--animate-zoom") {
      need(1); animate_zoom = std::stof(argv[++i]);
    } else if (a == "--animate-cam") {
      need(9);
      animate_camera.clear();
      for (int j = 0; j < 9; ++j) animate_camera.push_back(std::stod(argv[++i]));
    } else if (a == "--animate-fovy") {
      need(1); animate_fovy = std::stof(argv[++i]);
    } else if (a == "--animate-max-points") {
      need(1); animate_max_points = std::stoi(argv[++i]);
    } else if (a == "--animate-frame-target") {
      animate_frame_target = true;
    } else if (a == "--retry") {
      need(1); max_attempts = std::max(1, std::stoi(argv[++i]));
    } else if (a == "--cross-color-filter") {
      use_cross_color_filter = true;
    } else if (a == "--cross-color-min-overlap") {
      need(1); cross_color_min_overlap = std::stof(argv[++i]);
    } else if (a == "--delta-e-hist") {
      use_delta_e_hist = true;
    } else if (a == "--delta-e-radius") {
      need(1); delta_e_radius = std::stof(argv[++i]);
    } else if (a == "--delta-e-bins") {
      need(1); delta_e_bins = std::stoi(argv[++i]);
    } else if (a == "--delta-e-max") {
      need(1); delta_e_max = std::stof(argv[++i]);
    } else if (a == "--color-pfh") {
      use_color_pfh = true;
    } else if (a == "--color-pfh-radius") {
      need(1); color_pfh_radius = std::stof(argv[++i]);
    } else if (a == "--color-pfh-bins") {
      need(1); color_pfh_bins = std::stoi(argv[++i]);
    } else if (a == "--color-pfh-max-de") {
      need(1); color_pfh_max_de = std::stof(argv[++i]);
    } else if (a == "--perturb-rotation") {
      need(1); perturb_rotation_deg = std::stof(argv[++i]);
    } else if (a == "--perturb-translation") {
      need(1); perturb_translation = std::stof(argv[++i]);
    } else if (a == "--seed") {
      need(1); perturb_seed = static_cast<uint32_t>(std::stoul(argv[++i]));
    } else if (a == "-h" || a == "--help") {
      printUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown option: " << a << "\n";
      return 1;
    }
  }

  std::vector<GaussianSplat> src_splats, tgt_splats;
  if (!loadFile(src_path, src_splats) || !loadFile(tgt_path, tgt_splats)) {
    return 1;
  }

  std::vector<Eigen::Vector3f> src_points, tgt_points;
  std::vector<Eigen::Vector4f> src_colors, tgt_colors;
  extractPointsAndColors(src_splats, min_alpha, src_points, src_colors);
  extractPointsAndColors(tgt_splats, min_alpha, tgt_points, tgt_colors);

  std::cout << "Source: " << src_points.size() << " / " << src_splats.size() << " splats\n";
  std::cout << "Target: " << tgt_points.size() << " / " << tgt_splats.size() << " splats\n";

  Eigen::Matrix4f ground_truth_T = Eigen::Matrix4f::Identity();
  if (perturb_rotation_deg > 0.0f || perturb_translation > 0.0f) {
    std::mt19937 rng(perturb_seed);
    std::uniform_real_distribution<float> rot_dist(-perturb_rotation_deg, perturb_rotation_deg);
    std::uniform_real_distribution<float> trans_dist(-perturb_translation, perturb_translation);

    const float rx = rot_dist(rng) * M_PI / 180.0f;
    const float ry = rot_dist(rng) * M_PI / 180.0f;
    const float rz = rot_dist(rng) * M_PI / 180.0f;
    const Eigen::Matrix3f R =
        (Eigen::AngleAxisf(rz, Eigen::Vector3f::UnitZ()) *
         Eigen::AngleAxisf(ry, Eigen::Vector3f::UnitY()) *
         Eigen::AngleAxisf(rx, Eigen::Vector3f::UnitX())).toRotationMatrix();
    const Eigen::Vector3f t(trans_dist(rng), trans_dist(rng), trans_dist(rng));

    for (auto& p : src_points) p = R * p + t;

    // The source was perturbed by P=[R|t]; the matcher must recover its inverse
    // (T_target_source that undoes the perturbation). Ground truth = P^{-1}.
    Eigen::Matrix4f P = Eigen::Matrix4f::Identity();
    P.block<3,3>(0,0) = R;
    P.block<3,1>(0,3) = t;
    ground_truth_T = P.inverse();

    std::cout << "Perturbation (seed=" << perturb_seed << "):\n"
              << "  rotation  (deg): [" << rx*180/M_PI << ", " << ry*180/M_PI << ", " << rz*180/M_PI << "]\n"
              << "  translation (m): [" << t.transpose() << "]\n";
  }
  std::cout << "Config: voxel=" << voxel_size
            << " num_max_corr=" << num_max_corr
            << " min_alpha=" << min_alpha
            << " f=[" << f1 << "," << f2 << "," << f3 << "]"
            << " color=" << use_color
            << " opacity=" << use_opacity
            << " bins=" << bins
            << " centroid=" << centroid
            << "\n";

  gs3d_matcher::ColorKISSMatcherConfig cfg(
      voxel_size,
      use_voxel_sampling,
      false,   // use_quatro
      1.0f,    // thr_linearity
      num_max_corr,
      color_weight,
      f1, f2, f3,
      use_color, use_opacity,
      bins,
      centroid,
      use_joint_5d,
      voxel_size_ab,
      min_color_bins,
      use_cross_color_filter,
      cross_color_min_overlap,
      use_delta_e_hist,
      delta_e_radius,
      delta_e_bins,
      delta_e_max,
      use_color_pfh,
      color_pfh_radius,
      color_pfh_bins,
      color_pfh_max_de);

  gs3d_matcher::ColorKISSMatcher matcher(cfg);
  auto solution = matcher.estimate(src_points, tgt_points, src_colors, tgt_colors);
  for (int attempt = 2; attempt <= max_attempts && !solution.valid; ++attempt) {
    std::cout << "Coarse invalid; retry " << attempt << "/" << max_attempts << "...\n";
    solution = matcher.estimate(src_points, tgt_points, src_colors, tgt_colors);
  }

  std::cout << "Coarse (3DGS-Matcher): valid=" << (solution.valid ? "YES" : "NO")
            << "  matched pairs=" << matcher.getNumInliers() << "\n";

  // ── Final transform = coarse, then optional fine registration ──────────────
  // Convention: T maps source -> target (T_target_source).
  Eigen::Matrix3f R_final = solution.rotation.cast<float>();
  Eigen::Vector3f t_final = solution.translation.cast<float>();

  if (solution.valid && refine != "none") {
#ifdef COLOR_MATCHER_HAS_GICP
    Eigen::Isometry3d coarse_T = Eigen::Isometry3d::Identity();
    coarse_T.linear()      = solution.rotation;
    coarse_T.translation() = solution.translation;
    if (refine == "gicp") {
      small_gicp::RegistrationSetting gicp_cfg;
      gicp_cfg.type = small_gicp::RegistrationSetting::GICP;
      gicp_cfg.downsampling_resolution    = static_cast<double>(voxel_size);
      gicp_cfg.max_correspondence_distance = static_cast<double>(voxel_size) * 3.0;
      gicp_cfg.num_threads   = 4;
      gicp_cfg.max_iterations = 50;
      auto result = small_gicp::align(tgt_points, src_points, coarse_T, gicp_cfg);
      R_final = result.T_target_source.linear().cast<float>();
      t_final = result.T_target_source.translation().cast<float>();
      std::cout << "Refine: geometric GICP (" << result.iterations << " iters)\n";
    } else if (refine == "color-gicp") {
      auto result = small_gicp::align_colored_simple(
          tgt_points, tgt_colors, src_points, src_colors,
          static_cast<double>(voxel_size),
          /*color_weight=sigma; <=0 => auto*/ -1.0,
          /*registration_type=*/"GICP_DEW", coarse_T);
      R_final = result.T_target_source.linear().cast<float>();
      t_final = result.T_target_source.translation().cast<float>();
      std::cout << "Refine: colour-weighted GICP (3DGS-ICP, " << result.iterations << " iters)\n";
    } else {
      std::cerr << "Unknown --refine '" << refine << "' (use none|gicp|color-gicp); skipping refinement.\n";
    }
#else
    std::cerr << "--refine '" << refine << "' requested, but this build has no GICP support "
                 "(small_gicp not found at configure time); using coarse only.\n";
#endif
  }

  std::cout << "Final rotation:\n" << R_final << "\n";
  std::cout << "Final translation: " << t_final.transpose() << "\n";

  if (ground_truth_T != Eigen::Matrix4f::Identity()) {
    Eigen::Matrix4d est = Eigen::Matrix4d::Identity();
    est.block<3,3>(0,0) = R_final.cast<double>();
    est.block<3,1>(0,3) = t_final.cast<double>();
    Eigen::Matrix4d gt = ground_truth_T.cast<double>();
    const Eigen::Matrix4d err = est * gt.inverse();
    const Eigen::Matrix3d err_R = err.block<3,3>(0,0);
    const double rot_err = std::acos(std::max(-1.0, std::min(1.0, 0.5 * (err_R.trace() - 1.0)))) * 180.0 / M_PI;
    const double trans_err = err.block<3,1>(0,3).norm();
    std::cout << "Rotation error:    " << rot_err << " deg\n";
    std::cout << "Translation error: " << trans_err << " m\n";
  }

  if (viz) {
#ifdef COLOR_MATCHER_USE_PCL_VIZ
    std::vector<Eigen::Vector3f> src_aligned(src_points.size());
    for (size_t i = 0; i < src_points.size(); ++i) {
      src_aligned[i] = R_final * src_points[i] + t_final;
    }
    visualizeBeforeAfter(src_points, tgt_points, src_aligned,
                         matcher.getSrcProcessed(), matcher.getSrcProcessedColors(),
                         matcher.getTgtProcessed(), matcher.getTgtProcessedColors());
#else
    std::cerr << "--viz requested, but this build has no PCL visualization support.\n";
#endif
  }

  if (animate) {
#ifdef COLOR_MATCHER_USE_PCL_VIZ
    // Optionally thin the clouds for display only (matching used full data).
    auto subsample = [&](const std::vector<Eigen::Vector3f>& pts,
                         const std::vector<Eigen::Vector4f>& cols,
                         std::vector<Eigen::Vector3f>& op,
                         std::vector<Eigen::Vector4f>& oc) {
      if (animate_max_points <= 0 ||
          static_cast<int>(pts.size()) <= animate_max_points) {
        op = pts; oc = cols; return;
      }
      const size_t stride =
          (pts.size() + animate_max_points - 1) / animate_max_points;
      for (size_t i = 0; i < pts.size(); i += stride) {
        op.push_back(pts[i]); oc.push_back(cols[i]);
      }
    };
    std::vector<Eigen::Vector3f> asp, atp;
    std::vector<Eigen::Vector4f> asc, atc;
    subsample(src_points, src_colors, asp, asc);
    subsample(tgt_points, tgt_colors, atp, atc);
    std::cout << "Display points: src " << asp.size() << "  tgt " << atp.size() << "\n";

    animateAlignment(asp, asc, atp, atc,
                     R_final, t_final, animate_frames, animate_hold, animate_save,
                     animate_auto, animate_real_colors, animate_bg_white,
                     animate_zoom, animate_camera, animate_fovy,
                     animate_frame_target, animate_save_real);
#else
    std::cerr << "--animate requested, but this build has no PCL visualization support.\n";
#endif
  }

  return solution.valid ? 0 : 1;
}

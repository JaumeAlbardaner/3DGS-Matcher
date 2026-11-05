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
      << "Align two 3D Gaussian Splat maps and print the rigid transform that\n"
      << "maps source onto target (T_target_source).\n\n"
      << "Usage:\n"
      << "  " << prog << " <source.splat|.ply> <target.splat|.ply> [options]\n\n"
      << "Defaults are the configuration evaluated in the paper: coarse 3DGS-Matcher\n"
      << "(37-D descriptor = raw RGB + opacity + relative CIELAB ColorPFH, joint-5D\n"
      << "color-aware keypoint selection) followed by the color-weighted GICP\n"
      << "refinement (3DGS-ICP).\n\n"
      << "Installation check (aligns the sample splat to a randomly moved copy of\n"
      << "itself and prints PASS when the error is below 5 deg / 0.5 m):\n"
      << "  " << prog << " examples/splats/tramway.splat examples/splats/tramway.splat \\\n"
      << "      --perturb-rotation 180 --perturb-translation 500 --seed 1\n\n"
      << "Options:\n"
      << "  --voxel-size <float>   Downsampling resolution in meters (default: 0.5)\n"
      << "  --min-alpha <0..1>     Drop splats with alpha < threshold (default: 0.0;\n"
      << "                         raise it to filter near-transparent floaters)\n"
      << "  --refine <mode>        Fine registration: none | gicp | color-gicp\n"
      << "                         (default: color-gicp = the color-weighted 3DGS-ICP;\n"
      << "                          use 'none' for coarse-only)\n"
      << "  --centroid             Use centroid downsampling instead of the default\n"
      << "                         joint-5D color-aware keypoint selection (weaker baseline)\n"
      << "  --no-opacity           Drop opacity from the descriptor (36-D: RGB + ColorPFH)\n"
      << "  --voxel-ab <float>     Joint-5D chroma bin size in CIELAB a*/b* units (default: 5)\n"
      << "  --min-color-bins <int> Drop spatial voxels with fewer distinct chroma bins (default: 2)\n"
      << "  --cpfh-radius <float>  ColorPFH neighborhood radius in meters (default: 1.0)\n"
      << "  --num-max-corr <int>   Max correspondences passed to the solver (default: 5000)\n"
      << "  --perturb-rotation <deg>    Move the source by a random rotation of up to\n"
      << "                              +-deg about each axis, then report the error\n"
      << "  --perturb-translation <m>   ... and a random translation of up to +-m\n"
      << "  --seed <int>                RNG seed for the perturbation (default: 42)\n"
      << "  --viz                  Show before/after 3D viewer (requires PCL build)\n"
      << "  -h, --help             Show this help\n";
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

  std::cout << "Viewer: left=before | center=after | right=selected points (real colors).\n"
            << "src selected: " << src_selected.size()
            << "  tgt selected: " << tgt_selected.size() << "\n"
            << "Press 'q' to quit.\n";

  while (!viewer->wasStopped()) {
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

  // ── The descriptor/keypoint pipeline is fixed to the winning configuration
  //    (see config below); only the parameters a user would tune when aligning
  //    two real maps are exposed as flags. ──
  float voxel_size = 0.5f;             // --voxel-size : downsampling resolution
  float min_alpha  = 0.0f;             // --min-alpha  : drop near-transparent splats
  std::string refine = "color-gicp";   // --refine {none|gicp|color-gicp}
  bool use_joint_5d = true;            // --centroid disables this (centroid baseline)
  bool viz = false;                    // --viz
  bool use_opacity = true;             // --no-opacity
  float voxel_ab = 5.0f;               // --voxel-ab
  int min_color_bins = 2;              // --min-color-bins
  float cpfh_radius = 1.0f;            // --cpfh-radius
  int num_max_corr = 5000;             // --num-max-corr
  float perturb_rotation_deg = 0.0f;   // --perturb-rotation
  float perturb_translation = 0.0f;    // --perturb-translation
  uint32_t perturb_seed = 42;          // --seed

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
    } else if (a == "--min-alpha") {
      need(1); min_alpha = std::stof(argv[++i]);
    } else if (a == "--refine") {
      need(1); refine = argv[++i];   // none | gicp | color-gicp
    } else if (a == "--centroid") {
      use_joint_5d = false;          // centroid downsampling instead of joint-5D
    } else if (a == "--no-opacity") {
      use_opacity = false;
    } else if (a == "--voxel-ab") {
      need(1); voxel_ab = std::stof(argv[++i]);
    } else if (a == "--min-color-bins") {
      need(1); min_color_bins = std::stoi(argv[++i]);
    } else if (a == "--cpfh-radius") {
      need(1); cpfh_radius = std::stof(argv[++i]);
    } else if (a == "--num-max-corr") {
      need(1); num_max_corr = std::stoi(argv[++i]);
    } else if (a == "--perturb-rotation") {
      need(1); perturb_rotation_deg = std::stof(argv[++i]);
    } else if (a == "--perturb-translation") {
      need(1); perturb_translation = std::stof(argv[++i]);
    } else if (a == "--seed") {
      need(1); perturb_seed = static_cast<uint32_t>(std::stoul(argv[++i]));
    } else if (a == "--viz") {
      viz = true;
    } else if (a == "-h" || a == "--help") {
      printUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown option: " << a << "\n";
      printUsage(argv[0]);
      return 1;
    }
  }

  if (refine != "none" && refine != "gicp" && refine != "color-gicp") {
    std::cerr << "Invalid --refine '" << refine << "' (use none|gicp|color-gicp)\n";
    return 1;
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

  // Optional known perturbation of the source, to check the result against.
  // The matcher must recover P^-1 (T_target_source undoing P = [R|t]).
  Eigen::Matrix4f ground_truth_T = Eigen::Matrix4f::Identity();
  const bool perturbed = perturb_rotation_deg > 0.0f || perturb_translation > 0.0f;
  if (perturbed) {
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

    Eigen::Matrix4f P = Eigen::Matrix4f::Identity();
    P.block<3,3>(0,0) = R;
    P.block<3,1>(0,3) = t;
    ground_truth_T = P.inverse();
    std::cout << "Perturbation (seed=" << perturb_seed << "):\n"
              << "  rotation  (deg): [" << rx*180/M_PI << ", " << ry*180/M_PI << ", " << rz*180/M_PI << "]\n"
              << "  translation (m): [" << t.transpose() << "]\n";
  }

  std::cout << "Config: voxel=" << voxel_size << " m, min_alpha=" << min_alpha
            << "  [ColorPFH(r=" << cpfh_radius << ") + raw RGB" << (use_opacity ? " + opacity" : "")
            << ", " << (use_joint_5d ? "joint-5D" : "centroid") << " keypoints (v_ab=" << voxel_ab
            << ", min bins=" << min_color_bins << "), FPFH off]  refine=" << refine << "\n";

  // ── Fixed winning configuration (do not expose as flags) ──
  //   Coarse 3DGS-Matcher: 37-D descriptor = raw RGB (3) + opacity (1) + relative
  //   CIELAB ColorPFH histogram (3x11), joint-5D color-aware keypoint selection,
  //   FPFH geometric features disabled. These choices reproduce the published
  //   benchmark result (paper Sec. III-C, Table I "joint-5D + opacity" row).
  gs3d_matcher::ColorKISSMatcherConfig cfg(
      voxel_size,
      /*use_voxel_sampling=*/true,
      /*use_quatro=*/false,
      /*thr_linearity=*/1.0f,
      /*num_max_corr=*/num_max_corr,
      /*color_weight=*/1.0f,
      /*use_color=*/true, /*use_opacity=*/use_opacity, // raw RGB ON, opacity ON by default
      /*use_centroid_voxel=*/true,
      /*use_joint_5d=*/use_joint_5d,                  // joint-5D (default) or centroid (--centroid)
      /*voxel_size_ab=*/voxel_ab,
      /*min_color_bins=*/min_color_bins,
      /*use_color_pfh=*/true, /*color_pfh_radius=*/cpfh_radius, /*color_pfh_bins=*/11, /*color_pfh_max_de=*/100.0f);

  gs3d_matcher::ColorKISSMatcher matcher(cfg);
  const auto solution = matcher.estimate(src_points, tgt_points, src_colors, tgt_colors);

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
      std::cout << "Refine: color-weighted GICP (3DGS-ICP, " << result.iterations << " iters)\n";
    } else {
      std::cerr << "Unknown --refine '" << refine << "' (use none|gicp|color-gicp); skipping refinement.\n";
    }
#else
    std::cerr << "--refine '" << refine << "' requested, but this build has no GICP support "
                 "(small_gicp not found at configure time); using coarse only.\n";
#endif
  }

  std::cout << "Final transform (T_target_source):\n";
  std::cout << "  rotation:\n" << R_final << "\n";
  std::cout << "  translation: " << t_final.transpose() << "\n";

  bool check_ok = true;
  if (perturbed) {
    Eigen::Matrix4d est = Eigen::Matrix4d::Identity();
    est.block<3,3>(0,0) = R_final.cast<double>();
    est.block<3,1>(0,3) = t_final.cast<double>();
    const Eigen::Matrix4d err = est * ground_truth_T.cast<double>().inverse();
    const Eigen::Matrix3d err_R = err.block<3,3>(0,0);
    const double rot_err =
        std::acos(std::max(-1.0, std::min(1.0, 0.5 * (err_R.trace() - 1.0)))) * 180.0 / M_PI;
    const double trans_err = err.block<3,1>(0,3).norm();
    check_ok = rot_err < 5.0 && trans_err < 0.5;   // paper's success criterion
    std::cout << "Rotation error:    " << rot_err << " deg\n"
              << "Translation error: " << trans_err << " m\n"
              << (check_ok ? "PASS" : "FAIL") << " (success = rotation < 5 deg and translation < 0.5 m)\n";
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

  return (solution.valid && check_ok) ? 0 : 1;
}

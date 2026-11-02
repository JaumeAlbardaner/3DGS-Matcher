/**
 * Single Ablation Config Runner
 * 
 * Runs ONE feature configuration with specified parameters and exits.
 * Called by run_ablation.sh to avoid memory leak accumulation.
 * 
 * Usage: ./single_ablation <splat_file> <voxel_size> <f1> <f2> <f3> <color> <opacity> <bins> <noise> <color_weight> <seed> [max_translation] [max_rotation_deg] [centroid_voxel=1] [experiment=baseline|outliers|color_offset|split_half]
 * 
 * Noise model:
 *   - Position noise: Gaussian with std = noise
 *   - Color noise: CIELAB L*,a*,b* Gaussian with std = noise * 10
 *   - Alpha noise: Gaussian with std = noise * 0.1
 * 
 * Output: CSV line to stdout
 */

#include <3dgs_matcher/ColorFasterPFH.hpp>
#include <3dgs_matcher/ColorKISSMatcher.hpp>
#include <kiss_matcher/KISSMatcher.hpp>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <random>
#include <cmath>

using namespace gs3d_matcher;

struct GaussianSplat {
  Eigen::Vector3f position;
  Eigen::Vector3f scale;
  Eigen::Vector4f color_rgba;
};

bool loadSplatFile(const std::string& filename, std::vector<GaussianSplat>& splats) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return false;
  }

  file.seekg(0, std::ios::end);
  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  if (file_size % 32 != 0) {
    std::cerr << "Unsupported .splat format" << std::endl;
    return false;
  }

  size_t num_splats = file_size / 32;
  splats.reserve(num_splats);

  for (size_t i = 0; i < num_splats; ++i) {
    GaussianSplat splat;
    float data[6];
    file.read(reinterpret_cast<char*>(data), sizeof(float) * 6);
    
    splat.position = Eigen::Vector3f(data[0], data[1], data[2]);
    splat.scale = Eigen::Vector3f(data[3], data[4], data[5]);
    
    uint8_t color_rgba[4];
    file.read(reinterpret_cast<char*>(color_rgba), 4);
    uint8_t rotation[4];
    file.read(reinterpret_cast<char*>(rotation), 4);
    
    splat.color_rgba = Eigen::Vector4f(
        color_rgba[0] / 255.0f,
        color_rgba[1] / 255.0f,
        color_rgba[2] / 255.0f,
        color_rgba[3] / 255.0f
    );
    
    splats.push_back(splat);
  }

  return true;
}

bool loadPlyFile(const std::string& filename, std::vector<GaussianSplat>& splats) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) { std::cerr << "Failed to open: " << filename << "\n"; return false; }

  std::string line;
  if (!std::getline(file, line) || line != "ply") {
    std::cerr << "Not a PLY file: " << filename << "\n"; return false;
  }

  struct Prop { std::string type, name; bool is_list = false;
                std::string list_ct, list_vt; };
  bool binary_le = false, ascii = false, in_vertex = false;
  size_t vertex_count = 0;
  std::vector<Prop> props;

  while (std::getline(file, line)) {
    if (line == "end_header") break;
    std::istringstream iss(line); std::string tok; iss >> tok;
    std::transform(tok.begin(), tok.end(), tok.begin(), ::tolower);
    if (tok == "format") {
      std::string fmt; iss >> fmt;
      std::transform(fmt.begin(), fmt.end(), fmt.begin(), ::tolower);
      ascii     = (fmt == "ascii");
      binary_le = (fmt == "binary_little_endian");
    } else if (tok == "element") {
      std::string n; size_t c = 0; iss >> n >> c;
      std::transform(n.begin(), n.end(), n.begin(), ::tolower);
      in_vertex = (n == "vertex"); if (in_vertex) vertex_count = c;
    } else if (tok == "property" && in_vertex) {
      Prop p; std::string next; iss >> next;
      std::transform(next.begin(), next.end(), next.begin(), ::tolower);
      if (next == "list") { p.is_list = true; iss >> p.list_ct >> p.list_vt >> p.name; }
      else { p.type = next; iss >> p.name; }
      std::transform(p.name.begin(), p.name.end(), p.name.begin(), ::tolower);
      props.push_back(p);
    }
  }

  if (vertex_count == 0 || props.empty()) {
    std::cerr << "PLY has no vertex data: " << filename << "\n"; return false;
  }

  std::unordered_map<std::string, int> idx;
  for (int i = 0; i < (int)props.size(); ++i) idx[props[i].name] = i;
  auto find = [&](const std::string& n) { auto it = idx.find(n); return it == idx.end() ? -1 : it->second; };

  auto readBin = [&](const std::string& t) -> double {
    std::string lt = t; std::transform(lt.begin(), lt.end(), lt.begin(), ::tolower);
    if (lt=="uchar"||lt=="uint8")  { uint8_t  v; file.read((char*)&v,1); return v; }
    if (lt=="char" ||lt=="int8")   { int8_t   v; file.read((char*)&v,1); return v; }
    if (lt=="float"||lt=="float32"){ float    v; file.read((char*)&v,4); return v; }
    if (lt=="double"||lt=="float64"){double   v; file.read((char*)&v,8); return v; }
    if (lt=="int"  ||lt=="int32")  { int32_t  v; file.read((char*)&v,4); return v; }
    if (lt=="uint" ||lt=="uint32") { uint32_t v; file.read((char*)&v,4); return v; }
    return 0.0;
  };

  const int ix=find("x"),iy=find("y"),iz=find("z");
  if (ix<0||iy<0||iz<0) { std::cerr<<"PLY missing x/y/z\n"; return false; }
  const int idc0=find("f_dc_0"),idc1=find("f_dc_1"),idc2=find("f_dc_2");
  const int ir=find("red"),ig=find("green"),ib=find("blue"),ia=find("alpha");
  const int iop=find("opacity");
  const int is0=find("scale_0"),is1=find("scale_1"),is2=find("scale_2");
  constexpr double kSHC0 = 0.28209479177387814;
  auto clamp01 = [](float v){ return std::max(0.f,std::min(1.f,v)); };

  splats.clear(); splats.reserve(vertex_count);
  for (size_t vi = 0; vi < vertex_count; ++vi) {
    std::vector<double> vals(props.size(), 0.0);
    if (ascii) {
      std::getline(file, line);
      std::istringstream iss2(line);
      for (size_t pi = 0; pi < props.size(); ++pi) {
        if (props[pi].is_list) { int n; iss2>>n; for(int k=0;k<n;++k){double d;iss2>>d;} }
        else iss2 >> vals[pi];
      }
    } else {
      for (size_t pi = 0; pi < props.size(); ++pi) {
        if (props[pi].is_list) {
          int n = (int)readBin(props[pi].list_ct);
          for (int k=0;k<n;++k) readBin(props[pi].list_vt);
        } else { vals[pi] = readBin(props[pi].type); }
      }
    }
    GaussianSplat s;
    s.position = Eigen::Vector3f((float)vals[ix],(float)vals[iy],(float)vals[iz]);
    float r=0.5f,g=0.5f,b=0.5f;
    if (ir>=0&&ig>=0&&ib>=0) {
      r=clamp01((float)(vals[ir]>1.0?vals[ir]/255.0:vals[ir]));
      g=clamp01((float)(vals[ig]>1.0?vals[ig]/255.0:vals[ig]));
      b=clamp01((float)(vals[ib]>1.0?vals[ib]/255.0:vals[ib]));
    } else if (idc0>=0&&idc1>=0&&idc2>=0) {
      r=clamp01((float)(kSHC0*vals[idc0]+0.5));
      g=clamp01((float)(kSHC0*vals[idc1]+0.5));
      b=clamp01((float)(kSHC0*vals[idc2]+0.5));
    }
    float a=1.f;
    if (ia>=0) a=clamp01((float)(vals[ia]>1.0?vals[ia]/255.0:vals[ia]));
    else if (iop>=0) { double v=vals[iop]; a=(v>=0&&v<=1)?(float)v:clamp01((float)(1.0/(1.0+std::exp(-v)))); }
    s.color_rgba = Eigen::Vector4f(r,g,b,a);
    if (is0>=0&&is1>=0&&is2>=0)
      s.scale = Eigen::Vector3f((float)std::exp(vals[is0]),(float)std::exp(vals[is1]),(float)std::exp(vals[is2]));
    else s.scale = Eigen::Vector3f(1.f,1.f,1.f);
    splats.push_back(s);
  }
  return !splats.empty();
}

bool loadFile(const std::string& filename, std::vector<GaussianSplat>& splats) {
  const size_t dot = filename.rfind('.');
  std::string ext = (dot == std::string::npos) ? "" : filename.substr(dot);
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  return (ext == ".ply") ? loadPlyFile(filename, splats) : loadSplatFile(filename, splats);
}

void extractPointsAndColors(const std::vector<GaussianSplat>& splats,
                           std::vector<Eigen::Vector3f>& points,
                           std::vector<Eigen::Vector4f>& colors) {
  points.clear();
  colors.clear();
  points.reserve(splats.size());
  colors.reserve(splats.size());

  for (const auto& splat : splats) {
    points.push_back(splat.position);
    colors.push_back(splat.color_rgba);
  }
}

void applyRandomTransform(std::vector<Eigen::Vector3f>& points,
                         Eigen::Matrix4f& transform_out,
                         float max_translation,
                         float max_rotation_deg,
                         std::mt19937& gen) {
  std::uniform_real_distribution<float> trans_dist(-max_translation, max_translation);
  std::uniform_real_distribution<float> rot_dist(-max_rotation_deg, max_rotation_deg);

  float roll_deg = rot_dist(gen);
  float pitch_deg = rot_dist(gen);
  float yaw_deg = rot_dist(gen);

  Eigen::AngleAxisf rollAngle(roll_deg * M_PI / 180.0f, Eigen::Vector3f::UnitX());
  Eigen::AngleAxisf pitchAngle(pitch_deg * M_PI / 180.0f, Eigen::Vector3f::UnitY());
  Eigen::AngleAxisf yawAngle(yaw_deg * M_PI / 180.0f, Eigen::Vector3f::UnitZ());
  
  Eigen::Matrix3f rotation = (yawAngle * pitchAngle * rollAngle).toRotationMatrix();
  Eigen::Vector3f translation(trans_dist(gen), trans_dist(gen), trans_dist(gen));

  transform_out = Eigen::Matrix4f::Identity();
  transform_out.block<3, 3>(0, 0) = rotation;
  transform_out.block<3, 1>(0, 3) = translation;

  for (auto& pt : points) {
    Eigen::Vector4f pt_h(pt.x(), pt.y(), pt.z(), 1.0f);
    Eigen::Vector4f pt_transformed = transform_out * pt_h;
    pt = pt_transformed.head<3>();
  }
}

void addPositionNoise(std::vector<Eigen::Vector3f>& points,
                      float noise_std,
                      std::mt19937& gen) {
  if (noise_std <= 0.0f) return;
  
  std::normal_distribution<float> noise_dist(0.0f, noise_std);

  for (auto& pt : points) {
    pt.x() += noise_dist(gen);
    pt.y() += noise_dist(gen);
    pt.z() += noise_dist(gen);
  }
}

/**
 * @brief Convert CIELAB back to sRGB [0,1].
 */
Eigen::Vector3f labToRgb(float L, float a, float bLab) {
  float fy = (L + 16.0f) / 116.0f;
  float fx = a / 500.0f + fy;
  float fz = fy - bLab / 200.0f;

  constexpr float delta = 6.0f / 29.0f;
  auto finv = [&](float t) -> float {
    return (t > delta) ? t * t * t : 3.0f * delta * delta * (t - 4.0f / 29.0f);
  };

  constexpr float Xn = 0.95047f, Yn = 1.0f, Zn = 1.08883f;
  float x = Xn * finv(fx);
  float y = Yn * finv(fy);
  float z = Zn * finv(fz);

  // XYZ -> linear RGB
  float rl =  3.2404542f * x - 1.5371385f * y - 0.4985314f * z;
  float gl = -0.9692660f * x + 1.8760108f * y + 0.0415560f * z;
  float bl =  0.0556434f * x - 0.2040259f * y + 1.0572252f * z;

  // linear RGB -> sRGB
  auto gamma = [](float c) -> float {
    return (c <= 0.0031308f) ? 12.92f * c : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
  };
  return Eigen::Vector3f(
      std::max(0.0f, std::min(1.0f, gamma(rl))),
      std::max(0.0f, std::min(1.0f, gamma(gl))),
      std::max(0.0f, std::min(1.0f, gamma(bl))));
}

/**
 * Apply unified noise to colors in CIELAB space:
 *
 * @param colors    RGBA colors to modify (in place), sRGB [0,1] + alpha
 * @param noise     Unified noise level:
 *                  - Per-point CIELAB L*,a*,b* noise: Gaussian with std = noise * 10
 *                  - Per-point alpha noise: Gaussian with std = noise * 0.1
 * @param gen       Random generator
 */
void addColorNoise(std::vector<Eigen::Vector4f>& colors,
                   float noise,
                   std::mt19937& gen) {
  if (noise <= 0.0f) return;

  // CIELAB noise
  std::normal_distribution<float> lab_noise_dist(0.0f, noise * 10.0f);
  // Alpha noise
  std::normal_distribution<float> alpha_noise_dist(0.0f, noise * 0.1f);

  for (auto& c : colors) {
    Eigen::Vector3f lab = rgbToLab(c[0], c[1], c[2]);
    lab(0) += lab_noise_dist(gen);
    lab(1) += lab_noise_dist(gen);
    lab(2) += lab_noise_dist(gen);
    lab(0) = std::max(0.0f, std::min(100.0f, lab(0)));
    Eigen::Vector3f rgb = labToRgb(lab(0), lab(1), lab(2));
    c[0] = rgb(0);
    c[1] = rgb(1);
    c[2] = rgb(2);

    // Alpha noise
    c[3] += alpha_noise_dist(gen);
    c[3] = std::max(0.0f, std::min(1.0f, c[3]));
  }
}

void addOutliers(std::vector<Eigen::Vector3f>& points,
                 std::vector<Eigen::Vector4f>& colors,
                 float ratio,
                 float xyz_std,
                 float lab_std,
                 float alpha_std,
                 std::mt19937& gen) {
  if (ratio <= 0.0f || points.empty()) return;
  const size_t n = points.size();
  const size_t k = std::max<size_t>(1, static_cast<size_t>(std::round(ratio * static_cast<float>(n))));
  std::uniform_int_distribution<size_t> pick(0, n - 1);
  std::normal_distribution<float> d_xyz(0.0f, xyz_std);
  std::normal_distribution<float> d_lab(0.0f, lab_std);
  std::normal_distribution<float> d_alpha(0.0f, alpha_std);

  for (size_t t = 0; t < k; ++t) {
    const size_t i = pick(gen);
    points[i].x() += d_xyz(gen);
    points[i].y() += d_xyz(gen);
    points[i].z() += d_xyz(gen);

    Eigen::Vector3f lab = rgbToLab(colors[i][0], colors[i][1], colors[i][2]);
    lab[0] = std::max(0.0f, std::min(100.0f, lab[0] + d_lab(gen)));
    lab[1] += d_lab(gen);
    lab[2] += d_lab(gen);
    Eigen::Vector3f rgb = labToRgb(lab[0], lab[1], lab[2]);
    colors[i][0] = rgb[0];
    colors[i][1] = rgb[1];
    colors[i][2] = rgb[2];
    colors[i][3] = std::max(0.0f, std::min(1.0f, colors[i][3] + d_alpha(gen)));
  }
}

void applyGlobalColorOffset(std::vector<Eigen::Vector4f>& colors,
                            float offset_std,
                            float per_point_std,
                            float alpha_std,
                            std::mt19937& gen) {
  if (colors.empty()) return;
  std::normal_distribution<float> d_offset(0.0f, offset_std);
  std::normal_distribution<float> d_pp(0.0f, per_point_std);
  std::normal_distribution<float> d_alpha(0.0f, alpha_std);

  const float dL = d_offset(gen);
  const float dA = d_offset(gen);
  const float dB = d_offset(gen);

  for (auto& c : colors) {
    Eigen::Vector3f lab = rgbToLab(c[0], c[1], c[2]);
    lab[0] = std::max(0.0f, std::min(100.0f, lab[0] + dL + d_pp(gen)));
    lab[1] = lab[1] + dA + d_pp(gen);
    lab[2] = lab[2] + dB + d_pp(gen);
    Eigen::Vector3f rgb = labToRgb(lab[0], lab[1], lab[2]);
    c[0] = rgb[0];
    c[1] = rgb[1];
    c[2] = rgb[2];
    c[3] = std::max(0.0f, std::min(1.0f, c[3] + d_alpha(gen)));
  }
}


// ── Photometric and metric-scale perturbations (paper Sec. V-B) ───────────
// Each applies ONE named, physically interpretable change to the target copy and
// adds no colour noise, so its effect on registration can be read off directly.
static float srgbToLinear(float c) {
  return (c <= 0.04045f) ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}
static float linearToSrgb(float c) {
  c = std::max(0.0f, std::min(1.0f, c));
  return (c <= 0.0031308f) ? 12.92f * c : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
}

// Per-channel gains in linear light: exposure (g,g,g) or white balance (gR,1,gB).
void applyLinearGains(std::vector<Eigen::Vector4f>& colors, float gr, float gg, float gb) {
  for (auto& c : colors) {
    c[0] = linearToSrgb(srgbToLinear(c[0]) * gr);
    c[1] = linearToSrgb(srgbToLinear(c[1]) * gg);
    c[2] = linearToSrgb(srgbToLinear(c[2]) * gb);
  }
}

// Tone-curve change on the encoded values: c' = c^g.
void applyGammaCurve(std::vector<Eigen::Vector4f>& colors, float g) {
  for (auto& c : colors)
    for (int k = 0; k < 3; ++k) c[k] = std::pow(std::max(0.0f, std::min(1.0f, c[k])), g);
}

// Cast shadow: darken (linear gain) every point beyond the median plane along the
// map's longest axis. The mask is taken from the untransformed map, whose point
// order matches the target copy.
void applyHalfShadow(const std::vector<Eigen::Vector3f>& ref_points,
                     std::vector<Eigen::Vector4f>& colors, float gain) {
  if (ref_points.empty()) return;
  Eigen::Vector3f lo = ref_points[0], hi = ref_points[0];
  for (const auto& p : ref_points) { lo = lo.cwiseMin(p); hi = hi.cwiseMax(p); }
  int axis = 0; (hi - lo).maxCoeff(&axis);
  std::vector<float> vals; vals.reserve(ref_points.size());
  for (const auto& p : ref_points) vals.push_back(p[axis]);
  std::nth_element(vals.begin(), vals.begin() + vals.size() / 2, vals.end());
  const float median = vals[vals.size() / 2];
  for (size_t i = 0; i < colors.size() && i < ref_points.size(); ++i) {
    if (ref_points[i][axis] <= median) continue;
    for (int k = 0; k < 3; ++k) colors[i][k] = linearToSrgb(srgbToLinear(colors[i][k]) * gain);
  }
}

// Residual metric-scale error: scale the target about its own centroid. The best
// rigid alignment is then still exactly the ground-truth transform.
void applyScaleAboutCentroid(std::vector<Eigen::Vector3f>& points, float s) {
  if (points.empty()) return;
  Eigen::Vector3f c = Eigen::Vector3f::Zero();
  for (const auto& p : points) c += p;
  c /= static_cast<float>(points.size());
  for (auto& p : points) p = c + s * (p - c);
}

void splitHalfCloud(const std::vector<Eigen::Vector3f>& in_points,
                    const std::vector<Eigen::Vector4f>& in_colors,
                    int axis,
                    float overlap,
                    std::vector<Eigen::Vector3f>& src_points,
                    std::vector<Eigen::Vector4f>& src_colors,
                    std::vector<Eigen::Vector3f>& tgt_points,
                    std::vector<Eigen::Vector4f>& tgt_colors) {
  if (in_points.empty()) return;
  overlap = std::max(0.0f, std::min(0.95f, overlap));

  float mn = std::numeric_limits<float>::infinity();
  float mx = -std::numeric_limits<float>::infinity();
  for (const auto& p : in_points) {
    mn = std::min(mn, p(axis));
    mx = std::max(mx, p(axis));
  }
  const float span = std::max(1e-6f, mx - mn);
  const float mid = 0.5f * (mn + mx);
  const float half_ov = 0.5f * overlap * span;
  const float src_max = mid + half_ov;
  const float tgt_min = mid - half_ov;

  src_points.clear();
  src_colors.clear();
  tgt_points.clear();
  tgt_colors.clear();
  src_points.reserve(in_points.size());
  src_colors.reserve(in_colors.size());
  tgt_points.reserve(in_points.size());
  tgt_colors.reserve(in_colors.size());

  for (size_t i = 0; i < in_points.size(); ++i) {
    const float v = in_points[i](axis);
    if (v <= src_max) {
      src_points.push_back(in_points[i]);
      src_colors.push_back(in_colors[i]);
    }
    if (v >= tgt_min) {
      tgt_points.push_back(in_points[i]);
      tgt_colors.push_back(in_colors[i]);
    }
  }
}

float computeRotationError(const Eigen::Matrix4f& estimated, const Eigen::Matrix4f& gt) {
  Eigen::Matrix3f R_est = estimated.block<3, 3>(0, 0);
  Eigen::Matrix3f R_gt = gt.block<3, 3>(0, 0);
  Eigen::Matrix3f R_err = R_est.transpose() * R_gt;
  Eigen::AngleAxisf aa(R_err);
  return std::abs(aa.angle()) * 180.0f / M_PI;
}

float computeTranslationError(const Eigen::Matrix4f& estimated, const Eigen::Matrix4f& gt) {
  Eigen::Vector3f t_est = estimated.block<3, 1>(0, 3);
  Eigen::Vector3f t_gt = gt.block<3, 1>(0, 3);
  return (t_est - t_gt).norm();
}

int main(int argc, char** argv) {
  if (argc < 12 || argc > 30) {
    std::cerr << "Usage: " << argv[0]
              << " <file.splat|file.ply> <voxel_size> <f1> <f2> <f3> <color> <opacity> <bins>"
                 " <noise> <color_weight> <seed>"
                 " [max_translation] [max_rotation_deg] [centroid_voxel] [experiment] [matcher]"
                 " [joint_5d] [voxel_size_ab] [min_color_bins]"
                 " [cross_color_filter] [cross_color_min_overlap]"
                 " [delta_e_hist] [delta_e_radius] [delta_e_bins] [delta_e_max]\n"
              << "  joint_5d              : 0|1 (default: 0)\n"
              << "  voxel_size_ab         : LAB ab bin size in ΔE (default: 5.0)\n"
              << "  min_color_bins        : min distinct colors per spatial voxel (default: 2)\n"
              << "  cross_color_filter    : 0|1 — keep only shared color bins (default: 0)\n"
              << "  cross_color_min_overlap: min overlap ratio (default: 0.1)\n"
              << "  delta_e_hist          : 0|1 — append ΔEab histogram descriptor (default: 0)\n"
              << "  delta_e_radius        : spatial radius for ΔEab neighbors (default: 1.0)\n"
              << "  delta_e_bins          : histogram bins (default: 11)\n"
              << "  delta_e_max           : max ΔEab mapped to last bin (default: 50.0)\n";
    return 1;
  }
  
  // Parse args
  std::string splat_file = argv[1];
  float voxel_size = std::atof(argv[2]);
  bool f1 = std::atoi(argv[3]) != 0;
  bool f2 = std::atoi(argv[4]) != 0;
  bool f3 = std::atoi(argv[5]) != 0;
  bool use_color = std::atoi(argv[6]) != 0;
  bool use_opacity = std::atoi(argv[7]) != 0;
  int bins = std::atoi(argv[8]);
  float noise = std::atof(argv[9]);
  float color_weight = std::atof(argv[10]);
  unsigned int seed = std::atoi(argv[11]);
  float max_translation = (argc > 12) ? std::atof(argv[12]) : 5.0f;
  float max_rotation_deg = (argc > 13) ? std::atof(argv[13]) : 45.0f;
  bool centroid_voxel = (argc > 14) ? std::atoi(argv[14]) != 0 : true;
  std::string experiment = (argc > 15) ? argv[15] : "baseline";
  std::string matcher_type = (argc > 16) ? argv[16] : "color";
  bool use_joint_5d    = (argc > 17) ? std::atoi(argv[17]) != 0 : false;
  float voxel_size_ab  = (argc > 18) ? std::atof(argv[18]) : 5.0f;
  int min_color_bins   = (argc > 19) ? std::atoi(argv[19]) : 2;
  bool use_cross_color_filter   = (argc > 20) ? std::atoi(argv[20]) != 0 : false;
  float cross_color_min_overlap = (argc > 21) ? std::atof(argv[21]) : 0.1f;
  bool use_delta_e_hist  = (argc > 22) ? std::atoi(argv[22]) != 0 : false;
  float delta_e_radius   = (argc > 23) ? std::atof(argv[23]) : 1.0f;
  int   delta_e_bins     = (argc > 24) ? std::atoi(argv[24]) : 11;
  float delta_e_max      = (argc > 25) ? std::atof(argv[25]) : 50.0f;
  bool  use_color_pfh    = (argc > 26) ? std::atoi(argv[26]) != 0 : false;
  float color_pfh_radius = (argc > 27) ? std::atof(argv[27]) : 1.0f;
  int   color_pfh_bins   = (argc > 28) ? std::atoi(argv[28]) : 11;
  float color_pfh_max_de = (argc > 29) ? std::atof(argv[29]) : 100.0f;
  const bool use_vanilla = (matcher_type == "vanilla");
  if (!use_vanilla && matcher_type != "color") {
    std::cerr << "Unknown matcher mode '" << matcher_type << "'. Expected color|vanilla." << std::endl;
    return 1;
  }
  
  // Build config
  // Validate: at least one feature must be selected
  bool has_any = f1 || f2 || f3 || use_color || use_opacity || use_delta_e_hist || use_color_pfh;
  if (!use_vanilla && !has_any) {
    std::cerr << "Invalid config: no features selected" << std::endl;
    return 1;
  }

  // Compute total descriptor dimension
  int fpfh_dim = 0;
  if (f1) fpfh_dim += bins;
  if (f2) fpfh_dim += bins;
  if (f3) fpfh_dim += bins;
  int color_dim = 0;
  if (use_color) color_dim += 3;  // L*, a*, b*
  if (use_opacity) color_dim += 1;  // alpha
  int total_dim = fpfh_dim + color_dim;
  
  // Load splat data
  std::vector<GaussianSplat> splats;
  if (!loadFile(splat_file, splats)) {
    std::cerr << "Failed to load file: " << splat_file << std::endl;
    return 1;
  }
  
  // Extract source (original)
  std::vector<Eigen::Vector3f> src_points, tgt_points;
  std::vector<Eigen::Vector4f> src_colors, tgt_colors;
  extractPointsAndColors(splats, src_points, src_colors);

  if (experiment == "split_half") {
    // Use immutable copies as split input; outputs are written to src/tgt.
    const auto all_points = src_points;
    const auto all_colors = src_colors;
    splitHalfCloud(all_points, all_colors,
                   0,      // split axis x
                   0.25f,  // overlap ratio
                   src_points, src_colors, tgt_points, tgt_colors);
    if (src_points.empty() || tgt_points.empty()) {
      std::cerr << "split_half produced empty source/target" << std::endl;
      return 1;
    }
  } else {
    // Create target as transformed + noisy copy
    tgt_points = src_points;
    tgt_colors = src_colors;
  }
  
  // Random generator with seed
  std::mt19937 gen(seed);
  
  // Apply random rigid transform to target
  Eigen::Matrix4f gt_transform;
  applyRandomTransform(tgt_points, gt_transform, 
                       max_translation,
                       max_rotation_deg,
                       gen);
  
  // Add position noise to target
  addPositionNoise(tgt_points, noise, gen);
  
  if (experiment == "baseline") {
    addColorNoise(tgt_colors, noise, gen);
  } else if (experiment == "outliers") {
    addColorNoise(tgt_colors, noise, gen);
    addOutliers(tgt_points, tgt_colors,
                0.30f,                          // 30% points receive outlier perturbation
                std::max(1e-3f, noise * 50.0f), // very strong geometric outliers
                std::max(1e-2f, noise * 80.0f), // very strong color outliers in Lab
                std::max(5e-3f, noise * 0.8f),  // very strong opacity outliers
                gen);
  } else if (experiment == "color_offset") {
    applyGlobalColorOffset(tgt_colors,
                           std::max(1e-3f, noise * 6.0f),
                           std::max(1e-4f, noise * 2.0f),
                           std::max(1e-4f, noise * 0.05f),
                           gen);
  } else if (experiment == "split_half") {
    addColorNoise(tgt_colors, noise, gen);
  } else if (experiment == "clean") {
    // reference for the photometric/scale perturbations: rigid transform + position noise only
  } else if (experiment.rfind("exposure_", 0) == 0) {
    const float g = std::stof(experiment.substr(9));
    applyLinearGains(tgt_colors, g, g, g);
  } else if (experiment == "wb_warm") {
    applyLinearGains(tgt_colors, 1.15f, 1.0f, 0.85f);
  } else if (experiment == "wb_cool") {
    applyLinearGains(tgt_colors, 0.85f, 1.0f, 1.15f);
  } else if (experiment.rfind("gamma_", 0) == 0) {
    applyGammaCurve(tgt_colors, std::stof(experiment.substr(6)));
  } else if (experiment.rfind("shadow_", 0) == 0) {
    applyHalfShadow(src_points, tgt_colors, std::stof(experiment.substr(7)));
  } else if (experiment.rfind("scale_", 0) == 0) {
    applyScaleAboutCentroid(tgt_points, std::stof(experiment.substr(6)));
  } else {
    std::cerr << "Unknown experiment mode '" << experiment
              << "'. Expected baseline|outliers|color_offset|split_half|clean|exposure_<g>|"
                 "gamma_<g>|wb_warm|wb_cool|shadow_<g>|scale_<s>." << std::endl;
    return 1;
  }
  
  // Build config name
  std::string name;
  if (use_vanilla) {
    total_dim = 33;
    name = "vanillaKISS_33D";
    if (centroid_voxel) name += "_centroid";
    name += "_exp_" + experiment;
  } else {
    if (f1) name += "f1";
    if (f2) name += "f2";
    if (f3) name += "f3";
    if (name.empty()) name = "noFPFH";
    if (bins != 11 && !name.empty() && name != "noFPFH") name += "_b" + std::to_string(bins);
    if (use_color) name += "+RGB";
    if (use_opacity) name += "+A";
    name += "_" + std::to_string(total_dim) + "D";
    if (use_joint_5d)        name += "_joint5D_ab" + std::to_string((int)voxel_size_ab)
                                               + "_mb" + std::to_string(min_color_bins);
    else if (centroid_voxel) name += "_centroid";
    if (use_cross_color_filter) name += "_ccf" + std::to_string((int)(cross_color_min_overlap * 100));
    if (use_delta_e_hist)       name += "_de" + std::to_string((int)delta_e_bins);
    if (use_color_pfh)          name += "_cpfh" + std::to_string((int)color_pfh_bins);
    name += "_exp_" + experiment;
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  RegistrationSolution solution;
  size_t num_corr = 0;
  double total_ms = 0.0;
  float rot_err = -1.0f;
  float trans_err = -1.0f;

  try {
    if (use_vanilla) {
      kiss_matcher::KISSMatcherConfig vanilla_config(voxel_size, true, false, 1.0f, 5000);
      kiss_matcher::KISSMatcher vanilla_matcher(vanilla_config);
      solution = vanilla_matcher.estimate(src_points, tgt_points);
      num_corr = vanilla_matcher.getNumFinalInliers();
    } else {
      ColorKISSMatcherConfig matcher_config(voxel_size, true, false, 1.0f, 5000, color_weight,
                                            f1, f2, f3, use_color, use_opacity, bins, centroid_voxel,
                                            use_joint_5d, voxel_size_ab, min_color_bins,
                                            use_cross_color_filter, cross_color_min_overlap,
                                            use_delta_e_hist, delta_e_radius, delta_e_bins, delta_e_max,
                                            use_color_pfh, color_pfh_radius, color_pfh_bins, color_pfh_max_de);
      ColorKISSMatcher matcher(matcher_config);
      solution = matcher.estimate(src_points, tgt_points, src_colors, tgt_colors);
      num_corr = matcher.getNumInliers();
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    total_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // Convert solution to 4x4 matrix
    Eigen::Matrix4f estimated = Eigen::Matrix4f::Identity();
    estimated.block<3, 3>(0, 0) = solution.rotation.cast<float>();
    estimated.block<3, 1>(0, 3) = solution.translation.cast<float>();

    // Compute errors: estimated should match gt_transform (both align src to tgt)
    rot_err = computeRotationError(estimated, gt_transform);
    trans_err = computeTranslationError(estimated, gt_transform);
  } catch (const std::exception& e) {
    auto t2 = std::chrono::high_resolution_clock::now();
    total_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::cerr << "[single_ablation] matcher failed: " << e.what() << std::endl;
  }

  // Output CSV line
  // config,dimension,f1,f2,f3,color,opacity,bins,noise,color_weight,centroid_voxel,
  //   joint_5d,voxel_size_ab,min_color_bins,seed,total_ms,correspondences,rotation_error_deg,translation_error
  std::cout << name << ","
            << total_dim << ","
            << f1 << "," << f2 << "," << f3 << ","
            << use_color << "," << use_opacity << "," << bins << ","
            << noise << "," << color_weight << "," << centroid_voxel << ","
            << use_joint_5d << "," << voxel_size_ab << "," << min_color_bins << ","
            << use_cross_color_filter << "," << cross_color_min_overlap << ","
            << use_delta_e_hist << "," << delta_e_radius << "," << delta_e_bins << "," << delta_e_max << ","
            << use_color_pfh << "," << color_pfh_radius << "," << color_pfh_bins << "," << color_pfh_max_de << ","
            << seed << ","
            << total_ms << ","
            << num_corr << ","
            << rot_err << ","
            << trans_err << std::endl;

  return 0;
  // Memory leaks on exit, but process terminates so OS reclaims everything
}

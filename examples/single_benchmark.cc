/**
 * Single benchmark run - designed to be called from a bash script
 * to avoid memory leaks accumulating across iterations.
 * 
 * Usage: ./single_benchmark <splat_file> <voxel_size> <color_weight> <noise_std> <seed> <matcher_type>
 * 
 * Output: CSV line to stdout
 */

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <kiss_matcher/KISSMatcher.hpp>
#include <3dgs_matcher/ColorKISSMatcher.hpp>

// RAII class to suppress cout during matcher operations
class SuppressCout {
public:
  SuppressCout() : old_buf_(std::cout.rdbuf()) {
    std::cout.rdbuf(null_stream_.rdbuf());
  }
  ~SuppressCout() {
    std::cout.rdbuf(old_buf_);
  }
private:
  std::streambuf* old_buf_;
  std::ostringstream null_stream_;
};

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

void addGaussianNoise(std::vector<Eigen::Vector3f>& points,
                     float noise_std,
                     std::mt19937& gen) {
  std::normal_distribution<float> noise_dist(0.0f, noise_std);

  for (auto& pt : points) {
    pt.x() += noise_dist(gen);
    pt.y() += noise_dist(gen);
    pt.z() += noise_dist(gen);
  }
}

float computePointToPointRMSE(const std::vector<Eigen::Vector3f>& source,
                               const std::vector<Eigen::Vector3f>& target,
                               const Eigen::Matrix4d& transform) {
  if (source.size() != target.size()) {
    return std::numeric_limits<float>::max();
  }

  double sum_squared_error = 0.0;
  size_t count = std::min(source.size(), static_cast<size_t>(10000));  // Sample for speed
  size_t step = source.size() / count;
  
  for (size_t i = 0; i < source.size(); i += step) {
    Eigen::Vector4d src_h(source[i].x(), source[i].y(), source[i].z(), 1.0);
    Eigen::Vector4d src_transformed = transform * src_h;
    Eigen::Vector3d src_aligned = src_transformed.head<3>();
    
    Eigen::Vector3d tgt(target[i].x(), target[i].y(), target[i].z());
    double dist = (src_aligned - tgt).norm();
    sum_squared_error += dist * dist;
  }

  return std::sqrt(sum_squared_error / (source.size() / step));
}

void printUsage(const char* progname) {
  std::cerr << "Usage: " << progname << " <splat_file> <voxel_size> <color_weight> <noise_std> <seed> <matcher_type> [max_translation] [max_rotation_deg]\n";
  std::cerr << "  matcher_type: 'color' or 'vanilla'\n";
  std::cerr << "  max_translation: max translation in each axis (default: 2.0)\n";
  std::cerr << "  max_rotation_deg: max rotation per axis in degrees (default: 15.0)\n";
  std::cerr << "\nOutput: CSV line with columns:\n";
  std::cerr << "  color_weight,noise_level,seed,matcher_type,rotation_error_deg,translation_error,point_rmse,execution_time_ms\n";
}

int main(int argc, char** argv) {
  if (argc < 7 || argc > 9) {
    printUsage(argv[0]);
    return 1;
  }

  std::string splat_file = argv[1];
  float voxel_size = std::stof(argv[2]);
  float color_weight = std::stof(argv[3]);
  float noise_std = std::stof(argv[4]);
  unsigned int seed = static_cast<unsigned int>(std::stoul(argv[5]));
  std::string matcher_type = argv[6];
  
  // Optional parameters for initial transformation magnitude
  float max_translation = (argc > 7) ? std::stof(argv[7]) : 2.0f;
  float max_rotation_deg = (argc > 8) ? std::stof(argv[8]) : 15.0f;

  if (matcher_type != "color" && matcher_type != "vanilla") {
    std::cerr << "Invalid matcher_type: " << matcher_type << std::endl;
    printUsage(argv[0]);
    return 1;
  }

  // Load splat file
  std::vector<GaussianSplat> splats;
  if (!loadSplatFile(splat_file, splats)) {
    return 1;
  }

  std::vector<Eigen::Vector3f> src_points, tgt_points;
  std::vector<Eigen::Vector4f> src_colors, tgt_colors;
  extractPointsAndColors(splats, src_points, src_colors);
  tgt_points = src_points;
  tgt_colors = src_colors;

  // Apply random transform
  // Use configurable values for testing different initial offsets
  std::mt19937 gen(seed);
  Eigen::Matrix4f ground_truth;
  applyRandomTransform(tgt_points, ground_truth, max_translation, max_rotation_deg, gen);

  // Add noise to target point cloud only (source stays clean)
  if (noise_std > 0.0f) {
    addGaussianNoise(tgt_points, noise_std, gen);
  }

  // Run matcher (suppress stdout from matcher internals)
  Eigen::Matrix4d estimated = Eigen::Matrix4d::Identity();
  double execution_time_ms = 0.0;

  {
    SuppressCout suppress;
    
    if (matcher_type == "vanilla") {
      auto start = std::chrono::high_resolution_clock::now();
      
      kiss_matcher::KISSMatcherConfig config(voxel_size);
      kiss_matcher::KISSMatcher matcher(config);
      auto solution = matcher.estimate(src_points, tgt_points);
      
      auto end = std::chrono::high_resolution_clock::now();
      execution_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
      
      estimated.block<3,3>(0,0) = solution.rotation;
      estimated.block<3,1>(0,3) = solution.translation;
    } else {
      auto start = std::chrono::high_resolution_clock::now();
      
      gs3d_matcher::ColorKISSMatcherConfig config(voxel_size, true, false, 1.0, 5000, color_weight);
      gs3d_matcher::ColorKISSMatcher matcher(config);
      auto solution = matcher.estimate(src_points, tgt_points, src_colors, tgt_colors);
      
      auto end = std::chrono::high_resolution_clock::now();
      execution_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
      
      estimated.block<3,3>(0,0) = solution.rotation;
      estimated.block<3,1>(0,3) = solution.translation;
    }
  }  // suppress goes out of scope, restoring cout

  // Compute errors
  Eigen::Matrix4f est_f = estimated.cast<float>();
  
  // Rotation error
  Eigen::Matrix3f R_gt = ground_truth.block<3, 3>(0, 0);
  Eigen::Matrix3f R_est = est_f.block<3, 3>(0, 0);
  Eigen::Matrix3f R_error = R_gt.transpose() * R_est;
  float trace = R_error.trace();
  float rotation_error_rad = std::acos(std::min(1.0f, std::max(-1.0f, (trace - 1.0f) / 2.0f)));
  float rotation_error_deg = rotation_error_rad * 180.0f / M_PI;
  
  // Translation error
  Eigen::Vector3f t_gt = ground_truth.block<3, 1>(0, 3);
  Eigen::Vector3f t_est = est_f.block<3, 1>(0, 3);
  float translation_error = (t_gt - t_est).norm();
  
  // Point-to-point RMSE
  float point_rmse = computePointToPointRMSE(src_points, tgt_points, estimated);

  // Output CSV line
  std::cout << std::fixed << std::setprecision(6)
            << color_weight << ","
            << noise_std << ","
            << seed << ","
            << matcher_type << ","
            << rotation_error_deg << ","
            << translation_error << ","
            << point_rmse << ","
            << execution_time_ms << std::endl;

  return 0;
}

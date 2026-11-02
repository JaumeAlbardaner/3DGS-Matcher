/**
 * Benchmark: Compare KISS-Matcher vs ColorKISSMatcher performance
 * 
 * Tests both matchers across different noise levels with multiple iterations
 * to evaluate the benefit of using color and opacity information.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/registration/gicp.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <kiss_matcher/KISSMatcher.hpp>
#include <3dgs_matcher/ColorKISSMatcher.hpp>

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
                           std::vector<Eigen::Vector4f>& colors,
                           float opacity_threshold = 0.0f) {
  points.clear();
  colors.clear();
  points.reserve(splats.size());
  colors.reserve(splats.size());

  for (const auto& splat : splats) {
    if (splat.color_rgba.w() >= opacity_threshold) {
      points.push_back(splat.position);
      colors.push_back(splat.color_rgba);
    }
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

struct BenchmarkResult {
  float rotation_error_deg;
  float translation_error;
  double execution_time_ms;
  float point_to_point_rmse;  // RMS point-to-point distance after alignment
  
  // GICP refinement results
  float gicp_rotation_error_deg;
  float gicp_translation_error;
  double gicp_execution_time_ms;
  float gicp_point_to_point_rmse;
};

/**
 * @brief Refine transformation using GICP
 */
Eigen::Matrix4d refineWithGICP(const std::vector<Eigen::Vector3f>& src_points,
                               const std::vector<Eigen::Vector3f>& tgt_points,
                               const Eigen::Matrix4d& initial_transform,
                               double max_correspondence_distance,
                               double& gicp_time_ms) {
  // Convert to PCL point clouds
  pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr tgt_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  
  src_cloud->reserve(src_points.size());
  tgt_cloud->reserve(tgt_points.size());
  
  for (const auto& pt : src_points) {
    src_cloud->push_back(pcl::PointXYZ(pt.x(), pt.y(), pt.z()));
  }
  
  for (const auto& pt : tgt_points) {
    tgt_cloud->push_back(pcl::PointXYZ(pt.x(), pt.y(), pt.z()));
  }
  
  // Setup GICP
  pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
  gicp.setInputSource(src_cloud);
  gicp.setInputTarget(tgt_cloud);
  gicp.setMaxCorrespondenceDistance(max_correspondence_distance);
  gicp.setMaximumIterations(50);
  gicp.setTransformationEpsilon(1e-8);
  gicp.setEuclideanFitnessEpsilon(1e-6);
  
  // Perform GICP alignment
  pcl::PointCloud<pcl::PointXYZ> aligned;
  auto start = std::chrono::high_resolution_clock::now();
  gicp.align(aligned, initial_transform.cast<float>());
  auto end = std::chrono::high_resolution_clock::now();
  gicp_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
  
  if (!gicp.hasConverged()) {
    // Return initial transform if GICP failed to converge
    return initial_transform;
  }
  
  return gicp.getFinalTransformation().cast<double>();
};

float computePointToPointRMSE(const std::vector<Eigen::Vector3f>& source,
                               const std::vector<Eigen::Vector3f>& target,
                               const Eigen::Matrix4d& transform) {
  if (source.size() != target.size()) {
    return std::numeric_limits<float>::max();
  }

  double sum_squared_error = 0.0;
  for (size_t i = 0; i < source.size(); ++i) {
    Eigen::Vector4d src_h(source[i].x(), source[i].y(), source[i].z(), 1.0);
    Eigen::Vector4d src_transformed = transform * src_h;
    Eigen::Vector3d src_aligned = src_transformed.head<3>();
    
    Eigen::Vector3d tgt(target[i].x(), target[i].y(), target[i].z());
    double dist = (src_aligned - tgt).norm();
    sum_squared_error += dist * dist;
  }

  return std::sqrt(sum_squared_error / source.size());
}

BenchmarkResult runVanillaKISSMatcher(const std::vector<Eigen::Vector3f>& src_points,
                                      const std::vector<Eigen::Vector3f>& tgt_points,
                                      const Eigen::Matrix4f& ground_truth,
                                      float voxel_size) {
  BenchmarkResult result;
  
  auto start = std::chrono::high_resolution_clock::now();
  
  kiss_matcher::KISSMatcherConfig config(voxel_size);
  kiss_matcher::KISSMatcher matcher(config);
  
  auto solution = matcher.estimate(src_points, tgt_points);
  
  auto end = std::chrono::high_resolution_clock::now();
  result.execution_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
  
  // Compute errors
  Eigen::Matrix4d estimated = Eigen::Matrix4d::Identity();
  estimated.block<3,3>(0,0) = solution.rotation;
  estimated.block<3,1>(0,3) = solution.translation;
  
  Eigen::Matrix4f est_f = estimated.cast<float>();
  
  // Rotation error
  Eigen::Matrix3f R_gt = ground_truth.block<3, 3>(0, 0);
  Eigen::Matrix3f R_est = est_f.block<3, 3>(0, 0);
  Eigen::Matrix3f R_error = R_gt.transpose() * R_est;
  float trace = R_error.trace();
  float rotation_error_rad = std::acos(std::min(1.0f, std::max(-1.0f, (trace - 1.0f) / 2.0f)));
  result.rotation_error_deg = rotation_error_rad * 180.0f / M_PI;
  
  // Translation error
  Eigen::Vector3f t_gt = ground_truth.block<3, 1>(0, 3);
  Eigen::Vector3f t_est = est_f.block<3, 1>(0, 3);
  result.translation_error = (t_gt - t_est).norm();
  
  // Point-to-point RMSE
  result.point_to_point_rmse = computePointToPointRMSE(src_points, tgt_points, estimated);
  
  return result;
}

BenchmarkResult runColorKISSMatcher(const std::vector<Eigen::Vector3f>& src_points,
                                    const std::vector<Eigen::Vector3f>& tgt_points,
                                    const std::vector<Eigen::Vector4f>& src_colors,
                                    const std::vector<Eigen::Vector4f>& tgt_colors,
                                    const Eigen::Matrix4f& ground_truth,
                                    float voxel_size,
                                    float color_weight) {
  BenchmarkResult result;
  
  auto start = std::chrono::high_resolution_clock::now();
  
  gs3d_matcher::ColorKISSMatcherConfig config(voxel_size, true, false, 1.0, 5000, color_weight);
  gs3d_matcher::ColorKISSMatcher matcher(config);
  
  auto solution = matcher.estimate(src_points, tgt_points, src_colors, tgt_colors);
  
  auto end = std::chrono::high_resolution_clock::now();
  result.execution_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
  
  // Compute errors
  Eigen::Matrix4d estimated = Eigen::Matrix4d::Identity();
  estimated.block<3,3>(0,0) = solution.rotation;
  estimated.block<3,1>(0,3) = solution.translation;
  
  Eigen::Matrix4f est_f = estimated.cast<float>();
  
  // Rotation error
  Eigen::Matrix3f R_gt = ground_truth.block<3, 3>(0, 0);
  Eigen::Matrix3f R_est = est_f.block<3, 3>(0, 0);
  Eigen::Matrix3f R_error = R_gt.transpose() * R_est;
  float trace = R_error.trace();
  float rotation_error_rad = std::acos(std::min(1.0f, std::max(-1.0f, (trace - 1.0f) / 2.0f)));
  result.rotation_error_deg = rotation_error_rad * 180.0f / M_PI;
  
  // Translation error
  Eigen::Vector3f t_gt = ground_truth.block<3, 1>(0, 3);
  Eigen::Vector3f t_est = est_f.block<3, 1>(0, 3);
  result.translation_error = (t_gt - t_est).norm();
  
  // Point-to-point RMSE
  result.point_to_point_rmse = computePointToPointRMSE(src_points, tgt_points, estimated);
  
  // GICP disabled for benchmark speed (can take minutes on 600K points)
  result.gicp_rotation_error_deg = 0.0f;
  result.gicp_translation_error = 0.0f;
  result.gicp_execution_time_ms = 0.0;
  result.gicp_point_to_point_rmse = 0.0f;
  
  return result;
}

int main(int argc, char** argv) {
  std::string splat_file = "../examples/splats/tramway.splat";
  float voxel_size = 0.3f;
  int num_iterations = 50;
  float max_translation = 100.0f;
  float max_rotation_deg = 180.0f;
  float min_color_weight = 0.0f;
  float max_color_weight = 100.0f;
  float color_weight_step = 5.0f;
  float min_noise = 0.0f;
  float max_noise = 0.5f;
  float noise_step = 0.05f;
  std::string output_file = "benchmark_results.csv";

  if (argc > 1) splat_file = argv[1];
  if (argc > 2) voxel_size = std::stof(argv[2]);
  if (argc > 3) num_iterations = std::stoi(argv[3]);
  if (argc > 4) min_color_weight = std::stof(argv[4]);
  if (argc > 5) max_color_weight = std::stof(argv[5]);
  if (argc > 6) color_weight_step = std::stof(argv[6]);
  if (argc > 7) min_noise = std::stof(argv[7]);
  if (argc > 8) max_noise = std::stof(argv[8]);
  if (argc > 9) noise_step = std::stof(argv[9]);
  if (argc > 10) output_file = argv[10];

  std::cout << "=== ColorKISSMatcher vs KISS-Matcher Benchmark ===" << std::endl;
  std::cout << "Splat file: " << splat_file << std::endl;
  std::cout << "Voxel size: " << voxel_size << std::endl;
  std::cout << "Iterations per configuration: " << num_iterations << std::endl;
  std::cout << "Color weight range: [" << min_color_weight << ", " << max_color_weight 
            << "] step " << color_weight_step << std::endl;
  std::cout << "Noise range: [" << min_noise << ", " << max_noise 
            << "] step " << noise_step << std::endl;
  std::cout << "Output file: " << output_file << std::endl;

  // Load splat file
  std::vector<GaussianSplat> splats;
  if (!loadSplatFile(splat_file, splats)) {
    return 1;
  }

  std::vector<Eigen::Vector3f> original_points;
  std::vector<Eigen::Vector4f> original_colors;
  extractPointsAndColors(splats, original_points, original_colors, 0.0f);

  std::cout << "Loaded " << original_points.size() << " points" << std::endl;

  // Open output file
  std::ofstream outfile(output_file);
  outfile << "color_weight,noise_level,iteration,matcher_type,rotation_error_deg,translation_error,point_rmse,execution_time_ms,"
          << "gicp_rotation_error_deg,gicp_translation_error,gicp_point_rmse,gicp_execution_time_ms\n";
  outfile << std::fixed << std::setprecision(6);

  // Random generator with fixed seed
  std::mt19937 gen(42);

  // Generate color weight values
  std::vector<float> color_weights;
  for (float cw = min_color_weight; cw <= max_color_weight + 1e-6; cw += color_weight_step) {
    color_weights.push_back(cw);
  }

  // Generate noise levels
  std::vector<float> noise_levels;
  for (float noise = min_noise; noise <= max_noise + 1e-6; noise += noise_step) {
    noise_levels.push_back(noise);
  }

  int total_configs = color_weights.size() * noise_levels.size();
  int completed_configs = 0;
  int total_tests = total_configs * num_iterations * 2;  // 2 matchers per iteration

  std::cout << "\nTotal configurations: " << total_configs 
            << " (" << color_weights.size() << " color weights × " 
            << noise_levels.size() << " noise levels)" << std::endl;
  std::cout << "Total tests: " << total_tests << " (" << num_iterations << " iterations × 2 matchers)\n" << std::endl;

  for (float color_weight : color_weights) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Color weight: " << color_weight << std::endl;
    std::cout << "========================================" << std::endl;

    for (float noise_std : noise_levels) {
      std::cout << "  Noise level: " << noise_std << " ... " << std::flush;

      for (int iter = 0; iter < num_iterations; ++iter) {
        // Create fresh copies for this iteration
        std::vector<Eigen::Vector3f> src_points = original_points;
        std::vector<Eigen::Vector3f> tgt_points = original_points;
        std::vector<Eigen::Vector4f> src_colors = original_colors;
        std::vector<Eigen::Vector4f> tgt_colors = original_colors;

        // Apply transformation
        Eigen::Matrix4f ground_truth_transform;
        applyRandomTransform(tgt_points, ground_truth_transform, max_translation, max_rotation_deg, gen);

        // Add noise (only to coordinates)
        if (noise_std > 0.0f) {
          addGaussianNoise(tgt_points, noise_std, gen);
        }

        // Run vanilla KISS-Matcher (only once per noise level, since color_weight doesn't affect it)
        if (color_weight == color_weights[0]) {
          auto vanilla_result = runVanillaKISSMatcher(src_points, tgt_points, ground_truth_transform, voxel_size);
          
          // Write vanilla results
          outfile << color_weight << "," 
                  << noise_std << "," 
                  << iter << ","
                  << "vanilla,"
                  << vanilla_result.rotation_error_deg << ","
                  << vanilla_result.translation_error << ","
                  << vanilla_result.point_to_point_rmse << ","
                  << vanilla_result.execution_time_ms << ","
                  << vanilla_result.gicp_rotation_error_deg << ","
                  << vanilla_result.gicp_translation_error << ","
                  << vanilla_result.gicp_point_to_point_rmse << ","
                  << vanilla_result.gicp_execution_time_ms << "\n";
        }

        // Run ColorKISSMatcher with current color_weight
        auto color_result = runColorKISSMatcher(src_points, tgt_points, src_colors, tgt_colors,
                                                ground_truth_transform, voxel_size, color_weight);
        
        // Write color results
        outfile << color_weight << "," 
                << noise_std << "," 
                << iter << ","
                << "color,"
                << color_result.rotation_error_deg << ","
                << color_result.translation_error << ","
                << color_result.point_to_point_rmse << ","
                << color_result.execution_time_ms << ","
                << color_result.gicp_rotation_error_deg << ","
                << color_result.gicp_translation_error << ","
                << color_result.gicp_point_to_point_rmse << ","
                << color_result.gicp_execution_time_ms << "\n";

        // Flush periodically
        if ((iter + 1) % 10 == 0) {
          outfile.flush();
        }
      }

      completed_configs++;
      std::cout << "done (" << completed_configs << "/" << total_configs << " = " 
                << (100 * completed_configs / total_configs) << "%)" << std::endl;
      outfile.flush();
    }
  }

  outfile.close();
  
  std::cout << "\n========================================" << std::endl;
  std::cout << "=== Benchmark Complete ===" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Results saved to: " << output_file << std::endl;
  std::cout << "Configurations tested: " << total_configs << std::endl;
  std::cout << "  - Color weights: " << color_weights.size() << " values from " 
            << min_color_weight << " to " << max_color_weight << std::endl;
  std::cout << "  - Noise levels: " << noise_levels.size() << " values from " 
            << min_noise << " to " << max_noise << std::endl;
  std::cout << "  - Iterations per config: " << num_iterations << std::endl;

  return 0;
}

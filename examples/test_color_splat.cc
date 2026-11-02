/**
 * Example: Load Gaussian .splat file and test ColorKISSMatcher with transformations and noise
 * 
 * This example loads a .splat file, applies translation, rotation, and per-point noise,
 * then evaluates ColorKISSMatcher performance.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/gicp.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <3dgs_matcher/ColorKISSMatcher.hpp>

/**
 * @brief Structure to hold Gaussian splat data
 * 
 * Format matches LiDAR-3DGS-Relocalization:
 * - 32 bytes per splat: position(3f) + scales(3f) + color(4B RGBA) + rotation(4B)
 */
struct GaussianSplat {
  Eigen::Vector3f position;
  Eigen::Vector4f color_rgba;  // RGBA (normalized to [0,1])
  // Additional fields in the .splat format (rotation quaternion) are ignored for matching
};

/**
 * @brief Load .splat file (32-byte or 48-byte format)
 * @param filename Path to .splat file
 * @param splats Output vector of splats
 * @return true if successful
 */
bool loadSplatFile(const std::string& filename, std::vector<GaussianSplat>& splats) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return false;
  }

  // Get file size
  file.seekg(0, std::ios::end);
  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  size_t num_splats = file_size / 32;
  std::cout << "Loading " << num_splats << " splats (32 bytes per splat)..." << std::endl;

  splats.reserve(num_splats);

  // Read splats
  for (size_t i = 0; i < num_splats; ++i) {
    GaussianSplat splat;
    // Standard format: position(3f) + scales(3f) + color(4B RGBA) + rotation(4B)
    
    // Read position (3 floats = 12 bytes)
    float position[3];
    file.read(reinterpret_cast<char*>(position), sizeof(float) * 3);
    splat.position = Eigen::Vector3f(position[0], position[1], position[2]);
    
    // Read scales (3 floats = 12 bytes) - skip, not needed for matching
    float scales[3];
    file.read(reinterpret_cast<char*>(scales), sizeof(float) * 3);
          
    // Read color as 4 uint8 bytes (RGBA)
    uint8_t color_rgba[4];
    file.read(reinterpret_cast<char*>(color_rgba), 4);
          
    // Convert color from [0, 255] to [0, 1]
    splat.color_rgba = Eigen::Vector4f(
        color_rgba[0] / 255.0f,  // R
        color_rgba[1] / 255.0f,  // G
        color_rgba[2] / 255.0f,  // B
        color_rgba[3] / 255.0f   // A (opacity)
    );
    
    // Read rotation (4 uint8 bytes) - skip, not needed for matching
    uint8_t rotation[4];
    file.read(reinterpret_cast<char*>(rotation), 4);
    
    splats.push_back(splat);
  }

  std::cout << "Loaded " << splats.size() << " splats successfully." << std::endl;
  
  // Print bounding box to verify data
  if (!splats.empty()) {
    Eigen::Vector3f min_pt = splats[0].position;
    Eigen::Vector3f max_pt = splats[0].position;
    for (const auto& s : splats) {
      min_pt = min_pt.cwiseMin(s.position);
      max_pt = max_pt.cwiseMax(s.position);
    }
    std::cout << "Bounding box: Min=(" << min_pt.transpose() << "), Max=(" 
              << max_pt.transpose() << "), Extent=(" << (max_pt - min_pt).transpose() << ")" << std::endl;
  }
  
  return true;
}

/**
 * @brief Extract positions and colors from splats with optional opacity filtering
 * @param opacity_threshold Minimum opacity to include (0.0-1.0). Set to 0.0 to include all splats.
 */
void extractPointsAndColors(const std::vector<GaussianSplat>& splats,
                           std::vector<Eigen::Vector3f>& points,
                           std::vector<Eigen::Vector4f>& colors,
                           float opacity_threshold = 0.0f) {
  points.clear();
  colors.clear();
  points.reserve(splats.size());
  colors.reserve(splats.size());

  size_t filtered_count = 0;
  for (const auto& splat : splats) {
    // Filter by opacity if threshold is set
    if (splat.color_rgba.w() >= opacity_threshold) {
      points.push_back(splat.position);
      colors.push_back(splat.color_rgba);
    } else {
      filtered_count++;
    }
  }

  if (filtered_count > 0) {
    std::cout << "Filtered out " << filtered_count << " splats with opacity < " 
              << opacity_threshold << std::endl;
  }
}

/**
 * @brief Apply random transformation (rotation + translation) to point cloud
 */
void applyRandomTransform(std::vector<Eigen::Vector3f>& points,
                         Eigen::Matrix4f& transform_out,
                         float max_translation = 50.0f,
                         float max_rotation_deg = 180.0f,
                         int seed = 42) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> trans_dist(-max_translation, max_translation);
  std::uniform_real_distribution<float> rot_dist(-max_rotation_deg, max_rotation_deg);

  // Generate random rotation (Euler angles)
  float roll_deg = rot_dist(gen);
  float pitch_deg = rot_dist(gen);
  float yaw_deg = rot_dist(gen);

  Eigen::AngleAxisf rollAngle(roll_deg * M_PI / 180.0f, Eigen::Vector3f::UnitX());
  Eigen::AngleAxisf pitchAngle(pitch_deg * M_PI / 180.0f, Eigen::Vector3f::UnitY());
  Eigen::AngleAxisf yawAngle(yaw_deg * M_PI / 180.0f, Eigen::Vector3f::UnitZ());
  
  Eigen::Matrix3f rotation = (yawAngle * pitchAngle * rollAngle).toRotationMatrix();

  // Generate random translation
  Eigen::Vector3f translation(trans_dist(gen), trans_dist(gen), trans_dist(gen));

  // Build 4x4 transformation matrix
  transform_out = Eigen::Matrix4f::Identity();
  transform_out.block<3, 3>(0, 0) = rotation;
  transform_out.block<3, 1>(0, 3) = translation;

  std::cout << "\nApplied transformation:" << std::endl;
  std::cout << "  Translation: (" << translation.transpose() << ")" << std::endl;
  std::cout << "  Rotation (deg): roll=" << roll_deg << ", pitch=" << pitch_deg 
            << ", yaw=" << yaw_deg << std::endl;

  // Apply transformation
  for (auto& pt : points) {
    Eigen::Vector4f pt_h(pt.x(), pt.y(), pt.z(), 1.0f);
    Eigen::Vector4f pt_transformed = transform_out * pt_h;
    pt = pt_transformed.head<3>();
  }
}

/**
 * @brief Add Gaussian noise to point cloud
 */
void addGaussianNoise(std::vector<Eigen::Vector3f>& points,
                     float noise_std = 0.01f,
                     int seed = 42) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> noise_dist(0.0f, noise_std);

  for (auto& pt : points) {
    pt.x() += noise_dist(gen);
    pt.y() += noise_dist(gen);
    pt.z() += noise_dist(gen);
  }

  std::cout << "Added Gaussian noise (std=" << noise_std << ") to all points." << std::endl;
}

/**
 * @brief Refine transformation using GICP
 */
Eigen::Matrix4d refineWithGICP(const std::vector<Eigen::Vector3f>& src_points,
                               const std::vector<Eigen::Vector3f>& tgt_points,
                               const Eigen::Matrix4d& initial_transform,
                               double max_correspondence_distance = 1.0,
                               int max_iterations = 50) {
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
  gicp.setMaximumIterations(max_iterations);
  gicp.setTransformationEpsilon(1e-8);
  gicp.setEuclideanFitnessEpsilon(1e-6);
  
  // Perform GICP alignment with initial guess
  pcl::PointCloud<pcl::PointXYZ> aligned;
  auto start = std::chrono::high_resolution_clock::now();
  gicp.align(aligned, initial_transform.cast<float>());
  auto end = std::chrono::high_resolution_clock::now();
  double gicp_time = std::chrono::duration<double, std::milli>(end - start).count();
  
  std::cout << "\n=== GICP Refinement ==="  << std::endl;
  std::cout << "Converged: " << (gicp.hasConverged() ? "Yes" : "No") << std::endl;
  std::cout << "Fitness score: " << gicp.getFitnessScore() << std::endl;
  std::cout << "Iterations: " << gicp.nr_iterations_ << " / " << max_iterations << std::endl;
  std::cout << "Time: " << gicp_time << " ms" << std::endl;
  
  return gicp.getFinalTransformation().cast<double>();
}

/**
 * @brief Compute registration error
 */
void evaluateRegistration(const Eigen::Matrix4f& ground_truth,
                         const Eigen::Matrix4d& estimated,
                         const std::string& label = "Registration") {
  Eigen::Matrix4f est_f = estimated.cast<float>();
  
  // Rotation error (angle)
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

  std::cout << "\n=== " << label << " Evaluation ===" << std::endl;
  std::cout << "Rotation error: " << rotation_error_deg << " degrees" << std::endl;
  std::cout << "Translation error: " << translation_error << " units" << std::endl;
  
  std::cout << "\nGround truth transformation:" << std::endl;
  std::cout << ground_truth << std::endl;
  std::cout << "\nEstimated transformation:" << std::endl;
  std::cout << estimated << std::endl;
}

/**
 * @brief Visualize point clouds before and after matching
 * @param source_original Original source cloud (untransformed)
 * @param target_transformed Target cloud after transformation and noise
 * @param source_colors Colors for source
 * @param target_colors Colors for target
 * @param estimated_transform Transformation from matcher to align target back to source
 */
void visualizeResults(const std::vector<Eigen::Vector3f>& source_original,
                     const std::vector<Eigen::Vector3f>& target_transformed,
                     const std::vector<Eigen::Vector4f>& source_colors,
                     const std::vector<Eigen::Vector4f>& target_colors,
                     const Eigen::Matrix4d& vanilla_kiss_transform,
                     const Eigen::Matrix4d& color_kiss_transform) {
  
  pcl::visualization::PCLVisualizer::Ptr viewer(
      new pcl::visualization::PCLVisualizer("KISS-Matcher vs ColorKISSMatcher"));
  viewer->setBackgroundColor(0.1, 0.1, 0.1);
  viewer->addCoordinateSystem(1.0);
  viewer->initCameraParameters();

  // Create point clouds for visualization
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_source(
      new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_target_transformed(
      new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_target_aligned(
      new pcl::PointCloud<pcl::PointXYZRGB>);

  // Fill source (original, untransformed)
  for (size_t i = 0; i < source_original.size(); ++i) {
    pcl::PointXYZRGB pt;
    pt.x = source_original[i].x();
    pt.y = source_original[i].y();
    pt.z = source_original[i].z();
    pt.r = static_cast<uint8_t>(source_colors[i].x() * 255);
    pt.g = static_cast<uint8_t>(source_colors[i].y() * 255);
    pt.b = static_cast<uint8_t>(source_colors[i].z() * 255);
    cloud_source->push_back(pt);
  }

  // Fill target (transformed + noisy)
  for (size_t i = 0; i < target_transformed.size(); ++i) {
    pcl::PointXYZRGB pt;
    pt.x = target_transformed[i].x();
    pt.y = target_transformed[i].y();
    pt.z = target_transformed[i].z();
    pt.r = static_cast<uint8_t>(target_colors[i].x() * 255);
    pt.g = static_cast<uint8_t>(target_colors[i].y() * 255);
    pt.b = static_cast<uint8_t>(target_colors[i].z() * 255);
    cloud_target_transformed->push_back(pt);
  }

  // Apply inverse of vanilla KISS transformation to target to align it back to source
  auto vanilla_kiss_inv = std::make_unique<Eigen::Matrix4d>(vanilla_kiss_transform.inverse());
  
  for (size_t i = 0; i < target_transformed.size(); ++i) {
    Eigen::Vector4d pt_h(target_transformed[i].x(), target_transformed[i].y(),
                        target_transformed[i].z(), 1.0);
    Eigen::Vector4d pt_aligned = (*vanilla_kiss_inv) * pt_h;
    
    pcl::PointXYZRGB pt;
    pt.x = pt_aligned.x();
    pt.y = pt_aligned.y();
    pt.z = pt_aligned.z();
    pt.r = static_cast<uint8_t>(target_colors[i].x() * 255);
    pt.g = static_cast<uint8_t>(target_colors[i].y() * 255);
    pt.b = static_cast<uint8_t>(target_colors[i].z() * 255);
    cloud_target_aligned->push_back(pt);
  }

  // Create point cloud for ColorKISS-aligned target
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_target_color_aligned(new pcl::PointCloud<pcl::PointXYZRGB>);
  auto color_kiss_inv = std::make_unique<Eigen::Matrix4d>(color_kiss_transform.inverse());
  
  for (size_t i = 0; i < target_transformed.size(); ++i) {
    Eigen::Vector4d pt_h(target_transformed[i].x(), target_transformed[i].y(),
                        target_transformed[i].z(), 1.0);
    Eigen::Vector4d pt_aligned = (*color_kiss_inv) * pt_h;
    
    pcl::PointXYZRGB pt;
    pt.x = pt_aligned.x();
    pt.y = pt_aligned.y();
    pt.z = pt_aligned.z();
    pt.r = static_cast<uint8_t>(target_colors[i].x() * 255);
    pt.g = static_cast<uint8_t>(target_colors[i].y() * 255);
    pt.b = static_cast<uint8_t>(target_colors[i].z() * 255);
    cloud_target_color_aligned->push_back(pt);
  }

  // Create three viewports
  int v1(0), v2(1), v3(2);
  viewer->createViewPort(0.0, 0.0, 0.33, 1.0, v1);
  viewer->createViewPort(0.33, 0.0, 0.66, 1.0, v2);
  viewer->createViewPort(0.66, 0.0, 1.0, 1.0, v3);

  // Viewport 1: Source + transformed/noisy target before matching (green + red)
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> source_green_v1(
      cloud_source, 0, 255, 0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> target_red_v1(
      cloud_target_transformed, 255, 0, 0);
  
  viewer->addPointCloud<pcl::PointXYZRGB>(cloud_source, source_green_v1, "source_before", v1);
  viewer->addPointCloud<pcl::PointXYZRGB>(cloud_target_transformed, target_red_v1, "target_misaligned", v1);
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "source_before", v1);
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target_misaligned", v1);
  viewer->addText("V1: Before (displaced)", 10, 10, 14, 1, 1, 1, "v1_text", v1);

  // Viewport 2: Vanilla KISS-Matcher result (green + red)
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> source_green_v2(
      cloud_source, 0, 255, 0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> aligned_red_v2(
      cloud_target_aligned, 255, 0, 0);
  
  viewer->addPointCloud<pcl::PointXYZRGB>(cloud_source, source_green_v2, "source_vanilla", v2);
  viewer->addPointCloud<pcl::PointXYZRGB>(cloud_target_aligned, aligned_red_v2, "target_vanilla", v2);
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "source_vanilla", v2);
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target_vanilla", v2);
  viewer->addText("V2: KISS-Matcher", 10, 10, 14, 1, 1, 1, "v2_text", v2);

  // Viewport 3: ColorKISS-Matcher result (green + red)
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> source_green_v3(
      cloud_source, 0, 255, 0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> color_red_v3(
      cloud_target_color_aligned, 255, 0, 0);
  
  viewer->addPointCloud<pcl::PointXYZRGB>(cloud_source, source_green_v3, "source_color", v3);
  viewer->addPointCloud<pcl::PointXYZRGB>(cloud_target_color_aligned, color_red_v3, "target_color", v3);
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "source_color", v3);
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target_color", v3);
  viewer->addText("V3: ColorKISS-Matcher", 10, 10, 14, 1, 1, 1, "v3_text", v3);

  std::cout << "\n=== Visualization Controls ===" << std::endl;
  std::cout << "V1: Before - Green=source, Red=target displaced+noise" << std::endl;
  std::cout << "V2: KISS-Matcher - Green=source, Red=aligned target" << std::endl;
  std::cout << "V3: ColorKISS-Matcher - Green=source, Red=aligned target" << std::endl;
  std::cout << "Yellow overlap = good alignment" << std::endl;
  std::cout << "Press 'q' to close the visualization window" << std::endl;

  // Use the same pattern as run_matching_using_bunny.cc
  while (!viewer->wasStopped()) {
    viewer->spin();
  }
}

int main(int argc, char** argv) {
  // Parse arguments
  std::string splat_file = "../examples/splats/tramway.splat";
  float voxel_size = 0.3f;
  float color_weight = 1.0f;
  float noise_std = 0.01f;
  float max_translation = 20.0f;
  float max_rotation_deg = 90.0f;
  bool enable_visualization = false;  // Default off to avoid display issues

  if (argc > 1) splat_file = argv[1];
  if (argc > 2) voxel_size = std::stof(argv[2]);
  if (argc > 3) color_weight = std::stof(argv[3]);
  if (argc > 4) noise_std = std::stof(argv[4]);
  if (argc > 5) enable_visualization = (std::string(argv[5]) == "viz");

  std::cout << "=== ColorKISSMatcher Splat Test ===" << std::endl;
  std::cout << "Splat file: " << splat_file << std::endl;
  std::cout << "Voxel size: " << voxel_size << std::endl;
  std::cout << "Color weight: " << color_weight << std::endl;
  std::cout << "Noise std: " << noise_std << std::endl;

  // Load splat file
  std::vector<GaussianSplat> splats;
  if (!loadSplatFile(splat_file, splats)) {
    return 1;
  }

  // Extract points and colors (source)
  std::vector<Eigen::Vector3f> src_points, tgt_points, tgt_points_original;
  std::vector<Eigen::Vector4f> src_colors, tgt_colors;
  
  // Optional: filter by opacity (e.g., 0.2 to remove very transparent splats)
  float opacity_threshold = 0.0f;  // Set to 0.2 to match your LiDAR-3DGS code
  extractPointsAndColors(splats, src_points, src_colors, opacity_threshold);
  tgt_points = src_points;  // Copy for target
  tgt_points_original = src_points;  // Keep original for visualization
  tgt_colors = src_colors;

  std::cout << "\nExtracted " << src_points.size() << " points with colors." << std::endl;
  
  // Apply transformation to target
  Eigen::Matrix4f ground_truth_transform;
  applyRandomTransform(tgt_points, ground_truth_transform, max_translation, max_rotation_deg);

  // Add noise to target
  if (noise_std > 0.0f) {
    addGaussianNoise(tgt_points, noise_std);
  }

  // Run ColorKISSMatcher (geometry + color)
  std::cout << "\n=== Running ColorKISSMatcher (geometry + color) ===" << std::endl;
  
  try {
    gs3d_matcher::ColorKISSMatcherConfig config(
      voxel_size,      // From command line (0.5 in your test)
      true,            // use_voxel_sampling = true
      false,           // use_quatro = false
      1.0,             // thr_linearity = 1.0
      5000,            // num_max_corr = 5000
      color_weight     // From command line (1.0 in your test)
    );
    
    std::cout << "DEBUG: Creating ColorKISSMatcher..." << std::endl;
    gs3d_matcher::ColorKISSMatcher color_matcher(config);
    
    std::cout << "DEBUG: Calling estimate() with " << src_points.size() << " source and " 
              << tgt_points.size() << " target points..." << std::endl;
    auto color_solution = color_matcher.estimate(src_points, tgt_points, src_colors, tgt_colors);
    
    std::cout << "DEBUG: estimate() completed!" << std::endl;
    color_matcher.print();
    
    if (color_solution.valid) {
      // Build transformation matrix from rotation and translation
      Eigen::Matrix4d T_target_source = Eigen::Matrix4d::Identity();
      T_target_source.block<3,3>(0,0) = color_solution.rotation;
      T_target_source.block<3,1>(0,3) = color_solution.translation;
      
      std::cout << "\n=== Test Complete ===" << std::endl;
      std::cout << "ColorKISSMatcher successfully matched " << src_points.size() << " points" << std::endl;
      std::cout << "Inliers: " << color_matcher.getNumInliers() << std::endl;
      std::cout << "Transformation:\n" << T_target_source << std::endl;
      
      // Compute error vs ground truth
      Eigen::Matrix4d error = ground_truth_transform.cast<double>().inverse() * T_target_source;
      double translation_error = error.block<3,1>(0,3).norm();
      std::cout << "Translation error vs ground truth: " << translation_error << " units" << std::endl;
      
      // Visualization if requested
      if (enable_visualization) {
        std::cout << "\n=== Launching visualization ===" << std::endl;
        // Use identity transform for vanilla KISS (not implemented in this test)
        Eigen::Matrix4d vanilla_transform = Eigen::Matrix4d::Identity();
        visualizeResults(tgt_points_original, tgt_points, tgt_colors, tgt_colors,
                        vanilla_transform, T_target_source);
      }
    } else {
      std::cout << "ColorKISSMatcher failed to converge." << std::endl;
      return 1;
    }
  } catch (const std::exception& e) {
    std::cerr << "\nERROR: Exception caught: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "\nERROR: Unknown exception caught" << std::endl;
    return 1;
  }
  
  std::cout << "\n=== Program completed successfully ===" << std::endl;
  return 0;
}

/**
 * Example usage of ColorKISSMatcher for point cloud matching with color and opacity
 * 
 * This example demonstrates how to use the ColorKISSMatcher class to match
 * two point clouds that have color and opacity attributes.
 */

#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <3dgs_matcher/ColorKISSMatcher.hpp>

int main() {
  // Example 1: Create sample colored point clouds
  std::cout << "=== ColorKISSMatcher Example ===" << std::endl;
  
  // Create source point cloud (e.g., a small cube)
  std::vector<Eigen::Vector3f> src_points = {
      {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 0.0f},
      {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}
  };

  // Create colors for source (RGBA format, normalized to [0, 1])
  // Red points with varying opacity
  std::vector<Eigen::Vector4f> src_colors = {
      {1.0f, 0.0f, 0.0f, 1.0f},  // Red, opaque
      {1.0f, 0.0f, 0.0f, 1.0f},
      {1.0f, 0.0f, 0.0f, 0.8f},  // Red, slightly transparent
      {1.0f, 0.0f, 0.0f, 0.8f},
      {1.0f, 0.0f, 0.0f, 0.6f},  // Red, more transparent
      {1.0f, 0.0f, 0.0f, 0.6f},
      {1.0f, 0.0f, 0.0f, 0.4f},
      {1.0f, 0.0f, 0.0f, 0.4f}
  };

  // Create target point cloud (same shape, translated and rotated)
  // Apply a simple transformation: translate by (5, 5, 5)
  std::vector<Eigen::Vector3f> tgt_points;
  for (const auto& pt : src_points) {
    tgt_points.push_back(pt + Eigen::Vector3f(5.0f, 5.0f, 5.0f));
  }

  // Target colors (same as source for this example)
  std::vector<Eigen::Vector4f> tgt_colors = src_colors;

  std::cout << "Source points: " << src_points.size() << std::endl;
  std::cout << "Target points: " << tgt_points.size() << std::endl;

  // Example 2: Configure and create ColorKISSMatcher
  float voxel_size = 0.3f;
  float color_weight = 1.0f;  // Equal weight for color features
  
  // Option A: Simple constructor
  gs3d_matcher::ColorKISSMatcher matcher_simple(voxel_size, color_weight);

  // Option B: Advanced configuration
  gs3d_matcher::ColorKISSMatcherConfig config(
      voxel_size,         // voxel_size
      true,               // use_voxel_sampling
      false,              // use_quatro
      1.0,                // thr_linearity
      5000,               // num_max_corr
      color_weight        // color_weight (NEW!)
  );
  
  gs3d_matcher::ColorKISSMatcher matcher(config);

  // Example 3: Match the point clouds
  std::cout << "\nMatching point clouds with color information..." << std::endl;
  
  try {
    auto [src_matched, tgt_matched] = matcher.match(
        src_points, tgt_points,
        src_colors, tgt_colors
    );

    std::cout << "Matched keypoints: " << src_matched.size() << std::endl;

    // Example 4: Estimate full transformation
    std::cout << "\nEstimating transformation..." << std::endl;
    auto solution = matcher.estimate(
        src_points, tgt_points,
        src_colors, tgt_colors
    );

    std::cout << "Rotation matrix:" << std::endl;
    std::cout << solution.rotation << std::endl;
    std::cout << "Translation vector: " << solution.translation.transpose() << std::endl;

    // Example 5: Access available statistics
    std::cout << "\n=== Statistics ===" << std::endl;
    std::cout << "Number of inliers: " << matcher.getNumInliers() << std::endl;

    // Print full configuration
    std::cout << "\n";
    matcher.print();

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  // Example 7: Working with RGB values in [0, 255] range
  std::cout << "\n=== Color Normalization Example ===" << std::endl;
  std::cout << "ColorKISSMatcher automatically normalizes colors from [0, 255] to [0, 1]" << std::endl;
  
  std::vector<Eigen::Vector4f> colors_255 = {
      {255.0f, 0.0f, 0.0f, 255.0f},     // Red in [0, 255] range
      {0.0f, 255.0f, 0.0f, 255.0f},     // Green
      {0.0f, 0.0f, 255.0f, 255.0f},     // Blue
      {255.0f, 255.0f, 0.0f, 200.0f}    // Yellow with alpha=200
  };
  std::cout << "Colors in [0, 255] range are supported!" << std::endl;

  // Example 8: Adjusting color weight
  std::cout << "\n=== Color Weight Adjustment ===" << std::endl;
  std::cout << "color_weight = 0.0: Only geometric features (same as standard KISSMatcher)" << std::endl;
  std::cout << "color_weight = 1.0: Equal weight for color and geometric features" << std::endl;
  std::cout << "color_weight = 2.0: Emphasize color features more" << std::endl;

  return 0;
}

/* 
 * Expected output format:
 * 
 * === ColorKISSMatcher Example ===
 * Source points: 8
 * Target points: 8
 * 
 * Matching point clouds with color information...
 * ColorFasterPFH: Extended 8 descriptors from 33 to 37 dimensions (color_weight=1.0)
 * Matched keypoints: 8
 * 
 * Estimating transformation...
 * Transformation matrix:
 * [4x4 matrix]
 * 
 * === Statistics ===
 * Processing time: 0.001 s
 * Extraction time: 0.002 s
 * ...
 */

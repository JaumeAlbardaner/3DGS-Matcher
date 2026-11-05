// SPDX-FileCopyrightText: Copyright 2024 3DGS-Matcher authors
// SPDX-License-Identifier: MIT
#pragma once

#include <memory>
#include <vector>
#include <Eigen/Core>
#include <small_gicp/registration/registration.hpp>
#include <small_gicp/registration/registration_helper.hpp>
#include <small_gicp/registration/reduction_tbb.hpp>
#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/ann/kdtree_tbb.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/util/downsampling_omp.hpp>
#include <small_gicp/util/downsampling_tbb.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>
#include <small_gicp/util/normal_estimation_tbb.hpp>
#include <3dgs_matcher/small_gicp_color/point_cloud_color.hpp>
#include <3dgs_matcher/small_gicp_color/color_icp_factor.hpp>
#include <3dgs_matcher/small_gicp_color/color_gicp_factor.hpp>

namespace small_gicp {

/**
 * @brief Preprocess colored point cloud: downsample, estimate normals and covariances
 * 
 * @param points              Input points
 * @param colors              Input colors (RGBA)
 * @param downsampling_resolution  Voxel size for downsampling
 * @param num_neighbors       Number of neighbors for normal/covariance estimation
 * @param num_threads         Number of threads (0 = auto)
 * @return Preprocessed colored point cloud
 */
inline PointCloudWithColor::Ptr preprocess_colored_points(
  const std::vector<Eigen::Vector3f>& points,
  const std::vector<Eigen::Vector4f>& colors,
  double downsampling_resolution = 0.25,
  int num_neighbors = 10,
  int num_threads = 0) {
  
  if (points.size() != colors.size()) {
    throw std::runtime_error("preprocess_colored_points: points and colors size mismatch");
  }

  // Convert to PointCloudWithColor
  auto cloud = std::make_shared<PointCloudWithColor>();
  cloud->resize(points.size());
  
  for (size_t i = 0; i < points.size(); i++) {
    cloud->point(i) << points[i].cast<double>(), 1.0;
    cloud->color(i) << colors[i].cast<double>();
  }

  // Downsample (preserving colors). TBB version takes no num_threads arg.
  if (downsampling_resolution > 0.0) {
#ifdef _OPENMP
    cloud = voxelgrid_sampling_omp<PointCloudWithColor>(*cloud, downsampling_resolution, num_threads);
#else
    cloud = voxelgrid_sampling_tbb<PointCloudWithColor>(*cloud, downsampling_resolution);
#endif
  }

  // Estimate normals and covariances. TBB version takes no num_threads arg.
#ifdef _OPENMP
  estimate_normals_covariances_omp(*cloud, num_neighbors, num_threads);
#else
  estimate_normals_covariances_tbb(*cloud, num_neighbors);
#endif

  return cloud;
}

/**
 * @brief Easy-to-use color ICP registration
 * 
 * @param target              Target point cloud with colors
 * @param source              Source point cloud with colors
 * @param target_tree         KdTree for target (optional, will be created if nullptr)
 * @param init_T              Initial transformation guess
 * @param color_weight        Weight for color term
 * @param registration_type   "ICP", "PLANE_ICP", "GICP", or "VGICP"
 * @return Registration result
 */
template <typename TargetPointCloud, typename SourcePointCloud>
RegistrationResult align_colored(
  const TargetPointCloud& target,
  const SourcePointCloud& source,
  const KdTree<TargetPointCloud>* target_tree = nullptr,
  const Eigen::Isometry3d& init_T = Eigen::Isometry3d::Identity(),
  double color_weight = 1.0,
  const std::string& registration_type = "GICP") {
  
  // Build KdTree if not provided. KdTree takes shared_ptr<const PointCloud>.
  std::shared_ptr<KdTree<TargetPointCloud>> tree_ptr;
  if (target_tree == nullptr) {
    auto target_ptr = std::make_shared<const TargetPointCloud>(target);
#ifdef _OPENMP
    tree_ptr = std::make_shared<KdTree<TargetPointCloud>>(target_ptr, KdTreeBuilderOMP());
#else
    tree_ptr = std::make_shared<KdTree<TargetPointCloud>>(target_ptr, KdTreeBuilderTBB());
#endif
    target_tree = tree_ptr.get();
  }

  // Select registration factor based on type.
  // Registration::align() no longer takes factor settings; set via registration.point_factor.
  if (registration_type == "ICP") {
    Registration<ColorICPFactor, ParallelReductionTBB> registration;
    registration.point_factor = ColorICPFactor::Setting(color_weight);
    return registration.align(target, source, *target_tree, init_T);

  } else if (registration_type == "PLANE_ICP") {
    Registration<ColorPointToPlaneICPFactor, ParallelReductionTBB> registration;
    registration.point_factor = ColorPointToPlaneICPFactor::Setting(color_weight);
    return registration.align(target, source, *target_tree, init_T);

  } else if (registration_type == "GICP") {
    Registration<ColorGICPFactor, ParallelReductionTBB> registration;
    registration.point_factor = ColorGICPFactor::Setting(color_weight);
    return registration.align(target, source, *target_tree, init_T);

  } else if (registration_type == "VGICP") {
    Registration<ColorVGICPFactor, ParallelReductionTBB> registration;
    registration.point_factor = ColorVGICPFactor::Setting(color_weight, 1.0);
    return registration.align(target, source, *target_tree, init_T);

  } else if (registration_type == "GICP_DEW") {
    // ColorPFH-weighted GICP: geometric correspondences, color-agreement weighting
    // of the information matrix. `color_weight` is reinterpreted as the RGB sigma.
    Registration<ColorWeightedGICPFactor, ParallelReductionTBB> registration;
    registration.point_factor = ColorWeightedGICPFactor::Setting(color_weight);
    return registration.align(target, source, *target_tree, init_T);

  } else {
    throw std::runtime_error("Unknown registration type: " + registration_type);
  }
}

/**
 * @brief Complete color-aware registration pipeline from raw points
 * 
 * This is the simplest interface: just provide points and colors.
 * 
 * @param target_points       Target points
 * @param target_colors       Target colors (RGBA)
 * @param source_points       Source points
 * @param source_colors       Source colors (RGBA)
 * @param downsampling_resolution  Voxel size for downsampling
 * @param color_weight        Weight for color term
 * @param registration_type   "ICP", "PLANE_ICP", "GICP", or "VGICP"
 * @return Registration result
 */
inline RegistrationResult align_colored_simple(
  const std::vector<Eigen::Vector3f>& target_points,
  const std::vector<Eigen::Vector4f>& target_colors,
  const std::vector<Eigen::Vector3f>& source_points,
  const std::vector<Eigen::Vector4f>& source_colors,
  double downsampling_resolution = 0.25,
  double color_weight = 1.0,
  const std::string& registration_type = "GICP",
  const Eigen::Isometry3d& init_T = Eigen::Isometry3d::Identity()) {
  
  // Auto-calibrate the ColorPFH-weighted GICP sigma to the actual color scale.
  // 3DGS colors are SH-derived and not confined to [0,1], so a fixed sigma can
  // crush all weights to ~0 (optimizer then quits at iteration 0). Passing
  // color_weight <= 0 with GICP_DEW requests sigma = RGB standard deviation.
  double effective_color_weight = color_weight;
  if (registration_type == "GICP_DEW" && color_weight <= 0.0) {
    Eigen::Vector3d sum = Eigen::Vector3d::Zero(), sum2 = Eigen::Vector3d::Zero();
    const size_t n = target_colors.size();
    for (const auto& c : target_colors) {
      const Eigen::Vector3d cd = c.head<3>().cast<double>();
      sum += cd; sum2 += cd.cwiseProduct(cd);
    }
    double var = 0.0;
    if (n > 0) {
      const Eigen::Vector3d mean = sum / static_cast<double>(n);
      const Eigen::Vector3d msq  = sum2 / static_cast<double>(n);
      var = (msq - mean.cwiseProduct(mean)).cwiseMax(0.0).mean();
    }
    effective_color_weight = std::max(0.1, std::sqrt(var));  // sigma in RGB units
  }

  // Preprocess both clouds
  auto target = preprocess_colored_points(target_points, target_colors, downsampling_resolution);
  auto source = preprocess_colored_points(source_points, source_colors, downsampling_resolution);

  // Build KdTree for target. KdTree now takes shared_ptr<const PointCloud>.
#ifdef _OPENMP
  auto target_tree = std::make_shared<KdTree<PointCloudWithColor>>(
      std::const_pointer_cast<const PointCloudWithColor>(target), KdTreeBuilderOMP());
#else
  auto target_tree = std::make_shared<KdTree<PointCloudWithColor>>(
      std::const_pointer_cast<const PointCloudWithColor>(target), KdTreeBuilderTBB());
#endif

  // Align
  return align_colored(*target, *source, target_tree.get(), init_T, effective_color_weight, registration_type);
}

/**
 * @brief Convert std::vector of Eigen vectors to PointCloudWithColor
 * 
 * Helper function for easy conversion from standard vectors.
 */
inline PointCloudWithColor::Ptr make_colored_point_cloud(
  const std::vector<Eigen::Vector3f>& points,
  const std::vector<Eigen::Vector4f>& colors) {
  
  if (points.size() != colors.size()) {
    throw std::runtime_error("make_colored_point_cloud: points and colors size mismatch");
  }

  auto cloud = std::make_shared<PointCloudWithColor>();
  cloud->resize(points.size());
  
  for (size_t i = 0; i < points.size(); i++) {
    cloud->point(i) << points[i].cast<double>(), 1.0;
    cloud->color(i) << colors[i].cast<double>();
  }

  return cloud;
}

}  // namespace small_gicp

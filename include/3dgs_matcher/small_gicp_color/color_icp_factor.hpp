// SPDX-FileCopyrightText: Copyright 2024 3DGS-Matcher authors
// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <small_gicp/util/lie.hpp>
#include <small_gicp/ann/traits.hpp>
#include <small_gicp/points/traits.hpp>
#include <3dgs_matcher/small_gicp_color/point_cloud_color.hpp>

namespace small_gicp {

/**
 * @brief Color-aware Point-to-Point ICP factor
 * 
 * Extends the standard ICP factor with color term:
 * E = w_geom * ||p_t - T*p_s||^2 + w_color * ||c_t - c_s||^2
 * 
 * where:
 * - p_t, p_s: target and source points
 * - c_t, c_s: target and source colors (RGBA)
 * - w_geom, w_color: geometric and color weights
 */
struct ColorICPFactor {
  struct Setting {
    double color_weight = 1.0;  ///< Weight for color term (default: equal to geometry)
    
    Setting(double color_weight = 1.0) : color_weight(color_weight) {}
  };

  ColorICPFactor(const Setting& setting = Setting()) 
    : target_index(std::numeric_limits<size_t>::max()), 
      source_index(std::numeric_limits<size_t>::max()),
      color_weight(setting.color_weight) {}

  template <typename TargetPointCloud, typename SourcePointCloud, typename TargetTree, typename CorrespondenceRejector>
  bool linearize(
    const TargetPointCloud& target,
    const SourcePointCloud& source,
    const TargetTree& target_tree,
    const Eigen::Isometry3d& T,
    size_t source_index,
    const CorrespondenceRejector& rejector,
    Eigen::Matrix<double, 6, 6>* H,
    Eigen::Matrix<double, 6, 1>* b,
    double* e) {
    //
    this->source_index = source_index;
    this->target_index = std::numeric_limits<size_t>::max();

    const Eigen::Vector4d transed_source_pt = T * traits::point(source, source_index);

    size_t k_index;
    double k_sq_dist;
    if (!traits::nearest_neighbor_search(target_tree, transed_source_pt, &k_index, &k_sq_dist) || 
        rejector(target, source, T, k_index, source_index, k_sq_dist)) {
      return false;
    }

    target_index = k_index;
    
    // Geometric residual
    const Eigen::Vector4d geom_residual = traits::point(target, target_index) - transed_source_pt;

    // Geometric Jacobian
    Eigen::Matrix<double, 4, 6> J = Eigen::Matrix<double, 4, 6>::Zero();
    J.block<3, 3>(0, 0) = T.linear() * skew(traits::point(source, source_index).template head<3>());
    J.block<3, 3>(0, 3) = -T.linear();

    // Accumulate geometric term
    *H = J.transpose() * J;
    *b = J.transpose() * geom_residual;
    *e = 0.5 * geom_residual.squaredNorm();

    // Add color term if both clouds have color
    if constexpr (std::is_same_v<TargetPointCloud, PointCloudWithColor> && 
                  std::is_same_v<SourcePointCloud, PointCloudWithColor>) {
      if (traits::has_colors(target) && traits::has_colors(source) && color_weight > 0.0) {
        const Eigen::Vector4d color_residual = 
          traits::color(target, target_index) - traits::color(source, source_index);
        
        // Color doesn't depend on transformation, so no Jacobian w.r.t. pose
        // Just add to error term
        *e += 0.5 * color_weight * color_residual.squaredNorm();
        
        // Note: For optimization, we could add color gradient if we model color change
        // under transformation, but typically colors are transformation-invariant
      }
    }

    return true;
  }

  template <typename TargetPointCloud, typename SourcePointCloud>
  double error(const TargetPointCloud& target, const SourcePointCloud& source, const Eigen::Isometry3d& T) const {
    if (target_index == std::numeric_limits<size_t>::max()) {
      return 0.0;
    }

    // Geometric error
    const Eigen::Vector4d geom_residual = 
      traits::point(target, target_index) - T * traits::point(source, source_index);
    double error_total = 0.5 * geom_residual.squaredNorm();

    // Color error
    if constexpr (std::is_same_v<TargetPointCloud, PointCloudWithColor> && 
                  std::is_same_v<SourcePointCloud, PointCloudWithColor>) {
      if (traits::has_colors(target) && traits::has_colors(source) && color_weight > 0.0) {
        const Eigen::Vector4d color_residual = 
          traits::color(target, target_index) - traits::color(source, source_index);
        error_total += 0.5 * color_weight * color_residual.squaredNorm();
      }
    }

    return error_total;
  }

  bool inlier() const { return target_index != std::numeric_limits<size_t>::max(); }

  size_t target_index;
  size_t source_index;
  double color_weight;
};

/**
 * @brief Color-aware Point-to-Plane ICP factor
 * 
 * Extends point-to-plane ICP with color:
 * E = w_geom * ||(p_t - T*p_s) · n_t||^2 + w_color * ||c_t - c_s||^2
 */
struct ColorPointToPlaneICPFactor {
  struct Setting {
    double color_weight = 1.0;
    
    Setting(double color_weight = 1.0) : color_weight(color_weight) {}
  };

  ColorPointToPlaneICPFactor(const Setting& setting = Setting())
    : target_index(std::numeric_limits<size_t>::max()),
      source_index(std::numeric_limits<size_t>::max()),
      color_weight(setting.color_weight) {}

  template <typename TargetPointCloud, typename SourcePointCloud, typename TargetTree, typename CorrespondenceRejector>
  bool linearize(
    const TargetPointCloud& target,
    const SourcePointCloud& source,
    const TargetTree& target_tree,
    const Eigen::Isometry3d& T,
    size_t source_index,
    const CorrespondenceRejector& rejector,
    Eigen::Matrix<double, 6, 6>* H,
    Eigen::Matrix<double, 6, 1>* b,
    double* e) {
    //
    this->source_index = source_index;
    this->target_index = std::numeric_limits<size_t>::max();

    const Eigen::Vector4d transed_source_pt = T * traits::point(source, source_index);

    size_t k_index;
    double k_sq_dist;
    if (!traits::nearest_neighbor_search(target_tree, transed_source_pt, &k_index, &k_sq_dist) || 
        rejector(target, source, T, k_index, source_index, k_sq_dist)) {
      return false;
    }

    target_index = k_index;
    const Eigen::Vector4d target_pt = traits::point(target, target_index);
    const Eigen::Vector4d target_normal = traits::normal(target, target_index);

    // Point-to-plane residual
    const Eigen::Vector4d diff = target_pt - transed_source_pt;
    const double residual = diff.dot(target_normal);

    // Jacobian for point-to-plane
    Eigen::Matrix<double, 1, 6> J;
    J.template block<1, 3>(0, 0) = 
      target_normal.template head<3>().transpose() * 
      T.linear() * skew(traits::point(source, source_index).template head<3>());
    J.template block<1, 3>(0, 3) = -target_normal.template head<3>().transpose() * T.linear();

    *H = J.transpose() * J;
    *b = J.transpose() * residual;
    *e = 0.5 * residual * residual;

    // Add color term
    if constexpr (std::is_same_v<TargetPointCloud, PointCloudWithColor> && 
                  std::is_same_v<SourcePointCloud, PointCloudWithColor>) {
      if (traits::has_colors(target) && traits::has_colors(source) && color_weight > 0.0) {
        const Eigen::Vector4d color_residual = 
          traits::color(target, target_index) - traits::color(source, source_index);
        *e += 0.5 * color_weight * color_residual.squaredNorm();
      }
    }

    return true;
  }

  template <typename TargetPointCloud, typename SourcePointCloud>
  double error(const TargetPointCloud& target, const SourcePointCloud& source, const Eigen::Isometry3d& T) const {
    if (target_index == std::numeric_limits<size_t>::max()) {
      return 0.0;
    }

    // Point-to-plane error
    const Eigen::Vector4d diff = 
      traits::point(target, target_index) - T * traits::point(source, source_index);
    const Eigen::Vector4d target_normal = traits::normal(target, target_index);
    const double residual = diff.dot(target_normal);
    double error_total = 0.5 * residual * residual;

    // Color error
    if constexpr (std::is_same_v<TargetPointCloud, PointCloudWithColor> && 
                  std::is_same_v<SourcePointCloud, PointCloudWithColor>) {
      if (traits::has_colors(target) && traits::has_colors(source) && color_weight > 0.0) {
        const Eigen::Vector4d color_residual = 
          traits::color(target, target_index) - traits::color(source, source_index);
        error_total += 0.5 * color_weight * color_residual.squaredNorm();
      }
    }

    return error_total;
  }

  bool inlier() const { return target_index != std::numeric_limits<size_t>::max(); }

  size_t target_index;
  size_t source_index;
  double color_weight;
};

}  // namespace small_gicp

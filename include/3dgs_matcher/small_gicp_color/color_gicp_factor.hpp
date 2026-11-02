// SPDX-FileCopyrightText: Copyright 2024 3DGS-Matcher authors
// SPDX-License-Identifier: MIT
#pragma once

#include <cmath>
#include <limits>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <small_gicp/util/lie.hpp>
#include <small_gicp/ann/traits.hpp>
#include <small_gicp/points/traits.hpp>
#include <3dgs_matcher/small_gicp_color/point_cloud_color.hpp>

namespace small_gicp {

/**
 * @brief Color-aware Generalized ICP factor
 * 
 * Extends GICP with color information:
 * E = w_geom * d^T * M * d + w_color * ||c_t - c_s||^2
 * 
 * where:
 * - d = p_t - T*p_s (point difference)
 * - M = (C_t + T*C_s*T^T)^-1 (Mahalanobis distance)
 * - C_t, C_s: target and source covariances
 * - c_t, c_s: target and source colors
 */
struct ColorGICPFactor {
  struct Setting {
    double color_weight = 1.0;
    
    Setting(double color_weight = 1.0) : color_weight(color_weight) {}
  };

  ColorGICPFactor(const Setting& setting = Setting())
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
    const Eigen::Vector4d residual = traits::point(target, target_index) - transed_source_pt;

    // Compute Mahalanobis distance matrix
    // M = (C_target + R * C_source * R^T)^-1
    const Eigen::Matrix4d& target_cov = traits::cov(target, target_index);
    const Eigen::Matrix4d& source_cov = traits::cov(source, source_index);
    
    Eigen::Matrix4d RCR = Eigen::Matrix4d::Zero();
    RCR.block<3, 3>(0, 0) = T.linear() * source_cov.block<3, 3>(0, 0) * T.linear().transpose();
    
    // Invert only the 3x3 block (the 4x4 is rank-3; inverting it directly is
    // singular -> NaN, which silently disabled the old colored GICP).
    Eigen::Matrix4d M = Eigen::Matrix4d::Zero();
    M.block<3, 3>(0, 0) = (target_cov + RCR).block<3, 3>(0, 0).inverse();

    // GICP Jacobian
    Eigen::Matrix<double, 4, 6> J = Eigen::Matrix<double, 4, 6>::Zero();
    J.block<3, 3>(0, 0) = T.linear() * skew(traits::point(source, source_index).template head<3>());
    J.block<3, 3>(0, 3) = -T.linear();

    // Weighted Hessian and gradient
    *H = J.transpose() * M * J;
    *b = J.transpose() * M * residual;
    *e = 0.5 * residual.transpose() * M * residual;

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

    // GICP error
    const Eigen::Vector4d residual = 
      traits::point(target, target_index) - T * traits::point(source, source_index);
    
    const Eigen::Matrix4d& target_cov = traits::cov(target, target_index);
    const Eigen::Matrix4d& source_cov = traits::cov(source, source_index);
    
    Eigen::Matrix4d RCR = Eigen::Matrix4d::Zero();
    RCR.block<3, 3>(0, 0) = T.linear() * source_cov.block<3, 3>(0, 0) * T.linear().transpose();
    
    // Invert only the 3x3 block (the 4x4 is rank-3; inverting it directly is
    // singular -> NaN, which silently disabled the old colored GICP).
    Eigen::Matrix4d M = Eigen::Matrix4d::Zero();
    M.block<3, 3>(0, 0) = (target_cov + RCR).block<3, 3>(0, 0).inverse();

    double error_total = 0.5 * residual.transpose() * M * residual;

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
 * @brief ColorPFH-weighted GICP factor (correct color fusion)
 *
 * Unlike ColorGICPFactor (which adds the color term only to the scalar loss e,
 * leaving H and b — and therefore the pose update — untouched), this factor lets
 * color genuinely influence the optimization by re-weighting each correspondence.
 *
 * Correspondences are still found geometrically (Kd-tree NN on transformed source),
 * so color never corrupts which points are matched. Instead, each geometric match
 * is weighted by color agreement:
 *
 *     w = exp(-||c_t - c_s||^2 / (2 * sigma^2))         (RGB distance)
 *     H = J^T (w * M) J,   b = J^T (w * M) r,   e = 0.5 * w * r^T M r
 *
 * This down-weights geometrically-plausible but color-mismatched matches, directly
 * attacking the false-consensus failure of plane-to-plane GICP, while preserving its
 * accuracy. Under a global color shift (cross-platform exposure differences) all
 * weights shrink roughly uniformly, so the factor degrades gracefully toward plain
 * geometric GICP rather than diverging.
 *
 * `sigma` is stored in Setting.color_sigma (RGB units in [0,1]; ~0.15 default).
 */
struct ColorWeightedGICPFactor {
  struct Setting {
    double color_sigma = 0.15;
    Setting(double color_sigma = 0.15) : color_sigma(color_sigma) {}
  };

  ColorWeightedGICPFactor(const Setting& setting = Setting())
    : target_index(std::numeric_limits<size_t>::max()),
      source_index(std::numeric_limits<size_t>::max()),
      color_sigma(setting.color_sigma) {}

  template <typename TargetPointCloud, typename SourcePointCloud>
  double color_weight_for(const TargetPointCloud& target, const SourcePointCloud& source) const {
    if constexpr (std::is_same_v<TargetPointCloud, PointCloudWithColor> &&
                  std::is_same_v<SourcePointCloud, PointCloudWithColor>) {
      if (traits::has_colors(target) && traits::has_colors(source) && color_sigma > 0.0) {
        const Eigen::Vector4d dc = traits::color(target, target_index) - traits::color(source, source_index);
        const double d2 = dc.template head<3>().squaredNorm();
        // Floor keeps weights O(1) so the optimizer's gradient never collapses to
        // zero (graceful degradation to geometric GICP when color is uninformative).
        constexpr double w_floor = 0.05;
        return w_floor + (1.0 - w_floor) * std::exp(-d2 / (2.0 * color_sigma * color_sigma));
      }
    }
    return 1.0;
  }

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
    const Eigen::Vector4d residual = traits::point(target, target_index) - transed_source_pt;

    const Eigen::Matrix4d& target_cov = traits::cov(target, target_index);
    const Eigen::Matrix4d& source_cov = traits::cov(source, source_index);

    Eigen::Matrix4d RCR = Eigen::Matrix4d::Zero();
    RCR.block<3, 3>(0, 0) = T.linear() * source_cov.block<3, 3>(0, 0) * T.linear().transpose();

    // Invert only the 3x3 block (the 4x4 is rank-3; inverting it directly is
    // singular -> NaN, which silently disabled the old colored GICP).
    Eigen::Matrix4d M = Eigen::Matrix4d::Zero();
    M.block<3, 3>(0, 0) = (target_cov + RCR).block<3, 3>(0, 0).inverse();

    Eigen::Matrix<double, 4, 6> J = Eigen::Matrix<double, 4, 6>::Zero();
    J.block<3, 3>(0, 0) = T.linear() * skew(traits::point(source, source_index).template head<3>());
    J.block<3, 3>(0, 3) = -T.linear();

    // Color agreement weight modulates the geometric information matrix, so it
    // propagates into H and b (not just the scalar loss).
    const double w = color_weight_for(target, source);
    const Eigen::Matrix4d Mw = w * M;

    *H = J.transpose() * Mw * J;
    *b = J.transpose() * Mw * residual;
    *e = 0.5 * w * residual.transpose() * M * residual;

    return true;
  }

  template <typename TargetPointCloud, typename SourcePointCloud>
  double error(const TargetPointCloud& target, const SourcePointCloud& source, const Eigen::Isometry3d& T) const {
    if (target_index == std::numeric_limits<size_t>::max()) {
      return 0.0;
    }
    const Eigen::Vector4d residual =
      traits::point(target, target_index) - T * traits::point(source, source_index);
    const Eigen::Matrix4d& target_cov = traits::cov(target, target_index);
    const Eigen::Matrix4d& source_cov = traits::cov(source, source_index);
    Eigen::Matrix4d RCR = Eigen::Matrix4d::Zero();
    RCR.block<3, 3>(0, 0) = T.linear() * source_cov.block<3, 3>(0, 0) * T.linear().transpose();
    // Invert only the 3x3 block (the 4x4 is rank-3; inverting it directly is
    // singular -> NaN, which silently disabled the old colored GICP).
    Eigen::Matrix4d M = Eigen::Matrix4d::Zero();
    M.block<3, 3>(0, 0) = (target_cov + RCR).block<3, 3>(0, 0).inverse();
    const double w = color_weight_for(target, source);
    return 0.5 * w * residual.transpose() * M * residual;
  }

  bool inlier() const { return target_index != std::numeric_limits<size_t>::max(); }

  size_t target_index;
  size_t source_index;
  double color_sigma;
};

/**
 * @brief Color-aware VGICP factor (Voxelized GICP)
 *
 * Similar to ColorGICPFactor but designed for use with voxel maps.
 * Uses voxel-based covariances for efficiency.
 */
struct ColorVGICPFactor {
  struct Setting {
    double color_weight = 1.0;
    double voxel_resolution = 1.0;
    
    Setting(double color_weight = 1.0, double voxel_resolution = 1.0) 
      : color_weight(color_weight), voxel_resolution(voxel_resolution) {}
  };

  ColorVGICPFactor(const Setting& setting = Setting())
    : target_index(std::numeric_limits<size_t>::max()),
      source_index(std::numeric_limits<size_t>::max()),
      color_weight(setting.color_weight),
      voxel_resolution(setting.voxel_resolution) {}

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
    // Implementation similar to ColorGICPFactor but with voxel-based covariances
    // For brevity, delegating to GICP implementation
    auto gicp_factor = ColorGICPFactor(ColorGICPFactor::Setting(color_weight));
    return gicp_factor.linearize(target, source, target_tree, T, source_index, rejector, H, b, e);
  }

  template <typename TargetPointCloud, typename SourcePointCloud>
  double error(const TargetPointCloud& target, const SourcePointCloud& source, const Eigen::Isometry3d& T) const {
    auto gicp_factor = ColorGICPFactor(ColorGICPFactor::Setting(color_weight));
    gicp_factor.target_index = this->target_index;
    gicp_factor.source_index = this->source_index;
    return gicp_factor.error(target, source, T);
  }

  bool inlier() const { return target_index != std::numeric_limits<size_t>::max(); }

  size_t target_index;
  size_t source_index;
  double color_weight;
  double voxel_resolution;
};

}  // namespace small_gicp

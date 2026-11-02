// SPDX-FileCopyrightText: Copyright 2024 3DGS-Matcher authors
// SPDX-License-Identifier: MIT
#pragma once

#include <memory>
#include <vector>
#include <Eigen/Core>
#include <small_gicp/points/traits.hpp>

namespace small_gicp {

/**
 * @brief Point cloud with color and opacity attributes
 * 
 * Extends small_gicp::PointCloud with RGBA information for color-aware registration.
 * Compatible with the small_gicp trait system.
 */
struct PointCloudWithColor {
public:
  using Ptr = std::shared_ptr<PointCloudWithColor>;
  using ConstPtr = std::shared_ptr<const PointCloudWithColor>;

  /// @brief Constructor
  PointCloudWithColor() {}

  /// @brief Constructor from points and colors
  /// @param points  Points to initialize the point cloud
  /// @param colors  Colors (RGBA) for each point
  template <typename T, int D, typename Allocator>
  explicit PointCloudWithColor(
    const std::vector<Eigen::Matrix<T, D, 1>, Allocator>& points,
    const std::vector<Eigen::Matrix<T, 4, 1>, Allocator>& colors) {
    
    if (points.size() != colors.size()) {
      throw std::runtime_error("PointCloudWithColor: points and colors size mismatch");
    }
    
    this->resize(points.size());
    for (size_t i = 0; i < points.size(); i++) {
      this->point(i) << points[i].template cast<double>().template head<3>(), 1.0;
      this->color(i) << colors[i].template cast<double>();
    }
  }

  /// @brief Destructor
  ~PointCloudWithColor() {}

  /// @brief Number of points
  size_t size() const { return points.size(); }

  /// @brief Check if the point cloud is empty
  bool empty() const { return points.empty(); }

  /// @brief Resize all buffers
  /// @param n  Number of points
  void resize(size_t n) {
    points.resize(n);
    normals.resize(n);
    covs.resize(n);
    colors.resize(n);
  }

  /// @brief Get i-th point
  Eigen::Vector4d& point(size_t i) { return points[i]; }

  /// @brief Get i-th normal
  Eigen::Vector4d& normal(size_t i) { return normals[i]; }

  /// @brief Get i-th covariance
  Eigen::Matrix4d& cov(size_t i) { return covs[i]; }

  /// @brief Get i-th color (RGBA)
  Eigen::Vector4d& color(size_t i) { return colors[i]; }

  /// @brief Get i-th point (const)
  const Eigen::Vector4d& point(size_t i) const { return points[i]; }

  /// @brief Get i-th normal (const)
  const Eigen::Vector4d& normal(size_t i) const { return normals[i]; }

  /// @brief Get i-th covariance (const)
  const Eigen::Matrix4d& cov(size_t i) const { return covs[i]; }

  /// @brief Get i-th color (const)
  const Eigen::Vector4d& color(size_t i) const { return colors[i]; }

public:
  std::vector<Eigen::Vector4d> points;   ///< Point coordinates (x, y, z, 1)
  std::vector<Eigen::Vector4d> normals;  ///< Point normals (nx, ny, nz, 0)
  std::vector<Eigen::Matrix4d> covs;     ///< Point covariances (3x3 matrix) + zero padding
  std::vector<Eigen::Vector4d> colors;   ///< Point colors and opacity (r, g, b, a) in [0, 1]
};

namespace traits {

/**
 * @brief Traits specialization for PointCloudWithColor
 * 
 * Provides the interface for small_gicp's trait-based system while adding color support.
 */
template <>
struct Traits<PointCloudWithColor> {
  using Points = PointCloudWithColor;

  static size_t size(const Points& points) { return points.size(); }

  static bool has_points(const Points& points) { return !points.points.empty(); }
  static bool has_normals(const Points& points) { return !points.normals.empty(); }
  static bool has_covs(const Points& points) { return !points.covs.empty(); }
  static bool has_colors(const Points& points) { return !points.colors.empty(); }

  static const Eigen::Vector4d& point(const Points& points, size_t i) { return points.point(i); }
  static const Eigen::Vector4d& normal(const Points& points, size_t i) { return points.normal(i); }
  static const Eigen::Matrix4d& cov(const Points& points, size_t i) { return points.cov(i); }
  static const Eigen::Vector4d& color(const Points& points, size_t i) { return points.color(i); }

  static void resize(Points& points, size_t n) { points.resize(n); }
  static void set_point(Points& points, size_t i, const Eigen::Vector4d& pt) { points.point(i) = pt; }
  static void set_normal(Points& points, size_t i, const Eigen::Vector4d& n) { points.normal(i) = n; }
  static void set_cov(Points& points, size_t i, const Eigen::Matrix4d& cov) { points.cov(i) = cov; }
  static void set_color(Points& points, size_t i, const Eigen::Vector4d& color) { points.color(i) = color; }
};

// Helper functions for color access
template <typename T>
bool has_colors(const T& points) {
  if constexpr (std::is_same_v<T, PointCloudWithColor>) {
    return Traits<T>::has_colors(points);
  }
  return false;
}

template <typename T>
auto color(const T& points, size_t i) -> decltype(Traits<T>::color(points, i)) {
  static_assert(std::is_same_v<T, PointCloudWithColor>, "Point cloud type must have color attribute");
  return Traits<T>::color(points, i);
}

template <typename T>
void set_color(T& points, size_t i, const Eigen::Vector4d& color) {
  static_assert(std::is_same_v<T, PointCloudWithColor>, "Point cloud type must have color attribute");
  Traits<T>::set_color(points, i, color);
}

}  // namespace traits

}  // namespace small_gicp

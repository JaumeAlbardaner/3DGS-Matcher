/**
 * Copyright 2024, Massachusetts Institute of Technology
 * All Rights Reserved
 */
#pragma once

#include <Eigen/Core>
#include <cmath>
#include <vector>
#include <kiss_matcher/kdtree/kdtree.hpp>
#include <kiss_matcher/points/point_cloud.hpp>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

namespace gs3d_matcher {

/**
 * ΔEab neighborhood histogram descriptor.
 *
 * For each query (keypoint), finds all spatial neighbors within `radius` and
 * builds a normalized histogram of ab-plane color distances:
 *   ΔEab = sqrt((Δa*)² + (Δb*)²)
 * between the query point and each neighbor.
 *
 * This is a rotation-invariant, purely photometric descriptor — it captures
 * local color texture without relying on surface normals or geometry.
 * Dimensionality = num_bins.
 */
class DeltaEPFH {
public:
  DeltaEPFH(float radius, int num_bins, float max_delta_eab)
      : radius_(radius),
        num_bins_(num_bins),
        max_de_(std::max(max_delta_eab, 1e-6f)),
        inv_bw_(num_bins / std::max(max_delta_eab, 1e-6f)) {}

  int numBins() const { return num_bins_; }

  /**
   * Compute ΔEab histograms.
   *
   * @param keypoints      Query points (FasterPFH-selected subset or all points)
   * @param keypoint_lab   CIELAB (L*,a*,b*) for each keypoint — caller provides this
   *                       to avoid a 1-NN lookup (which hits a KISS-Matcher uint32/size_t bug)
   * @param all_points     Full dense cloud used for KD-tree construction
   * @param all_lab        CIELAB (L*,a*,b*) per point in all_points
   * @param out            Output: one num_bins-D descriptor per keypoint
   */
  void compute(const std::vector<Eigen::Vector3f>& keypoints,
               const std::vector<Eigen::Vector3f>& keypoint_lab,
               const std::vector<Eigen::Vector3f>& all_points,
               const std::vector<Eigen::Vector3f>& all_lab,
               std::vector<Eigen::VectorXf>& out) const {
    if (keypoints.empty() || all_points.empty()) {
      out.assign(keypoints.size(), Eigen::VectorXf::Zero(num_bins_));
      return;
    }

    // Build KD-tree over all_points
    kiss_matcher::PointCloud kd_cloud;
    kd_cloud.resize(all_points.size());
    for (size_t i = 0; i < all_points.size(); ++i)
      kd_cloud.point(i) << all_points[i].cast<double>(), 1.0;

    kiss_matcher::UnsafeKdTree<kiss_matcher::PointCloud> kdtree(kd_cloud);

    const size_t N = keypoints.size();
    const double sq_radius = static_cast<double>(radius_) * static_cast<double>(radius_);
    out.assign(N, Eigen::VectorXf::Zero(num_bins_));

    tbb::parallel_for(tbb::blocked_range<size_t>(0, N, 32),
      [&](const tbb::blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i < range.end(); ++i) {
          Eigen::Vector4d q;
          q << keypoints[i].cast<double>(), 1.0;

          const Eigen::Vector3f& c_lab = keypoint_lab[i];

          std::vector<std::pair<uint32_t, double>> results;
          kdtree.radius_search(q, sq_radius, results);
          if (results.empty()) continue;

          Eigen::VectorXf hist = Eigen::VectorXf::Zero(num_bins_);
          for (const auto& r : results) {
            if (r.second < 1e-10) continue;  // skip self (sq_dist≈0)
            const Eigen::Vector3f& nb_lab = all_lab[r.first];
            const float da = nb_lab[1] - c_lab[1];
            const float db = nb_lab[2] - c_lab[2];
            const float de = std::sqrt(da * da + db * db);
            const int bin = std::min(num_bins_ - 1, static_cast<int>(de * inv_bw_));
            hist[bin] += 1.0f;
          }

          const float s = hist.sum();
          if (s > 0.0f) hist /= s;
          out[i] = hist;
        }
      });
  }

private:
  float radius_;
  int   num_bins_;
  float max_de_;
  float inv_bw_;
};

}  // namespace gs3d_matcher

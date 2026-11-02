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
 * Color Point Feature Histogram (ColorPFH).
 *
 * Direct color analog of FPFH. For each keypoint, finds spatial neighbors
 * within `radius` and builds three sub-histograms from the CIELAB color
 * differences to each neighbor:
 *
 *   α  = atan2(Δb*, Δa*) / π  ∈ [-1, 1]   — hue direction of color change
 *   φ  = ΔL* / ΔE              ∈ [-1, 1]   — lightness fraction of change
 *   ψ  = ΔE / max_de           ∈ [ 0, 1]   — normalized color change magnitude
 *
 * where ΔE = sqrt(ΔL*² + Δa*² + Δb*²)  (full 3D LAB distance).
 *
 * Each sub-histogram has `num_bins` bins and is L1-normalized independently.
 * Total descriptor dimensionality = 3 × num_bins.
 *
 * This is rotation-invariant in 3D space: α and φ depend only on LAB color
 * vectors, not on the spatial orientation between points.
 */
class ColorPFH {
public:
  ColorPFH(float radius, int num_bins, float max_de)
      : radius_(radius),
        num_bins_(num_bins),
        max_de_(std::max(max_de, 1e-6f)) {}

  int totalDim() const { return 3 * num_bins_; }

  /**
   * Compute ColorPFH descriptors.
   *
   * @param keypoints     Query points (FasterPFH subset or full cloud)
   * @param keypoint_lab  LAB color for each keypoint (caller provides, avoids 1-NN lookup)
   * @param all_points    Full dense cloud for KD-tree
   * @param all_lab       LAB color for each point in all_points
   * @param out           Output: one (3×num_bins)-D descriptor per keypoint
   */
  void compute(const std::vector<Eigen::Vector3f>& keypoints,
               const std::vector<Eigen::Vector3f>& keypoint_lab,
               const std::vector<Eigen::Vector3f>& all_points,
               const std::vector<Eigen::Vector3f>& all_lab,
               std::vector<Eigen::VectorXf>& out) const {
    const int D = totalDim();
    if (keypoints.empty() || all_points.empty()) {
      out.assign(keypoints.size(), Eigen::VectorXf::Zero(D));
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
    out.assign(N, Eigen::VectorXf::Zero(D));

    tbb::parallel_for(tbb::blocked_range<size_t>(0, N, 32),
      [&](const tbb::blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i < range.end(); ++i) {
          Eigen::Vector4d q;
          q << keypoints[i].cast<double>(), 1.0;

          std::vector<std::pair<uint32_t, double>> results;
          kdtree.radius_search(q, sq_radius, results);
          if (results.empty()) continue;

          const Eigen::Vector3f& c_s = keypoint_lab[i];  // (L, a*, b*) of keypoint

          // Three sub-histograms: α, φ, ψ
          Eigen::VectorXf h_alpha = Eigen::VectorXf::Zero(num_bins_);
          Eigen::VectorXf h_phi   = Eigen::VectorXf::Zero(num_bins_);
          Eigen::VectorXf h_psi   = Eigen::VectorXf::Zero(num_bins_);

          int count = 0;
          for (const auto& r : results) {
            if (r.second < 1e-10) continue;  // skip self

            const Eigen::Vector3f& c_t = all_lab[r.first];
            const float dL = c_t[0] - c_s[0];
            const float da = c_t[1] - c_s[1];
            const float db = c_t[2] - c_s[2];
            const float dE = std::sqrt(dL * dL + da * da + db * db);
            if (dE < 1e-6f) continue;  // identical color, skip

            // α: hue direction of color change in ab-plane, mapped to [0,1]
            const float alpha = std::atan2(db, da) / static_cast<float>(M_PI);  // [-1,1]
            const int bin_a = std::min(num_bins_ - 1,
                                       static_cast<int>((alpha + 1.0f) * 0.5f * num_bins_));

            // φ: lightness fraction of color change, mapped [−1,1] → [0,1]
            const float phi = dL / dE;  // [-1, 1]
            const int bin_p = std::min(num_bins_ - 1,
                                       static_cast<int>((phi + 1.0f) * 0.5f * num_bins_));

            // ψ: normalized magnitude, mapped [0, max_de] → [0,1]
            const float psi = std::min(1.0f, dE / max_de_);
            const int bin_s = std::min(num_bins_ - 1,
                                       static_cast<int>(psi * num_bins_));

            h_alpha[bin_a] += 1.0f;
            h_phi[bin_p]   += 1.0f;
            h_psi[bin_s]   += 1.0f;
            ++count;
          }

          if (count == 0) continue;

          // L1-normalize each sub-histogram independently (like FPFH)
          const float sa = h_alpha.sum(); if (sa > 0) h_alpha /= sa;
          const float sp = h_phi.sum();   if (sp > 0) h_phi   /= sp;
          const float ss = h_psi.sum();   if (ss > 0) h_psi   /= ss;

          out[i].segment(0,           num_bins_) = h_alpha;
          out[i].segment(num_bins_,   num_bins_) = h_phi;
          out[i].segment(2*num_bins_, num_bins_) = h_psi;
        }
      });
  }

private:
  float radius_;
  int   num_bins_;
  float max_de_;
};

}  // namespace gs3d_matcher

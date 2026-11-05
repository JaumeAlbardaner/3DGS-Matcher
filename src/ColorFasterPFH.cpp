/**
 * Copyright 2024, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Hyungtae Lim, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#include "3dgs_matcher/ColorFasterPFH.hpp"
#include <stdexcept>

namespace gs3d_matcher {

void ColorFasterPFH::ComputeFeature(std::vector<Eigen::Vector3f>& points,
                                    std::vector<Eigen::VectorXf>& descriptors) {
  if (colors_.empty()) {
    throw std::runtime_error(
        "ColorFasterPFH::ComputeFeature: Colors not set! Call setColorAttributes() first.");
  }
  if (points_.empty()) {
    throw std::runtime_error(
        "ColorFasterPFH::ComputeFeature: Points not set! Call setInputCloud() first.");
  }
  if (points_.size() != colors_.size()) {
    throw std::runtime_error(
        "ColorFasterPFH::ComputeFeature: Points and colors size mismatch!");
  }

  const int color_dim = (use_color_ ? 3 : 0) + (use_opacity_ ? 1 : 0);
  const int cpfh_dim  = use_color_pfh_ ? 3 * color_pfh_bins_ : 0;
  const int total_dim = color_dim + cpfh_dim;

  // Every input point is a keypoint.
  const size_t n = points_.size();
  points = points_;

  descriptors.clear();
  descriptors.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    descriptors.emplace_back(total_dim);
    descriptors[i].setZero();
  }

  // Raw color block: R, G, B (optionally opacity), each scaled by color_weight.
  for (size_t i = 0; i < n; ++i) {
    int offset = 0;
    const Eigen::Vector3f rgb = colors_[i].head<3>();
    if (use_color_) {
      descriptors[i](offset++) = rgb(0) * color_weight_;
      descriptors[i](offset++) = rgb(1) * color_weight_;
      descriptors[i](offset++) = rgb(2) * color_weight_;
    }
    if (use_opacity_) {
      descriptors[i](offset++) = colors_[i](3) * color_weight_;
    }
  }

  // ColorPFH block: relative CIELAB hue-direction / lightness / magnitude histograms.
  if (use_color_pfh_ && color_pfh_) {
    std::vector<Eigen::Vector3f> all_lab(n);
    for (size_t i = 0; i < n; ++i)
      all_lab[i] = rgbToLab(colors_[i][0], colors_[i][1], colors_[i][2]);

    std::vector<Eigen::VectorXf> cpfh_hists;
    color_pfh_->compute(points, all_lab, points_, all_lab, cpfh_hists);

    for (size_t i = 0; i < n; ++i)
      for (int j = 0; j < cpfh_dim; ++j)
        descriptors[i](color_dim + j) = cpfh_hists[i](j) * color_weight_;
  }
}

}  // namespace gs3d_matcher

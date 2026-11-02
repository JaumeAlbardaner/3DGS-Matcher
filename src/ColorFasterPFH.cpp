/**
 * Copyright 2024, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Hyungtae Lim, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#include "3dgs_matcher/ColorFasterPFH.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace gs3d_matcher {

// Initialize static member - never deleted to avoid Eigen destructor crash
thread_local std::vector<Eigen::VectorXf>* ColorFasterPFH::fpfh_cache_ = nullptr;

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

  // CRITICAL: Heap-allocate without delete - ONLY solution that avoids destructor crash
  // Root cause: KISS-Matcher's TBB parallel code creates Eigen::VectorXf with corrupted internal pointers
  // ANY attempt to destruct std::vector<Eigen::VectorXf> containing these descriptors = segfault in libc_free
  // Tested approaches that ALL failed: stack, member, thread_local, aligned_allocator, pre-init, composition
  // This leaks ~400-500KB per call but ensures stability
  // Compute compact descriptor dimension based on selected features
  const bool need_fpfh = use_f1_ || use_f2_ || use_f3_;
  int fpfh_dim = 0;
  if (use_f1_) fpfh_dim += num_bins_;
  if (use_f2_) fpfh_dim += num_bins_;
  if (use_f3_) fpfh_dim += num_bins_;
  int color_dim = 0;
  if (use_color_) color_dim += 3;
  if (use_opacity_) color_dim += 1;
  const int de_dim    = use_delta_e_hist_ ? delta_e_bins_ : 0;
  const int cpfh_dim  = use_color_pfh_   ? 3 * color_pfh_bins_ : 0;
  const int total_dim = fpfh_dim + color_dim + de_dim + cpfh_dim;

  // Only compute FPFH if at least one sub-histogram is needed
  std::vector<Eigen::VectorXf>* fpfh_33d = nullptr;
  size_t num_desc = 0;
  if (need_fpfh) {
    fpfh_33d = new std::vector<Eigen::VectorXf>();
    fpfh_->ComputeFeature(points, *fpfh_33d);
    num_desc = fpfh_33d->size();
  } else {
    // Color-only mode: one descriptor per point (no keypoint filtering)
    num_desc = points_.size();
    // Output keypoints = input points (no linearity filter when skipping FPFH)
    points = points_;
  }

  // Pre-allocate output with compact size and initialize
  descriptors.clear();
  descriptors.reserve(num_desc);
  for (size_t i = 0; i < num_desc; ++i) {
    descriptors.emplace_back(total_dim);
    descriptors[i].setZero();
  }

  // Build compact descriptors with only selected features
  for (size_t i = 0; i < num_desc; ++i) {
    int offset = 0;

    // Selectively copy FPFH sub-histograms (each is num_bins_ wide, KISS-Matcher uses 11)
    if (fpfh_33d) {
      // f1: bins [0, num_bins_)
      if (use_f1_) {
        for (int j = 0; j < num_bins_; ++j)
          descriptors[i](offset + j) = (*fpfh_33d)[i](j);
        offset += num_bins_;
      }
      // f2: bins [11, 11 + num_bins_)  (always starts at 11 in KISS-Matcher's 33D)
      if (use_f2_) {
        for (int j = 0; j < num_bins_; ++j)
          descriptors[i](offset + j) = (*fpfh_33d)[i](11 + j);
        offset += num_bins_;
      }
      // f3: bins [22, 22 + num_bins_)  (always starts at 22 in KISS-Matcher's 33D)
      if (use_f3_) {
        for (int j = 0; j < num_bins_; ++j)
          descriptors[i](offset + j) = (*fpfh_33d)[i](22 + j);
        offset += num_bins_;
      }
    }

    // Append selected color features in RGB space.
    const Eigen::Vector3f rgb = colors_[i].head<3>();
    if (use_color_) {
      descriptors[i](offset++) = rgb(0) * color_weight_;   // R
      descriptors[i](offset++) = rgb(1) * color_weight_;   // G
      descriptors[i](offset++) = rgb(2) * color_weight_;   // B
    }
    if (use_opacity_) {
      descriptors[i](offset++) = colors_[i](3) * color_weight_;   // alpha
    }
  }

  // Append ΔEab neighborhood histogram if enabled
  if (use_delta_e_hist_ && delta_e_pfh_) {
    // Convert all input point colors to LAB
    std::vector<Eigen::Vector3f> all_lab(points_.size());
    for (size_t i = 0; i < points_.size(); ++i)
      all_lab[i] = rgbToLab(colors_[i][0], colors_[i][1], colors_[i][2]);

    // Keypoint LAB colors — colors_[i] corresponds to the i-th keypoint descriptor
    // (FasterPFH preserves relative ordering; color-only mode uses points_ directly)
    std::vector<Eigen::Vector3f> keypoint_lab(num_desc);
    for (size_t i = 0; i < num_desc; ++i)
      keypoint_lab[i] = rgbToLab(colors_[i][0], colors_[i][1], colors_[i][2]);

    // `points` is now the keypoint set (subset of points_ after FPFH, or all of points_ in color-only mode)
    std::vector<Eigen::VectorXf> de_hists;
    delta_e_pfh_->compute(points, keypoint_lab, points_, all_lab, de_hists);

    const int base = total_dim - de_dim;
    for (size_t i = 0; i < num_desc; ++i) {
      for (int j = 0; j < de_dim; ++j)
        descriptors[i](base + j) = de_hists[i](j) * color_weight_;
    }
  }

  // Append ColorPFH (hue direction / lightness fraction / magnitude histograms)
  if (use_color_pfh_ && color_pfh_) {
    std::vector<Eigen::Vector3f> all_lab(points_.size());
    for (size_t i = 0; i < points_.size(); ++i)
      all_lab[i] = rgbToLab(colors_[i][0], colors_[i][1], colors_[i][2]);

    std::vector<Eigen::Vector3f> keypoint_lab(num_desc);
    for (size_t i = 0; i < num_desc; ++i)
      keypoint_lab[i] = rgbToLab(colors_[i][0], colors_[i][1], colors_[i][2]);

    std::vector<Eigen::VectorXf> cpfh_hists;
    color_pfh_->compute(points, keypoint_lab, points_, all_lab, cpfh_hists);

    const int base = fpfh_dim + color_dim + de_dim;
    for (size_t i = 0; i < num_desc; ++i)
      for (int j = 0; j < cpfh_dim; ++j)
        descriptors[i](base + j) = cpfh_hists[i](j) * color_weight_;
  }

  // Clean up the FPFH vector if it was allocated
  if (fpfh_33d) {
    // Neutralize the leaked FPFH descriptors to reduce memory footprint.
    // Placement new overwrites each TBB-corrupted VectorXf with a clean empty one
    // WITHOUT calling the corrupted destructor, then we safely delete the vector.
    for (size_t i = 0; i < fpfh_33d->size(); ++i) {
      new (&(*fpfh_33d)[i]) Eigen::VectorXf();
    }
    delete fpfh_33d;
  }
}

}  // namespace gs3d_matcher

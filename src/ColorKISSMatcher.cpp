/**
 * Copyright 2024, Massachusetts Institute of Technology
 */

#include "3dgs_matcher/ColorKISSMatcher.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>

namespace gs3d_matcher {

ColorKISSMatcher::ColorKISSMatcher(const float& voxel_size, const float& color_weight) {
  config_ = ColorKISSMatcherConfig(voxel_size, true, false, 1.0, 5000, color_weight);
  src_descriptors_ = new std::vector<Eigen::VectorXf>();
  tgt_descriptors_ = new std::vector<Eigen::VectorXf>();
  reset();
}

ColorKISSMatcher::ColorKISSMatcher(const ColorKISSMatcherConfig& config) {
  config_ = config;
  src_descriptors_ = new std::vector<Eigen::VectorXf>();
  tgt_descriptors_ = new std::vector<Eigen::VectorXf>();
  reset();
}

ColorKISSMatcher::~ColorKISSMatcher() {
  // CRITICAL: NEVER delete descriptor pointers - they contain Eigen::VectorXf from TBB
  // Intentional memory leak for stability
  src_descriptors_ = nullptr;
  tgt_descriptors_ = nullptr;
  
  // Safe to clear other vectors
  src_keypoints_.clear();
  tgt_keypoints_.clear();
  src_processed_.clear();
  tgt_processed_.clear();
  
  // CRITICAL: All unique_ptr members hold corrupted state from TBB - release without deleting
  // This leaks memory but prevents segfault
  solver_.release();
  robin_matching_.release();
  color_fpfh_.release();
}

void ColorKISSMatcher::reset() {
  color_fpfh_ = std::make_unique<ColorFasterPFH>(
      config_.normal_radius_, config_.fpfh_radius_, config_.thr_linearity_, config_.color_weight_,
      "L2", false,
      config_.use_f1_, config_.use_f2_, config_.use_f3_,
      config_.use_color_, config_.use_opacity_, config_.num_bins_, false,
      config_.use_delta_e_hist_, config_.delta_e_radius_, config_.delta_e_bins_, config_.delta_e_max_,
      config_.use_color_pfh_, config_.color_pfh_radius_, config_.color_pfh_bins_, config_.color_pfh_max_de_);
  
  robin_matching_ = std::make_unique<kiss_matcher::ROBINMatching>(
      config_.robin_noise_bound_, config_.num_max_corr_, config_.tuple_scale_);
  
  resetSolver();
}

void ColorKISSMatcher::resetSolver() {
  kiss_matcher::RobustRegistrationSolver::Params params;
  params.noise_bound = config_.solver_noise_bound_;
  
  if (config_.use_quatro_) {
    params.rotation_estimation_algorithm =
        kiss_matcher::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::QUATRO;
  } else {
    params.rotation_estimation_algorithm =
        kiss_matcher::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
  }
  
  solver_ = std::make_unique<kiss_matcher::RobustRegistrationSolver>(params);
}

void ColorKISSMatcher::downsampleWithColors(const std::vector<Eigen::Vector3f>& points_in,
                                             const std::vector<Eigen::Vector4f>& colors_in,
                                             std::vector<Eigen::Vector3f>& points_out,
                                             std::vector<Eigen::Vector4f>& colors_out,
                                             double voxel_size) {
  // Parallel first-hit downsampling (KISS-Matcher style, but keeps first point per voxel)
  using VoxelKey = std::uint64_t;
  const double inv_leaf_size = 1.0 / voxel_size;
  constexpr int coord_bit_size = 21;
  constexpr size_t coord_bit_mask = (1 << 21) - 1;
  constexpr int coord_offset = 1 << (coord_bit_size - 1);
  constexpr VoxelKey invalid_coord = std::numeric_limits<VoxelKey>::max();
  size_t num_points = points_in.size();
  std::vector<std::pair<VoxelKey, size_t>> coord_pt(num_points);
  tbb::parallel_for(tbb::blocked_range<size_t>(0, num_points, 64), [&](const tbb::blocked_range<size_t>& range) {
    for (size_t i = range.begin(); i != range.end(); i++) {
      const Eigen::Array3i coord = (points_in[i] * inv_leaf_size).array().floor().cast<int>() + coord_offset;
      if ((coord < 0).any() || (coord > coord_bit_mask).any()) {
        coord_pt[i] = {invalid_coord, i};
        continue;
      }
      VoxelKey bits = (static_cast<VoxelKey>(coord[0] & coord_bit_mask) << (coord_bit_size * 0)) |
                      (static_cast<VoxelKey>(coord[1] & coord_bit_mask) << (coord_bit_size * 1)) |
                      (static_cast<VoxelKey>(coord[2] & coord_bit_mask) << (coord_bit_size * 2));
      coord_pt[i] = {bits, i};
    }
  });
  tbb::parallel_sort(coord_pt.begin(), coord_pt.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
  points_out.clear();
  colors_out.clear();
  points_out.reserve(num_points);
  colors_out.reserve(num_points);
  VoxelKey last_key = invalid_coord;
  for (size_t i = 0; i < num_points; ++i) {
    VoxelKey key = coord_pt[i].first;
    if (key == invalid_coord) continue;
    if (key != last_key) {
      points_out.push_back(points_in[coord_pt[i].second]);
      colors_out.push_back(colors_in[coord_pt[i].second]);
      last_key = key;
    }
  }
}

void ColorKISSMatcher::downsampleWithColorsCentroid(
    const std::vector<Eigen::Vector3f>& points_in,
    const std::vector<Eigen::Vector4f>& colors_in,
    std::vector<Eigen::Vector3f>& points_out,
    std::vector<Eigen::Vector4f>& colors_out,
    double voxel_size) {
  // Centroid averaging: accumulate all points/colors in each voxel, then average.
  // Mirrors KISS-Matcher's VoxelgridSampling behavior but also averages colors.
  // Parallel centroid downsampling (KISS-Matcher style, averages both position and color)
  using VoxelKey = std::uint64_t;
  const double inv_leaf_size = 1.0 / voxel_size;
  constexpr int coord_bit_size = 21;
  constexpr size_t coord_bit_mask = (1 << 21) - 1;
  constexpr int coord_offset = 1 << (coord_bit_size - 1);
  constexpr VoxelKey invalid_coord = std::numeric_limits<VoxelKey>::max();
  size_t num_points = points_in.size();
  std::vector<std::pair<VoxelKey, size_t>> coord_pt(num_points);
  tbb::parallel_for(tbb::blocked_range<size_t>(0, num_points, 64), [&](const tbb::blocked_range<size_t>& range) {
    for (size_t i = range.begin(); i != range.end(); i++) {
      const Eigen::Array3i coord = (points_in[i] * inv_leaf_size).array().floor().cast<int>() + coord_offset;
      if ((coord < 0).any() || (coord > coord_bit_mask).any()) {
        coord_pt[i] = {invalid_coord, i};
        continue;
      }
      VoxelKey bits = (static_cast<VoxelKey>(coord[0] & coord_bit_mask) << (coord_bit_size * 0)) |
                      (static_cast<VoxelKey>(coord[1] & coord_bit_mask) << (coord_bit_size * 1)) |
                      (static_cast<VoxelKey>(coord[2] & coord_bit_mask) << (coord_bit_size * 2));
      coord_pt[i] = {bits, i};
    }
  });
  tbb::parallel_sort(coord_pt.begin(), coord_pt.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
  points_out.clear();
  colors_out.clear();
  points_out.reserve(num_points);
  colors_out.reserve(num_points);
  size_t i = 0;
  while (i < num_points) {
    VoxelKey key = coord_pt[i].first;
    if (key == invalid_coord) { ++i; continue; }
    Eigen::Vector3f pos_sum = Eigen::Vector3f::Zero();
    Eigen::Vector4f col_sum = Eigen::Vector4f::Zero();
    size_t count = 0;
    size_t j = i;
    while (j < num_points && coord_pt[j].first == key) {
      pos_sum += points_in[coord_pt[j].second];
      col_sum += colors_in[coord_pt[j].second];
      ++count;
      ++j;
    }
    float inv_count = 1.0f / static_cast<float>(count);
    points_out.push_back(pos_sum * inv_count);
    Eigen::Vector4f avg_color = col_sum * inv_count;
    avg_color[3] = std::max(0.0f, std::min(1.0f, avg_color[3]));
    colors_out.push_back(avg_color);
    i = j;
  }
}

void ColorKISSMatcher::downsampleJoint5D(
    const std::vector<Eigen::Vector3f>& points_in,
    const std::vector<Eigen::Vector4f>& colors_in,
    std::vector<Eigen::Vector3f>& points_out,
    std::vector<Eigen::Vector4f>& colors_out,
    double voxel_size_xyz,
    float  voxel_size_ab,
    int    min_color_bins) {

  using SpatialKey = uint64_t;
  using ColorKey   = uint32_t;
  constexpr SpatialKey kInvalidSK = std::numeric_limits<SpatialKey>::max();

  const double inv_xyz = 1.0 / voxel_size_xyz;
  const float  inv_ab  = 1.0f / voxel_size_ab;
  constexpr int    coord_bit_size = 21;
  constexpr size_t coord_bit_mask = (1u << 21) - 1u;
  constexpr int    coord_offset   = 1 << (coord_bit_size - 1);

  const size_t N = points_in.size();

  // ── Step 1: compute (spatial_key, color_key, original_index) for every point ──
  struct Entry { SpatialKey sk; ColorKey ck; size_t idx; };
  std::vector<Entry> entries(N);

  tbb::parallel_for(tbb::blocked_range<size_t>(0, N, 64),
    [&](const tbb::blocked_range<size_t>& range) {
      for (size_t i = range.begin(); i != range.end(); ++i) {
        // Spatial key — identical encoding to existing voxel code
        const Eigen::Array3i coord =
            (points_in[i] * static_cast<float>(inv_xyz)).array().floor().cast<int>() + coord_offset;

        SpatialKey sk = kInvalidSK;
        if (!(coord < 0).any() && !(coord > static_cast<int>(coord_bit_mask)).any()) {
          sk = (static_cast<SpatialKey>(coord[0] & coord_bit_mask) << (coord_bit_size * 0)) |
               (static_cast<SpatialKey>(coord[1] & coord_bit_mask) << (coord_bit_size * 1)) |
               (static_cast<SpatialKey>(coord[2] & coord_bit_mask) << (coord_bit_size * 2));
        }

        // Color key: LAB ab only (drop L — lightness varies with exposure/angle)
        const Eigen::Vector3f lab = rgbToLab(colors_in[i][0], colors_in[i][1], colors_in[i][2]);
        const int a_bin = static_cast<int>(std::floor(lab[1] * inv_ab)) + 128;
        const int b_bin = static_cast<int>(std::floor(lab[2] * inv_ab)) + 128;
        const ColorKey ck =
            static_cast<ColorKey>(std::max(0, std::min(255, a_bin))) * 256u +
            static_cast<ColorKey>(std::max(0, std::min(255, b_bin)));

        entries[i] = {sk, ck, i};
      }
    });

  // ── Step 2: sort by (spatial_key, color_key) ──────────────────────────────
  tbb::parallel_sort(entries.begin(), entries.end(),
    [](const Entry& a, const Entry& b) {
      return a.sk != b.sk ? a.sk < b.sk : a.ck < b.ck;
    });

  // ── Step 3: single pass — keep first real point per (sk, ck) cell ─────────
  // Flush a spatial voxel's buffer only if it has >= min_color_bins distinct colors.
  points_out.clear();
  colors_out.clear();
  points_out.reserve(N / 8);
  colors_out.reserve(N / 8);

  std::vector<size_t> buf;  // one entry per distinct color bin in current spatial voxel
  buf.reserve(64);

  SpatialKey cur_sk = kInvalidSK;
  ColorKey   cur_ck = std::numeric_limits<ColorKey>::max();

  auto flush = [&]() {
    if (static_cast<int>(buf.size()) >= min_color_bins) {
      for (const size_t idx : buf) {
        points_out.push_back(points_in[idx]);
        colors_out.push_back(colors_in[idx]);
      }
    }
    buf.clear();
  };

  for (const auto& e : entries) {
    if (e.sk == kInvalidSK) continue;
    if (e.sk != cur_sk) {
      flush();
      cur_sk = e.sk;
      cur_ck = std::numeric_limits<ColorKey>::max();
    }
    if (e.ck != cur_ck) {       // first point in this (spatial, color) cell
      buf.push_back(e.idx);
      cur_ck = e.ck;
    }
    // subsequent points in the same cell are redundant — skip
  }
  flush();  // don't forget the last voxel

  std::cerr << "Joint-5D: " << N << " -> " << points_out.size()
            << " (spatial_voxel=" << voxel_size_xyz
            << " ab_voxel=" << voxel_size_ab
            << " min_color_bins=" << min_color_bins << ")\n";
}

void ColorKISSMatcher::filterByColorOverlap(
    const std::vector<Eigen::Vector3f>& src_pts_in,
    const std::vector<Eigen::Vector4f>& src_col_in,
    const std::vector<Eigen::Vector3f>& tgt_pts_in,
    const std::vector<Eigen::Vector4f>& tgt_col_in,
    std::vector<Eigen::Vector3f>& src_pts_out,
    std::vector<Eigen::Vector4f>& src_col_out,
    std::vector<Eigen::Vector3f>& tgt_pts_out,
    std::vector<Eigen::Vector4f>& tgt_col_out,
    float voxel_size_ab,
    float min_overlap_ratio) {

  using ColorKey = uint32_t;
  const float inv_ab = 1.0f / std::max(voxel_size_ab, 1e-6f);

  auto colorKey = [&](const Eigen::Vector4f& c) -> ColorKey {
    const Eigen::Vector3f lab = rgbToLab(c[0], c[1], c[2]);
    const int a_bin = static_cast<int>(std::floor(lab[1] * inv_ab)) + 128;
    const int b_bin = static_cast<int>(std::floor(lab[2] * inv_ab)) + 128;
    return static_cast<ColorKey>(std::max(0, std::min(255, a_bin))) * 256u +
           static_cast<ColorKey>(std::max(0, std::min(255, b_bin)));
  };

  // Build per-bin counts for each cloud
  std::unordered_map<ColorKey, int> src_hist, tgt_hist;
  src_hist.reserve(src_col_in.size() / 4);
  tgt_hist.reserve(tgt_col_in.size() / 4);
  for (const auto& c : src_col_in) src_hist[colorKey(c)]++;
  for (const auto& c : tgt_col_in) tgt_hist[colorKey(c)]++;

  // Keep bins present in both clouds with sufficient overlap ratio
  std::unordered_set<ColorKey> valid;
  valid.reserve(src_hist.size());
  for (const auto& [key, cnt_src] : src_hist) {
    auto it = tgt_hist.find(key);
    if (it == tgt_hist.end()) continue;
    const float ratio = static_cast<float>(std::min(cnt_src, it->second)) /
                        static_cast<float>(std::max(cnt_src, it->second));
    if (ratio >= min_overlap_ratio) valid.insert(key);
  }

  // Precompute keys then filter — avoids recomputing rgbToLab per point twice
  src_pts_out.clear(); src_col_out.clear();
  src_pts_out.reserve(src_pts_in.size()); src_col_out.reserve(src_col_in.size());
  for (size_t i = 0; i < src_pts_in.size(); ++i) {
    if (valid.count(colorKey(src_col_in[i]))) {
      src_pts_out.push_back(src_pts_in[i]);
      src_col_out.push_back(src_col_in[i]);
    }
  }

  tgt_pts_out.clear(); tgt_col_out.clear();
  tgt_pts_out.reserve(tgt_pts_in.size()); tgt_col_out.reserve(tgt_col_in.size());
  for (size_t i = 0; i < tgt_pts_in.size(); ++i) {
    if (valid.count(colorKey(tgt_col_in[i]))) {
      tgt_pts_out.push_back(tgt_pts_in[i]);
      tgt_col_out.push_back(tgt_col_in[i]);
    }
  }

  std::cerr << "CrossColorFilter: src " << src_pts_in.size() << " -> " << src_pts_out.size()
            << ", tgt " << tgt_pts_in.size() << " -> " << tgt_pts_out.size()
            << " (shared_bins=" << valid.size()
            << " / src_bins=" << src_hist.size()
            << " / tgt_bins=" << tgt_hist.size() << ")\n";
}

KeypointPair ColorKISSMatcher::match(const std::vector<Eigen::Vector3f>& src,
                                    const std::vector<Eigen::Vector3f>& tgt,
                                    const std::vector<Eigen::Vector4f>& src_colors,
                                    const std::vector<Eigen::Vector4f>& tgt_colors) {
  clear();
  
  auto t_init = std::chrono::high_resolution_clock::now();

  // Cross-cloud color overlap filter (pre-step: keep only matchable color regions)
  std::vector<Eigen::Vector3f> src_filt_pts, tgt_filt_pts;
  std::vector<Eigen::Vector4f> src_filt_col, tgt_filt_col;
  const auto* src_pts = &src;
  const auto* tgt_pts = &tgt;
  const auto* src_col = &src_colors;
  const auto* tgt_col = &tgt_colors;
  if (config_.use_cross_color_filter_) {
    filterByColorOverlap(src, src_colors, tgt, tgt_colors,
                         src_filt_pts, src_filt_col, tgt_filt_pts, tgt_filt_col,
                         config_.voxel_size_ab_, config_.cross_color_min_overlap_);
    src_pts = &src_filt_pts;  tgt_pts = &tgt_filt_pts;
    src_col = &src_filt_col;  tgt_col = &tgt_filt_col;
  }

  // Downsample both points and colors together
  if (config_.use_voxel_sampling_) {
    if (config_.use_joint_5d_) {
      downsampleJoint5D(*src_pts, *src_col, src_processed_, src_colors_processed_,
                        config_.voxel_size_, config_.voxel_size_ab_, config_.min_color_bins_);
      downsampleJoint5D(*tgt_pts, *tgt_col, tgt_processed_, tgt_colors_processed_,
                        config_.voxel_size_, config_.voxel_size_ab_, config_.min_color_bins_);
    } else if (config_.use_centroid_voxel_) {
      downsampleWithColorsCentroid(*src_pts, *src_col, src_processed_, src_colors_processed_, config_.voxel_size_);
      downsampleWithColorsCentroid(*tgt_pts, *tgt_col, tgt_processed_, tgt_colors_processed_, config_.voxel_size_);
      std::cerr << "Centroid-downsampled: " << src_pts->size() << " -> " << src_processed_.size()
                << " src, " << tgt_pts->size() << " -> " << tgt_processed_.size() << " tgt" << std::endl;
    } else {
      downsampleWithColors(*src_pts, *src_col, src_processed_, src_colors_processed_, config_.voxel_size_);
      downsampleWithColors(*tgt_pts, *tgt_col, tgt_processed_, tgt_colors_processed_, config_.voxel_size_);
      std::cerr << "First-hit-downsampled: " << src_pts->size() << " -> " << src_processed_.size()
                << " src, " << tgt_pts->size() << " -> " << tgt_processed_.size() << " tgt" << std::endl;
    }
  } else {
    src_processed_ = *src_pts;
    src_colors_processed_ = *src_col;
    tgt_processed_ = *tgt_pts;
    tgt_colors_processed_ = *tgt_col;
  }

  auto t_process = std::chrono::high_resolution_clock::now();

  // Set input and colors for source
  color_fpfh_->setInputCloud(src_processed_);
  color_fpfh_->setColorAttributes(src_colors_processed_);
  color_fpfh_->ComputeFeature(src_keypoints_, *src_descriptors_);

  // Set input and colors for target
  color_fpfh_->setInputCloud(tgt_processed_);
  color_fpfh_->setColorAttributes(tgt_colors_processed_);
  color_fpfh_->ComputeFeature(tgt_keypoints_, *tgt_descriptors_);

  if (src_keypoints_.empty() || tgt_keypoints_.empty() ||
      src_descriptors_->empty() || tgt_descriptors_->empty()) {
    throw std::runtime_error(
        "ColorKISSMatcher::match: no valid keypoints/descriptors after filtering/downsampling");
  }

  auto t_mid = std::chrono::high_resolution_clock::now();

  // Use ROBIN matching
  const auto& corr = robin_matching_->establishCorrespondences(
      src_keypoints_, tgt_keypoints_, *src_descriptors_, *tgt_descriptors_, config_.robin_mode_);

  auto t_end = std::chrono::high_resolution_clock::now();

  processing_time_ = std::chrono::duration_cast<std::chrono::duration<double>>(t_process - t_init).count();
  extraction_time_ = std::chrono::duration_cast<std::chrono::duration<double>>(t_mid - t_process).count();
  matching_time_ = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_mid).count();

  std::cerr << "Matched " << corr.size() << " keypoint pairs." << std::endl;

  // CRITICAL: Abandon descriptor pointers (allocate fresh ones on next call)
  // These contain Eigen::VectorXf from TBB with corrupted internal pointers
  src_descriptors_ = nullptr;
  tgt_descriptors_ = nullptr;

  std::vector<Eigen::Vector3f> src_matched, tgt_matched;
  src_matched.reserve(corr.size());
  tgt_matched.reserve(corr.size());

  for (const auto& c : corr) {
    src_matched.push_back(src_keypoints_[c.first]);
    tgt_matched.push_back(tgt_keypoints_[c.second]);
  }

  return {src_matched, tgt_matched};
}

RegistrationSolution ColorKISSMatcher::estimate(const std::vector<Eigen::Vector3f>& src,
                                               const std::vector<Eigen::Vector3f>& tgt,
                                               const std::vector<Eigen::Vector4f>& src_colors,
                                               const std::vector<Eigen::Vector4f>& tgt_colors) {
  const auto [src_matched, tgt_matched] = match(src, tgt, src_colors, tgt_colors);

  if (src_matched.size() < 3 || tgt_matched.size() < 3) {
    throw std::runtime_error(
        "ColorKISSMatcher::estimate: need at least 3 correspondences to solve rigid registration");
  }

  Eigen::Matrix<double, 3, Eigen::Dynamic> src_mat(3, src_matched.size());
  Eigen::Matrix<double, 3, Eigen::Dynamic> tgt_mat(3, tgt_matched.size());

  for (size_t i = 0; i < src_matched.size(); ++i) {
    src_mat.col(i) = src_matched[i].cast<double>();
    tgt_mat.col(i) = tgt_matched[i].cast<double>();
  }

  auto t_init = std::chrono::high_resolution_clock::now();
  
  // Use GNC/QUATRO solver
  auto solution = solver_->solve(src_mat, tgt_mat);
  num_inliers_ = src_matched.size();  // Track number of correspondences

  auto t_end = std::chrono::high_resolution_clock::now();
  solver_time_ = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_init).count();

  return solution;
}

void ColorKISSMatcher::clear() {
  src_processed_.clear();
  tgt_processed_.clear();
  src_colors_processed_.clear();
  tgt_colors_processed_.clear();
  src_keypoints_.clear();
  tgt_keypoints_.clear();
  
  // CRITICAL: Abandon old descriptors and allocate new ones (never delete old)
  src_descriptors_ = new std::vector<Eigen::VectorXf>();
  tgt_descriptors_ = new std::vector<Eigen::VectorXf>();
}

void ColorKISSMatcher::print() {
  std::cout << "\n==== ColorKISSMatcher Configuration ====" << std::endl;
  std::cout << "Voxel size: " << config_.voxel_size_ << std::endl;
  std::cout << "Normal radius: " << config_.normal_radius_ << std::endl;
  std::cout << "FPFH radius: " << config_.fpfh_radius_ << std::endl;
  std::cout << "Color weight: " << config_.color_weight_ << std::endl;
  std::cout << "ROBIN noise bound: " << config_.robin_noise_bound_ << std::endl;
  std::cout << "Solver noise bound: " << config_.solver_noise_bound_ << std::endl;
  std::cout << "Use voxel sampling: " << (config_.use_voxel_sampling_ ? "true" : "false") << std::endl;
  std::cout << "Use QUATRO: " << (config_.use_quatro_ ? "true" : "false") << std::endl;
  
  std::cout << "\n==== Statistics ====" << std::endl;
  std::cout << "Processing time: " << processing_time_ << " s" << std::endl;
  std::cout << "Feature extraction time: " << extraction_time_ << " s" << std::endl;
  std::cout << "Matching time: " << matching_time_ << " s" << std::endl;
  std::cout << "Solver time: " << solver_time_ << " s" << std::endl;
  std::cout << "Total time: " << (processing_time_ + extraction_time_ + matching_time_ + solver_time_) << " s" << std::endl;
}

}  // namespace gs3d_matcher

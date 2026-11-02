// SPDX-FileCopyrightText: Copyright 2024 3DGS-Matcher authors
// SPDX-License-Identifier: MIT
#pragma once

// Multimodal ICP: color (and optionally opacity) enters the correspondence
// search by augmenting the position feature vector.  Two variants are exposed:
//
//   align_color_correspondence  –  color guides kNN only; SVD transform is uniform
//   align_color_full            –  color guides kNN AND weights the SVD transform
//
// Both are warm-startable from an initial Isometry3d and produce an
// Isometry3d result (T maps source → target).

#include <Eigen/Core>
#include <Eigen/SVD>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

namespace gs3d_matcher {

// ── Config ────────────────────────────────────────────────────────────────────

struct MultimodalICPConfig {
  double color_scale            = 0.5;   // γ: color contribution in feature space (0 = geometry-only)
  double opacity_scale          = 0.0;   // δ: opacity contribution (0 = disabled)
  double alpha                  = 5.0;   // weight decay for color distance  (full variant only)
  double beta                   = 0.0;   // weight decay for opacity distance (full variant only)
  int    max_iters              = 50;
  double tolerance              = 1e-4;
  double downsampling_resolution = 0.0;  // voxel size for pre-ICP downsampling (0 = no downsampling)
};

struct MultimodalICPResult {
  Eigen::Isometry3d T  = Eigen::Isometry3d::Identity();
  double rmse          = -1.0;
  int    iterations    =  0;
  bool   converged     = false;
};

// ── Internal helpers ──────────────────────────────────────────────────────────

namespace detail {

// Voxel grid downsampling that preserves per-point colors.
// For each occupied voxel the first point that falls into it is kept
// (deterministic and fast — no centroid averaging needed for ICP).
inline void voxel_downsample(
    const std::vector<Eigen::Vector3f>& pts,
    const std::vector<Eigen::Vector4f>& cols,
    float voxel_size,
    std::vector<Eigen::Vector3f>& out_pts,
    std::vector<Eigen::Vector4f>& out_cols)
{
  const float inv_v = 1.0f / voxel_size;
  std::unordered_map<std::int64_t, std::size_t> vox;
  vox.reserve(pts.size());
  out_pts.clear(); out_cols.clear();
  out_pts.reserve(pts.size() / 4);
  out_cols.reserve(pts.size() / 4);

  for (std::size_t i = 0; i < pts.size(); ++i) {
    const auto& p = pts[i];
    std::int64_t ix = static_cast<std::int64_t>(std::floor(p.x() * inv_v));
    std::int64_t iy = static_cast<std::int64_t>(std::floor(p.y() * inv_v));
    std::int64_t iz = static_cast<std::int64_t>(std::floor(p.z() * inv_v));
    std::int64_t key = (ix * 100003LL + iy) * 100003LL + iz;
    if (vox.emplace(key, out_pts.size()).second) {
      out_pts.push_back(p);
      out_cols.push_back(cols[i]);
    }
  }
}

// Feature vector: [x_norm, y_norm, z_norm, γ·r, γ·g, γ·b, δ·a]
// Positions are normalised to [0,1] w.r.t. a fixed bounding box so that γ
// expresses color influence in units of scene extent.
inline int feature_dim(bool use_color, bool use_opacity) {
  return 3 + (use_color ? 3 : 0) + (use_opacity ? 1 : 0);
}

// Fill one feature into a pre-allocated flat buffer at offset i*dim.
inline void fill_feature(
    float* dst,
    const Eigen::Vector3f& pos,
    const Eigen::Vector4f& col,
    const Eigen::Vector3f& bbox_min,
    const Eigen::Vector3f& bbox_range,
    float cs, float os,
    bool use_color, bool use_opacity)
{
  Eigen::Vector3f pn = (pos - bbox_min).cwiseQuotient(bbox_range);
  dst[0] = pn.x(); dst[1] = pn.y(); dst[2] = pn.z();
  int k = 3;
  if (use_color)   { dst[k++] = col.x()*cs; dst[k++] = col.y()*cs; dst[k++] = col.z()*cs; }
  if (use_opacity) { dst[k]   = col.w()*os; }
}

inline float fdist2(const float* a, const float* b, int dim) {
  float d = 0.f;
  for (int k = 0; k < dim; ++k) { float dd = a[k]-b[k]; d += dd*dd; }
  return d;
}

// Brute-force nearest neighbour in augmented feature space, parallelised over
// source points with TBB. Returns correspondence index into target for each src.
inline std::vector<int> find_nn(
    const std::vector<float>& src_f, int n_src,
    const std::vector<float>& tgt_f, int n_tgt,
    int dim)
{
  std::vector<int> corr(n_src);
  tbb::parallel_for(tbb::blocked_range<int>(0, n_src),
    [&](const tbb::blocked_range<int>& r) {
      for (int i = r.begin(); i < r.end(); ++i) {
        float best = std::numeric_limits<float>::max();
        int   best_j = 0;
        const float* si = src_f.data() + static_cast<ptrdiff_t>(i) * dim;
        for (int j = 0; j < n_tgt; ++j) {
          float d = fdist2(si, tgt_f.data() + static_cast<ptrdiff_t>(j) * dim, dim);
          if (d < best) { best = d; best_j = j; }
        }
        corr[i] = best_j;
      }
    });
  return corr;
}

// Weighted SVD rigid transform.
// Returns T such that T * src[i] ≈ tgt[corr[i]] in a weighted least-squares sense.
inline Eigen::Isometry3d weighted_svd(
    const std::vector<Eigen::Vector3f>& src,
    const std::vector<Eigen::Vector3f>& tgt,
    const std::vector<int>&             corr,
    const std::vector<float>&           w)
{
  double wsum = 0.0;
  Eigen::Vector3d ps = Eigen::Vector3d::Zero();
  Eigen::Vector3d pt = Eigen::Vector3d::Zero();
  for (int i = 0, n = static_cast<int>(src.size()); i < n; ++i) {
    wsum += w[i];
    ps   += w[i] * src[i].cast<double>();
    pt   += w[i] * tgt[corr[i]].cast<double>();
  }
  ps /= wsum; pt /= wsum;

  Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
  for (int i = 0, n = static_cast<int>(src.size()); i < n; ++i)
    H += w[i] * (src[i].cast<double>() - ps) *
                (tgt[corr[i]].cast<double>() - pt).transpose();

  Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d R = svd.matrixV() * svd.matrixU().transpose();
  if (R.determinant() < 0) {
    Eigen::Matrix3d V = svd.matrixV(); V.col(2) *= -1;
    R = V * svd.matrixU().transpose();
  }

  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear()      = R;
  T.translation() = pt - R * ps;
  return T;
}

inline double compute_rmse(
    const std::vector<Eigen::Vector3f>& src,
    const std::vector<Eigen::Vector3f>& tgt,
    const std::vector<int>&             corr)
{
  double sum = 0.0;
  for (int i = 0, n = static_cast<int>(src.size()); i < n; ++i)
    sum += (src[i].cast<double>() - tgt[corr[i]].cast<double>()).squaredNorm();
  return std::sqrt(sum / src.size());
}

} // namespace detail

// ── Core ICP loop ─────────────────────────────────────────────────────────────

// use_weighted_transform = false  →  color_correspondence variant
// use_weighted_transform = true   →  color_full variant
inline MultimodalICPResult align_multimodal(
    std::vector<Eigen::Vector3f>        src_pts,   // by value — modified in place
    std::vector<Eigen::Vector4f>        src_cols,  // by value — may be replaced by downsampled
    std::vector<Eigen::Vector3f>        tgt_pts,   // by value — may be replaced by downsampled
    std::vector<Eigen::Vector4f>        tgt_cols,  // by value — may be replaced by downsampled
    const MultimodalICPConfig&          cfg,
    const Eigen::Isometry3d&            init_T,
    bool                                use_weighted_transform)
{
  // Optional voxel downsampling — keeps the method consistent with small_gicp
  // GICP which also downsamples internally at the same resolution.
  if (cfg.downsampling_resolution > 0.0) {
    const float vs = static_cast<float>(cfg.downsampling_resolution);
    std::vector<Eigen::Vector3f> ds_pts;
    std::vector<Eigen::Vector4f> ds_cols;
    detail::voxel_downsample(src_pts, src_cols, vs, ds_pts, ds_cols);
    src_pts = std::move(ds_pts); src_cols = std::move(ds_cols);
    detail::voxel_downsample(tgt_pts, tgt_cols, vs, ds_pts, ds_cols);
    tgt_pts = std::move(ds_pts); tgt_cols = std::move(ds_cols);
  }

  // Apply initial transform to source positions (colors are pose-invariant).
  for (auto& p : src_pts)
    p = (init_T * p.cast<double>()).cast<float>();

  // Bounding box over initial src ∪ tgt — fixed for the entire run so that
  // the normalised position scale is consistent across iterations.
  Eigen::Vector3f bmin = tgt_pts[0], bmax = tgt_pts[0];
  for (const auto& p : tgt_pts) { bmin = bmin.cwiseMin(p); bmax = bmax.cwiseMax(p); }
  for (const auto& p : src_pts) { bmin = bmin.cwiseMin(p); bmax = bmax.cwiseMax(p); }
  Eigen::Vector3f brange = (bmax - bmin).cwiseMax(1e-6f);

  const bool  use_color   = cfg.color_scale   > 0.0;
  const bool  use_opacity = cfg.opacity_scale > 0.0;
  const int   dim         = detail::feature_dim(use_color, use_opacity);
  const int   n_src       = static_cast<int>(src_pts.size());
  const int   n_tgt       = static_cast<int>(tgt_pts.size());
  const float cs          = static_cast<float>(cfg.color_scale);
  const float os          = static_cast<float>(cfg.opacity_scale);

  // Build fixed target features once.
  std::vector<float> tgt_f(static_cast<size_t>(n_tgt) * dim);
  for (int j = 0; j < n_tgt; ++j)
    detail::fill_feature(tgt_f.data() + static_cast<ptrdiff_t>(j)*dim,
                         tgt_pts[j], tgt_cols[j], bmin, brange, cs, os,
                         use_color, use_opacity);

  Eigen::Isometry3d T_total = init_T;
  double prev_rmse = std::numeric_limits<double>::max();
  MultimodalICPResult result;

  for (int iter = 0; iter < cfg.max_iters; ++iter) {
    // Source features rebuilt every iteration (positions change, colors fixed).
    std::vector<float> src_f(static_cast<size_t>(n_src) * dim);
    for (int i = 0; i < n_src; ++i)
      detail::fill_feature(src_f.data() + static_cast<ptrdiff_t>(i)*dim,
                           src_pts[i], src_cols[i], bmin, brange, cs, os,
                           use_color, use_opacity);

    auto corr = detail::find_nn(src_f, n_src, tgt_f, n_tgt, dim);

    // Per-correspondence weights (uniform for correspondence-only variant).
    std::vector<float> weights(n_src, 1.0f);
    if (use_weighted_transform && (use_color || use_opacity)) {
      const float fa = static_cast<float>(cfg.alpha);
      const float fb = static_cast<float>(cfg.beta);
      for (int i = 0; i < n_src; ++i) {
        const auto& sc = src_cols[i];
        const auto& tc = tgt_cols[corr[i]];
        float w = 1.0f;
        if (use_color) {
          float dr=sc.x()-tc.x(), dg=sc.y()-tc.y(), db=sc.z()-tc.z();
          float dc = std::sqrt((dr*dr + dg*dg + db*db) / 3.0f);
          w *= std::exp(-fa * dc * dc);
        }
        if (use_opacity) {
          float da = std::abs(sc.w() - tc.w());
          w *= std::exp(-fb * da * da);
        }
        weights[i] = w;
      }
    }

    Eigen::Isometry3d dT = detail::weighted_svd(src_pts, tgt_pts, corr, weights);

    for (auto& p : src_pts)
      p = (dT * p.cast<double>()).cast<float>();
    T_total = dT * T_total;

    double rmse = detail::compute_rmse(src_pts, tgt_pts, corr);
    result.iterations = iter + 1;
    if (std::abs(prev_rmse - rmse) < cfg.tolerance) {
      result.converged = true;
      result.rmse = rmse;
      break;
    }
    prev_rmse  = rmse;
    result.rmse = rmse;
  }

  result.T = T_total;
  return result;
}

// ── Public API ────────────────────────────────────────────────────────────────

// Color guides which points are matched (augmented kNN); transform is uniform SVD.
// At color_scale=0 this degenerates to point-to-point ICP (no covariances).
inline MultimodalICPResult align_color_correspondence(
    std::vector<Eigen::Vector3f> src_pts,
    std::vector<Eigen::Vector4f> src_cols,
    std::vector<Eigen::Vector3f> tgt_pts,
    std::vector<Eigen::Vector4f> tgt_cols,
    const MultimodalICPConfig& cfg,
    const Eigen::Isometry3d& init_T = Eigen::Isometry3d::Identity())
{
  return align_multimodal(
      std::move(src_pts), std::move(src_cols),
      std::move(tgt_pts), std::move(tgt_cols),
      cfg, init_T, false);
}

// Color guides kNN AND weights each correspondence in the SVD transform.
inline MultimodalICPResult align_color_full(
    std::vector<Eigen::Vector3f> src_pts,
    std::vector<Eigen::Vector4f> src_cols,
    std::vector<Eigen::Vector3f> tgt_pts,
    std::vector<Eigen::Vector4f> tgt_cols,
    const MultimodalICPConfig& cfg,
    const Eigen::Isometry3d& init_T = Eigen::Isometry3d::Identity())
{
  return align_multimodal(
      std::move(src_pts), std::move(src_cols),
      std::move(tgt_pts), std::move(tgt_cols),
      cfg, init_T, true);
}

} // namespace gs3d_matcher

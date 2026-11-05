/**
 * Copyright 2024, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Hyungtae Lim, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */
#pragma once

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <cmath>
#include <memory>
#include <vector>
#include "3dgs_matcher/ColorPFH.hpp"

namespace gs3d_matcher {

/**
 * @brief Convert sRGB [0,1] to CIELAB.
 *        L* in [0,100], a* ~ [-128,127], b* ~ [-128,127].
 */
inline Eigen::Vector3f rgbToLab(float r, float g, float b) {
  // sRGB → linear RGB
  auto linearize = [](float c) -> float {
    return (c <= 0.04045f) ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
  };
  float rl = linearize(r);
  float gl = linearize(g);
  float bl = linearize(b);

  // Linear RGB → XYZ (D65)
  float x = 0.4124564f * rl + 0.3575761f * gl + 0.1804375f * bl;
  float y = 0.2126729f * rl + 0.7151522f * gl + 0.0721750f * bl;
  float z = 0.0193339f * rl + 0.1191920f * gl + 0.9503041f * bl;

  // XYZ → Lab
  constexpr float Xn = 0.95047f, Yn = 1.0f, Zn = 1.08883f;
  auto f = [](float t) -> float {
    constexpr float delta = 6.0f / 29.0f;
    constexpr float delta3 = delta * delta * delta;
    if (t > delta3) return std::cbrt(t);
    return t / (3.0f * delta * delta) + 4.0f / 29.0f;
  };

  float fx = f(x / Xn);
  float fy = f(y / Yn);
  float fz = f(z / Zn);

  float L = 116.0f * fy - 16.0f;
  float a = 500.0f * (fx - fy);
  float bLab = 200.0f * (fy - fz);

  return Eigen::Vector3f(L, a, bLab);
}

/**
 * @brief Per-keypoint color descriptor used by 3DGS-Matcher.
 *
 * Every input point is treated as a keypoint (no geometric pre-filtering). The
 * descriptor is the raw normalized RGB block (optionally with opacity)
 * concatenated with the ColorPFH relative-CIELAB histogram:
 *
 *   [ R G B (A) ] ++ [ ColorPFH (3 × color_pfh_bins) ]
 *
 * Each block is scaled by color_weight.
 */
class ColorFasterPFH {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit ColorFasterPFH(const float color_weight     = 1.0f,
                          const bool  use_color         = true,
                          const bool  use_opacity       = true,
                          const bool  use_color_pfh     = true,
                          const float color_pfh_radius  = 1.0f,
                          const int   color_pfh_bins    = 11,
                          const float color_pfh_max_de  = 100.0f)
      : color_weight_(color_weight),
        use_color_(use_color),
        use_opacity_(use_opacity),
        use_color_pfh_(use_color_pfh),
        color_pfh_bins_(color_pfh_bins) {
    if (use_color_pfh_)
      color_pfh_ = std::make_unique<ColorPFH>(color_pfh_radius, color_pfh_bins, color_pfh_max_de);
  }

  inline void setInputCloud(const std::vector<Eigen::Vector3f>& points) { points_ = points; }

  template <typename Alloc>
  inline void setColorAttributes(const std::vector<Eigen::Vector4f, Alloc>& colors) {
    if (!points_.empty() && colors.size() != points_.size()) {
      throw std::runtime_error(
          "ColorFasterPFH::setColorAttributes: Number of colors (" + std::to_string(colors.size()) +
          ") does not match number of points (" + std::to_string(points_.size()) + ")");
    }
    colors_.assign(colors.begin(), colors.end());
  }

  // On return, `points` is the keypoint set (== input cloud) and `descriptors`
  // holds one feature vector per keypoint.
  void ComputeFeature(std::vector<Eigen::Vector3f>& points,
                      std::vector<Eigen::VectorXf>& descriptors);

  inline float getColorWeight() const { return color_weight_; }
  inline void setColorWeight(float weight) { color_weight_ = weight; }

 private:
  std::vector<Eigen::Vector3f> points_;
  std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f>> colors_;
  float color_weight_ = 1.0f;
  bool  use_color_    = true;   ///< Include raw R, G, B
  bool  use_opacity_  = false;  ///< Include alpha/opacity
  bool  use_color_pfh_ = true;  ///< Append ColorPFH (3 × color_pfh_bins)
  int   color_pfh_bins_ = 11;
  std::unique_ptr<ColorPFH> color_pfh_;
};

}  // namespace gs3d_matcher

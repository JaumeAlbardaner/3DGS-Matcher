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
#include <iostream>
#include <memory>
#include <vector>
#include <kiss_matcher/FasterPFH.hpp>
#include "3dgs_matcher/DeltaEPFH.hpp"
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
 * @brief Color-enhanced FasterPFH that wraps KISS-Matcher's FasterPFH
 * 
 * Uses composition instead of inheritance to avoid Eigen destructor issues.
 * Computes 33D FPFH using internal FasterPFH, then appends weighted 4D
 * RGB+alpha (R, G, B, opacity) to create 37-dimensional descriptors.
 */
class ColorFasterPFH {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  explicit ColorFasterPFH(const float normal_radius,
                          const float fpfh_radius,
                          const float thr_linearity,
                          const float color_weight                = 1.0,
                          const std::string criteria              = "L2",
                          const bool use_non_maxima_suppression   = false,
                          const bool use_f1                       = true,
                          const bool use_f2                       = true,
                          const bool use_f3                       = true,
                          const bool use_color                    = true,
                          const bool use_opacity                  = true,
                          const int  num_bins                     = 11,
                          const bool colors_are_lab               = false,
                          const bool use_delta_e_hist             = false,
                          const float delta_e_radius              = 1.0f,
                          const int   delta_e_bins                = 11,
                          const float delta_e_max                 = 50.0f,
                          const bool use_color_pfh                = false,
                          const float color_pfh_radius            = 1.0f,
                          const int   color_pfh_bins              = 11,
                          const float color_pfh_max_de            = 100.0f)
      : color_weight_(color_weight),
        use_f1_(use_f1), use_f2_(use_f2), use_f3_(use_f3),
        use_color_(use_color), use_opacity_(use_opacity),
        num_bins_(num_bins),
        colors_are_lab_(colors_are_lab),
        use_delta_e_hist_(use_delta_e_hist),
        delta_e_bins_(delta_e_bins),
        use_color_pfh_(use_color_pfh),
        color_pfh_bins_(color_pfh_bins) {
    fpfh_ = std::make_unique<kiss_matcher::FasterPFH>(
        normal_radius, fpfh_radius, thr_linearity, criteria, use_non_maxima_suppression);
    if (use_delta_e_hist_)
      delta_e_pfh_ = std::make_unique<DeltaEPFH>(delta_e_radius, delta_e_bins, delta_e_max);
    if (use_color_pfh_)
      color_pfh_ = std::make_unique<ColorPFH>(color_pfh_radius, color_pfh_bins, color_pfh_max_de);
  }

  inline void setInputCloud(const std::vector<Eigen::Vector3f>& points) {
    fpfh_->setInputCloud(points);
    points_ = points;
  }

  template<typename Alloc>
  inline void setColorAttributes(const std::vector<Eigen::Vector4f, Alloc>& colors) {
    if (!points_.empty() && colors.size() != points_.size()) {
      throw std::runtime_error(
          "ColorFasterPFH::setColorAttributes: Number of colors (" + std::to_string(colors.size()) +
          ") does not match number of points (" + std::to_string(points_.size()) + ")");
    }
    colors_.assign(colors.begin(), colors.end());
  }

  void ComputeFeature(std::vector<Eigen::Vector3f>& points,
                      std::vector<Eigen::VectorXf>& descriptors);

  inline float getColorWeight() const { return color_weight_; }
  inline void setColorWeight(float weight) { color_weight_ = weight; }

 private:
  std::unique_ptr<kiss_matcher::FasterPFH> fpfh_;
  std::vector<Eigen::Vector3f> points_;
  std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f>> colors_;
  float color_weight_ = 1.0;

  // Feature selection for ablation study
  bool use_f1_ = true;       ///< Include FPFH sub-histogram 1 (bins [0..num_bins))
  bool use_f2_ = true;       ///< Include FPFH sub-histogram 2 (bins [num_bins..2*num_bins))
  bool use_f3_ = true;       ///< Include FPFH sub-histogram 3 (bins [2*num_bins..3*num_bins))
  bool use_color_ = true;    ///< Include R, G, B color features
  bool use_opacity_ = true;  ///< Include alpha/opacity feature
  int  num_bins_ = 11;       ///< Number of bins per FPFH sub-histogram (KISS-Matcher always produces 11)
  bool colors_are_lab_ = false;  ///< Deprecated: color attributes are treated as RGB+A.
  bool use_delta_e_hist_ = false; ///< Append ΔEab neighborhood histogram
  int  delta_e_bins_ = 11;       ///< Bins in the ΔEab histogram
  std::unique_ptr<DeltaEPFH> delta_e_pfh_;

  bool use_color_pfh_ = false;   ///< Append ColorPFH (3×num_bins hue/lightness/magnitude)
  int  color_pfh_bins_ = 11;     ///< Bins per ColorPFH sub-histogram
  std::unique_ptr<ColorPFH> color_pfh_;

  // Static thread-local to avoid destructor issues with Eigen VectorXf from TBB
  static thread_local std::vector<Eigen::VectorXf>* fpfh_cache_;
};

}  // namespace gs3d_matcher

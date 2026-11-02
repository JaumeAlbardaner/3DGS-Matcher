/**
 * Copyright 2024, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 */
#pragma once

#include "3dgs_matcher/ColorFasterPFH.hpp"
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <memory>
#include <vector>
#include <kiss_matcher/KISSMatcher.hpp>
#include <kiss_matcher/ROBINMatching.hpp>
#include <kiss_matcher/GncSolver.hpp>
#include <kiss_matcher/points/downsampling.hpp>

namespace gs3d_matcher {

// Use KISS-Matcher types
using KeypointPair = kiss_matcher::KeypointPair;
using RegistrationSolution = kiss_matcher::RegistrationSolution;

/**
 * @brief Configuration extending KISSMatcherConfig with color support
 */
struct ColorKISSMatcherConfig : public kiss_matcher::KISSMatcherConfig {
  float color_weight_ = 1.0;  ///< Weight for color features

  // Feature selection flags for ablation study
  bool use_f1_ = true;       ///< Include FPFH sub-histogram 1
  bool use_f2_ = true;       ///< Include FPFH sub-histogram 2
  bool use_f3_ = true;       ///< Include FPFH sub-histogram 3
  bool use_color_ = true;    ///< Include L*, a*, b* color features
  bool use_opacity_ = true;  ///< Include alpha/opacity feature
  int  num_bins_ = 11;       ///< Bins per sub-histogram (KISS-Matcher always produces 11)
  bool  use_centroid_voxel_ = false;  ///< Use centroid averaging (like KISS-Matcher) vs first-hit

  // Joint 5D (spatial + color) voxelization
  bool  use_joint_5d_    = false;  ///< Replace spatial-only downsample with joint (x,y,z,a,b) voxelization
  float voxel_size_ab_   = 5.0f;  ///< LAB ab color voxel size (ΔE units); controls color resolution
  int   min_color_bins_  = 2;     ///< Min distinct color bins per spatial voxel; drops uniform regions

  // Cross-cloud color overlap filter
  bool  use_cross_color_filter_  = false;  ///< Pre-filter: keep only points in color bins shared by both clouds
  float cross_color_min_overlap_ = 0.1f;  ///< Min overlap ratio min(cnt_src,cnt_tgt)/max(cnt_src,cnt_tgt) per bin

  // ΔEab neighborhood histogram descriptor
  bool  use_delta_e_hist_ = false;   ///< Append ΔEab neighborhood histogram to each descriptor
  float delta_e_radius_   = 1.0f;   ///< Spatial search radius for ΔEab neighbors (metres)
  int   delta_e_bins_     = 11;     ///< Histogram bins
  float delta_e_max_      = 50.0f;  ///< Max ΔEab value mapped to last bin

  // ColorPFH descriptor (color analog of FPFH: hue direction / lightness / magnitude)
  bool  use_color_pfh_        = false;   ///< Append ColorPFH (3×bins-D) to each descriptor
  float color_pfh_radius_     = 1.0f;   ///< Spatial search radius for ColorPFH neighbors
  int   color_pfh_bins_       = 11;     ///< Bins per ColorPFH sub-histogram (total dim = 3×bins)
  float color_pfh_max_de_     = 100.0f; ///< Max ΔE (3D LAB) mapped to last ψ-bin

  ColorKISSMatcherConfig(float voxel_size = 0.3,
                        bool use_voxel_sampling = true,
                        bool use_quatro = false,
                        float thr_linearity = 1.0,
                        int num_max_corr = 5000,
                        float color_weight = 1.0,
                        bool use_f1 = true,
                        bool use_f2 = true,
                        bool use_f3 = true,
                        bool use_color = true,
                        bool use_opacity = true,
                        int  num_bins = 11,
                        bool use_centroid_voxel = false,
                        bool use_joint_5d = false,
                        float voxel_size_ab = 5.0f,
                        int min_color_bins = 2,
                        bool use_cross_color_filter = false,
                        float cross_color_min_overlap = 0.1f,
                        bool use_delta_e_hist = false,
                        float delta_e_radius = 1.0f,
                        int   delta_e_bins = 11,
                        float delta_e_max = 50.0f,
                        bool  use_color_pfh = false,
                        float color_pfh_radius = 1.0f,
                        int   color_pfh_bins = 11,
                        float color_pfh_max_de = 100.0f)
      : KISSMatcherConfig(voxel_size, use_voxel_sampling, use_quatro, thr_linearity, num_max_corr),
        color_weight_(color_weight),
        use_f1_(use_f1), use_f2_(use_f2), use_f3_(use_f3),
        use_color_(use_color), use_opacity_(use_opacity),
        num_bins_(num_bins),
        use_centroid_voxel_(use_centroid_voxel),
        use_joint_5d_(use_joint_5d),
        voxel_size_ab_(voxel_size_ab),
        min_color_bins_(min_color_bins),
        use_cross_color_filter_(use_cross_color_filter),
        cross_color_min_overlap_(cross_color_min_overlap),
        use_delta_e_hist_(use_delta_e_hist),
        delta_e_radius_(delta_e_radius),
        delta_e_bins_(delta_e_bins),
        delta_e_max_(delta_e_max),
        use_color_pfh_(use_color_pfh),
        color_pfh_radius_(color_pfh_radius),
        color_pfh_bins_(color_pfh_bins),
        color_pfh_max_de_(color_pfh_max_de) {}
};

/**
 * @brief Color-enhanced KISS-Matcher using full KISS-Matcher pipeline
 */
class ColorKISSMatcher {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  explicit ColorKISSMatcher(const float& voxel_size, const float& color_weight = 1.0);
  explicit ColorKISSMatcher(const ColorKISSMatcherConfig& config);
  ~ColorKISSMatcher();

  void reset();
  void resetSolver();

  KeypointPair match(const std::vector<Eigen::Vector3f>& src,
                    const std::vector<Eigen::Vector3f>& tgt,
                    const std::vector<Eigen::Vector4f>& src_colors,
                    const std::vector<Eigen::Vector4f>& tgt_colors);

  RegistrationSolution estimate(const std::vector<Eigen::Vector3f>& src,
                               const std::vector<Eigen::Vector3f>& tgt,
                               const std::vector<Eigen::Vector4f>& src_colors,
                               const std::vector<Eigen::Vector4f>& tgt_colors);

  void clear();
  void print();
  inline size_t getNumInliers() const { return num_inliers_; }
  inline const std::vector<Eigen::Vector3f>& getSrcProcessed() const { return src_processed_; }
  inline const std::vector<Eigen::Vector3f>& getTgtProcessed() const { return tgt_processed_; }
  inline const std::vector<Eigen::Vector4f>& getSrcProcessedColors() const { return src_colors_processed_; }
  inline const std::vector<Eigen::Vector4f>& getTgtProcessedColors() const { return tgt_colors_processed_; }

 private:
   // Helper to downsample points with colors (first-hit: keeps first point per voxel)
   static void downsampleWithColors(const std::vector<Eigen::Vector3f>& points_in,
                                     const std::vector<Eigen::Vector4f>& colors_in,
                                     std::vector<Eigen::Vector3f>& points_out,
                                     std::vector<Eigen::Vector4f>& colors_out,
                                     double voxel_size);

   // Centroid downsampling: averages both positions and colors per voxel
   static void downsampleWithColorsCentroid(const std::vector<Eigen::Vector3f>& points_in,
                                             const std::vector<Eigen::Vector4f>& colors_in,
                                             std::vector<Eigen::Vector3f>& points_out,
                                             std::vector<Eigen::Vector4f>& colors_out,
                                             double voxel_size);

   // Joint 5D voxelization: one real point per (spatial, color-ab) cell.
   // Spatial voxels with fewer than min_color_bins distinct color cells are dropped
   // (uniform regions like grass/sky). No color averaging — every output point is real.
   static void downsampleJoint5D(const std::vector<Eigen::Vector3f>& points_in,
                                  const std::vector<Eigen::Vector4f>& colors_in,
                                  std::vector<Eigen::Vector3f>& points_out,
                                  std::vector<Eigen::Vector4f>& colors_out,
                                  double voxel_size_xyz,
                                  float  voxel_size_ab,
                                  int    min_color_bins);

   // Cross-cloud color overlap filter: retains only points whose LAB-ab color bin
   // is present in BOTH src and tgt with overlap ratio >= min_overlap_ratio.
   // Applied as a pre-step before any downsampling; uses voxel_size_ab for binning.
   static void filterByColorOverlap(
       const std::vector<Eigen::Vector3f>& src_pts_in,
       const std::vector<Eigen::Vector4f>& src_col_in,
       const std::vector<Eigen::Vector3f>& tgt_pts_in,
       const std::vector<Eigen::Vector4f>& tgt_col_in,
       std::vector<Eigen::Vector3f>& src_pts_out,
       std::vector<Eigen::Vector4f>& src_col_out,
       std::vector<Eigen::Vector3f>& tgt_pts_out,
       std::vector<Eigen::Vector4f>& tgt_col_out,
       float voxel_size_ab,
       float min_overlap_ratio);
  
  ColorKISSMatcherConfig config_;
  
  std::unique_ptr<ColorFasterPFH> color_fpfh_;
  std::unique_ptr<kiss_matcher::ROBINMatching> robin_matching_;
  std::unique_ptr<kiss_matcher::RobustRegistrationSolver> solver_;
  
  std::vector<Eigen::Vector3f> src_processed_;
  std::vector<Eigen::Vector3f> tgt_processed_;
  std::vector<Eigen::Vector4f> src_colors_processed_;
  std::vector<Eigen::Vector4f> tgt_colors_processed_;
  std::vector<Eigen::Vector3f> src_keypoints_;
  std::vector<Eigen::Vector3f> tgt_keypoints_;
  // CRITICAL: Use raw pointers to prevent destructor calls on Eigen::VectorXf from TBB
  std::vector<Eigen::VectorXf>* src_descriptors_;
  std::vector<Eigen::VectorXf>* tgt_descriptors_;

  size_t num_inliers_ = 0;
  double processing_time_ = -1.0;
  double extraction_time_ = -1.0;
  double matching_time_ = -1.0;
  double solver_time_ = -1.0;
};

}  // namespace gs3d_matcher

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

  // Photometric descriptor flags
  bool use_color_   = true;    ///< Include raw R, G, B
  bool use_opacity_ = true;    ///< Include alpha/opacity as a fourth raw channel

  // Keypoint downsampling
  bool  use_centroid_voxel_ = false;  ///< Average position+color per voxel (vs first-hit)
  bool  use_joint_5d_   = false;      ///< Joint (x,y,z,a*,b*) color-aware voxelization
  float voxel_size_ab_  = 5.0f;       ///< LAB ab color voxel size (ΔE units)
  int   min_color_bins_ = 2;          ///< Min distinct color bins per spatial voxel

  // ColorPFH descriptor (color analog of FPFH: hue direction / lightness / magnitude)
  bool  use_color_pfh_    = false;   ///< Append ColorPFH (3×bins-D) to each descriptor
  float color_pfh_radius_ = 1.0f;    ///< Spatial search radius for ColorPFH neighbors
  int   color_pfh_bins_   = 11;      ///< Bins per ColorPFH sub-histogram (total dim = 3×bins)
  float color_pfh_max_de_ = 100.0f;  ///< Max ΔE (3D LAB) mapped to last ψ-bin

  ColorKISSMatcherConfig(float voxel_size = 0.3,
                        bool use_voxel_sampling = true,
                        bool use_quatro = false,
                        float thr_linearity = 1.0,
                        int num_max_corr = 5000,
                        float color_weight = 1.0,
                        bool use_color = true,
                        bool use_opacity = true,
                        bool use_centroid_voxel = false,
                        bool use_joint_5d = false,
                        float voxel_size_ab = 5.0f,
                        int min_color_bins = 2,
                        bool  use_color_pfh = false,
                        float color_pfh_radius = 1.0f,
                        int   color_pfh_bins = 11,
                        float color_pfh_max_de = 100.0f)
      : KISSMatcherConfig(voxel_size, use_voxel_sampling, use_quatro, thr_linearity, num_max_corr),
        color_weight_(color_weight),
        use_color_(use_color), use_opacity_(use_opacity),
        use_centroid_voxel_(use_centroid_voxel),
        use_joint_5d_(use_joint_5d),
        voxel_size_ab_(voxel_size_ab),
        min_color_bins_(min_color_bins),
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

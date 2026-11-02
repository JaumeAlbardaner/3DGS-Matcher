#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <3dgs_matcher/ColorFasterPFH.hpp>

struct GaussianSplat {
  Eigen::Vector3f position;
  Eigen::Vector4f color_rgba;
};

static bool loadSplatFile(const std::string& filename, std::vector<GaussianSplat>& splats) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return false;
  }

  file.seekg(0, std::ios::end);
  const size_t file_size = static_cast<size_t>(file.tellg());
  file.seekg(0, std::ios::beg);

  if (file_size % 32 != 0) {
    std::cerr << "Unsupported .splat format (expected 32 bytes per splat)." << std::endl;
    return false;
  }

  const size_t num_splats = file_size / 32;
  splats.clear();
  splats.reserve(num_splats);

  for (size_t i = 0; i < num_splats; ++i) {
    GaussianSplat splat;

    float data[6];
    file.read(reinterpret_cast<char*>(data), sizeof(float) * 6);
    splat.position = Eigen::Vector3f(data[0], data[1], data[2]);

    uint8_t rgba[4];
    file.read(reinterpret_cast<char*>(rgba), 4);
    uint8_t rotation[4];
    file.read(reinterpret_cast<char*>(rotation), 4);

    splat.color_rgba = Eigen::Vector4f(
        rgba[0] / 255.0f,
        rgba[1] / 255.0f,
        rgba[2] / 255.0f,
        rgba[3] / 255.0f);
    splats.push_back(splat);
  }

  return true;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " <reference.splat> [min_de_std=5.0] [search_radius=1.5] [min_neighbors=3] [opacity_threshold=0.0]"
              << std::endl;
    return 1;
  }

  const std::string splat_file = argv[1];
  const float min_de_std = (argc > 2) ? std::stof(argv[2]) : 5.0f;
  const float search_radius = (argc > 3) ? std::stof(argv[3]) : 1.5f;
  const int min_neighbors = (argc > 4) ? std::stoi(argv[4]) : 3;
  const float opacity_threshold = (argc > 5) ? std::stof(argv[5]) : 0.0f;

  std::vector<GaussianSplat> splats;
  if (!loadSplatFile(splat_file, splats)) return 1;

  std::vector<Eigen::Vector3f> points;
  std::vector<Eigen::Vector4f> ref_colors;
  points.reserve(splats.size());
  ref_colors.reserve(splats.size());
  for (const auto& s : splats) {
    if (s.color_rgba[3] >= opacity_threshold) {
      points.push_back(s.position);
      ref_colors.push_back(s.color_rgba);
    }
  }

  if (points.empty()) {
    std::cerr << "No points left after opacity filtering." << std::endl;
    return 1;
  }

  std::vector<Eigen::Vector3f> lab_colors(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    lab_colors[i] = gs3d_matcher::rgbToLab(ref_colors[i][0], ref_colors[i][1], ref_colors[i][2]);
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr search_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  search_cloud->reserve(points.size());
  for (const auto& p : points) {
    pcl::PointXYZ xyz;
    xyz.x = p.x();
    xyz.y = p.y();
    xyz.z = p.z();
    search_cloud->push_back(xyz);
  }

  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(search_cloud);

  const float sq_min_de_std = min_de_std * min_de_std;
  std::vector<size_t> selected_indices;
  selected_indices.reserve(points.size() / 4);

  std::vector<int> nn_indices;
  std::vector<float> nn_sq_dists;
  for (size_t i = 0; i < points.size(); ++i) {
    nn_indices.clear();
    nn_sq_dists.clear();
    kdtree.radiusSearch((*search_cloud)[i], search_radius, nn_indices, nn_sq_dists);

    if (nn_indices.size() < static_cast<size_t>(std::max(min_neighbors, 1))) {
      continue;
    }

    const Eigen::Vector3f& p_lab = lab_colors[i];
    float sum_sq = 0.0f;
    float sum_sq2 = 0.0f;
    int n_nb = 0;

    for (const int idx : nn_indices) {
      if (idx == static_cast<int>(i)) continue;
      const Eigen::Vector3f dlab = lab_colors[static_cast<size_t>(idx)] - p_lab;
      const float sq_de = dlab.y() * dlab.y() + dlab.z() * dlab.z();
      sum_sq += sq_de;
      sum_sq2 += sq_de * sq_de;
      ++n_nb;
    }

    if (n_nb > 1) {
      const float inv_n = 1.0f / static_cast<float>(n_nb);
      const float mean_sq = sum_sq * inv_n;
      const float var_sq = sum_sq2 * inv_n - mean_sq * mean_sq;
      if (var_sq < sq_min_de_std * sq_min_de_std) {
        continue;
      }
    }

    selected_indices.push_back(i);
  }

  std::cout << "Loaded points: " << points.size() << std::endl;
  std::cout << "Filter: var(Delta_ab^2) >= min_de_std^4" << std::endl;
  std::cout << "min_de_std=" << min_de_std << ", search_radius=" << search_radius
            << ", min_neighbors=" << min_neighbors << std::endl;
  std::cout << "Selected points: " << selected_indices.size() << std::endl;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr full_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr selected_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  full_cloud->reserve(points.size());
  selected_cloud->reserve(selected_indices.size());

  for (size_t i = 0; i < points.size(); ++i) {
    pcl::PointXYZRGB p;
    p.x = points[i].x();
    p.y = points[i].y();
    p.z = points[i].z();
    p.r = static_cast<uint8_t>(std::round(ref_colors[i][0] * 255.0f));
    p.g = static_cast<uint8_t>(std::round(ref_colors[i][1] * 255.0f));
    p.b = static_cast<uint8_t>(std::round(ref_colors[i][2] * 255.0f));
    full_cloud->push_back(p);
  }

  for (const size_t idx : selected_indices) {
    pcl::PointXYZRGB p;
    p.x = points[idx].x();
    p.y = points[idx].y();
    p.z = points[idx].z();
    p.r = static_cast<uint8_t>(std::round(ref_colors[idx][0] * 255.0f));
    p.g = static_cast<uint8_t>(std::round(ref_colors[idx][1] * 255.0f));
    p.b = static_cast<uint8_t>(std::round(ref_colors[idx][2] * 255.0f));
    selected_cloud->push_back(p);
  }

  pcl::visualization::PCLVisualizer::Ptr viewer(
      new pcl::visualization::PCLVisualizer("DeltaE Splat Visualizer"));
  viewer->setBackgroundColor(0.08, 0.08, 0.08);
  viewer->addCoordinateSystem(1.0);
  viewer->initCameraParameters();

  viewer->addPointCloud<pcl::PointXYZRGB>(full_cloud, "full_cloud");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.0, "full_cloud");

  viewer->addPointCloud<pcl::PointXYZRGB>(selected_cloud, "selected_points");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7.0, "selected_points");

  viewer->addText("Full cloud + points selected by ablation DeltaE std filter", 10, 10, 14, 1.0, 1.0, 1.0, "title");
  viewer->addText("Press 'q' to quit", 10, 30, 14, 1.0, 1.0, 1.0, "hint");

  while (!viewer->wasStopped()) {
    viewer->spin();
  }

  return 0;
}

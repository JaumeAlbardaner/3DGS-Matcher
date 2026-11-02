// Prints the random perturbation benchmark_submap applies for a given seed,
// so post-hoc analysis can undo it. MUST stay bit-identical to the RNG code in
// examples/benchmark_submap.cc (same distributions, same call order, same
// libstdc++), otherwise the recovered T_pert diverges from the benchmark's.
//
// Usage: print_perturbation <seed> <max_rot_deg> <max_trans_m>
// Output: one line, 12 floats: R row-major (9) then t (3).
//
// Build: g++ -O2 -std=c++17 -I/usr/include/eigen3 -o build/print_perturbation \
//            scripts/print_perturbation.cc

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

Eigen::Matrix3f randomRotation(std::mt19937& gen, float max_angle_deg) {
  std::uniform_real_distribution<float> u01(0.0f, 1.0f);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  Eigen::Vector3f axis(nd(gen), nd(gen), nd(gen));
  if (axis.norm() < 1e-8f) axis = Eigen::Vector3f::UnitX();
  axis.normalize();

  float angle = u01(gen) * max_angle_deg * static_cast<float>(M_PI) / 180.0f;
  if (u01(gen) < 0.5f) angle = -angle;

  return Eigen::AngleAxisf(angle, axis).toRotationMatrix();
}

Eigen::Vector3f randomTranslation(std::mt19937& gen, float max_t) {
  std::uniform_real_distribution<float> ut(-max_t, max_t);
  return Eigen::Vector3f(ut(gen), ut(gen), ut(gen));
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "Usage: %s <seed> <max_rot_deg> <max_trans_m>\n", argv[0]);
    return 1;
  }
  const uint32_t seed = static_cast<uint32_t>(std::atoi(argv[1]));
  const float max_rot = std::atof(argv[2]);
  const float max_trans = std::atof(argv[3]);

  std::mt19937 gen(seed);
  const Eigen::Matrix3f R = randomRotation(gen, max_rot);
  const Eigen::Vector3f t = randomTranslation(gen, max_trans);

  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) std::printf("%.9g ", R(r, c));
  std::printf("%.9g %.9g %.9g\n", t.x(), t.y(), t.z());
  return 0;
}

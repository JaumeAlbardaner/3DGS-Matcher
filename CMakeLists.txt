cmake_minimum_required(VERSION 3.10)
project(3DGS-Matcher VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)

# Find ROBIN (required by KISS-Matcher)
find_package(robin REQUIRED)

# Find KISS-Matcher (required for base functionality)
# Try standard locations first
find_package(kiss_matcher QUIET)
if(NOT kiss_matcher_FOUND)
    # Try common local paths produced by scripts/bootstrap.sh
    set(_kiss_matcher_candidates
        "${CMAKE_CURRENT_SOURCE_DIR}/../KISS-Matcher/build/install/lib/cmake/kiss_matcher"
        "${CMAKE_CURRENT_SOURCE_DIR}/../KISS-Matcher/build/install/lib64/cmake/kiss_matcher"
        "${CMAKE_CURRENT_SOURCE_DIR}/../KISS-Matcher/build"
    )
    foreach(_kiss_dir IN LISTS _kiss_matcher_candidates)
        if(EXISTS "${_kiss_dir}")
            set(kiss_matcher_DIR "${_kiss_dir}")
            find_package(kiss_matcher QUIET)
            if(kiss_matcher_FOUND)
                break()
            endif()
        endif()
    endforeach()
endif()
if(NOT kiss_matcher_FOUND)
    message(FATAL_ERROR "Could not find kiss_matcher. Run scripts/bootstrap.sh to build/install local dependencies.")
endif()

# Optional: MPI for PCL/VTK compatibility
find_package(MPI QUIET)

# Option to build examples
option(BUILD_EXAMPLES "Build example executables" ON)

# 3DGS-Matcher library
add_library(3dgs_matcher
    src/ColorFasterPFH.cpp
    src/ColorKISSMatcher.cpp
)

target_include_directories(3dgs_matcher
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(3dgs_matcher
    PUBLIC
        Eigen3::Eigen
        TBB::tbb
        kiss_matcher::kiss_matcher_core
)

# Set compile options
# -mno-avx: nanoflann's KDTree build (divideTree/computeMinMax) is auto-vectorized
# with AVX at -O3/-march=native, generating vmovdqa (32-byte aligned load) for
# vind[], which std::vector allocates with 16-byte alignment. Use SSE4.2 instead.
target_compile_options(3dgs_matcher PRIVATE
    -Wall
    -Wextra
    -march=native
    -mno-avx
    -O3
)

# Install targets
install(TARGETS 3dgs_matcher
    EXPORT 3dgs_matcher-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

# Install documentation
install(DIRECTORY docs/
    DESTINATION share/3dgs_matcher/docs
    FILES_MATCHING PATTERN "*.md"
)

# Export targets
install(EXPORT 3dgs_matcher-targets
    FILE 3dgs_matcher-targets.cmake
    NAMESPACE 3dgs_matcher::
    DESTINATION lib/cmake/3dgs_matcher
)

# Create config files
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/3dgs_matcher-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/3dgs_matcher-config.cmake
    INSTALL_DESTINATION lib/cmake/3dgs_matcher
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/3dgs_matcher-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/3dgs_matcher-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/3dgs_matcher-config-version.cmake
    DESTINATION lib/cmake/3dgs_matcher
)

# Examples
if(BUILD_EXAMPLES)
    # Single-process ablation runner (no PCL dependency)
    add_executable(single_ablation examples/single_ablation.cc)
    target_link_libraries(single_ablation
        3dgs_matcher
    )

    # Replays benchmark_submap's random perturbation for a seed (used by the
    # scoring scripts). Must share benchmark_submap's RNG code and toolchain.
    add_executable(print_perturbation scripts/print_perturbation.cc)
    target_link_libraries(print_perturbation Eigen3::Eigen)

    # Two-file .splat alignment runner (optional PCL visualization via --viz)
    add_executable(align_two_splats examples/align_two_splats.cc)
    target_link_libraries(align_two_splats
        3dgs_matcher
    )

    # Find MPI before PCL to avoid VTK linking errors
    find_package(MPI)
    
    # Find PCL for examples (suppress policy warnings)
    set(CMAKE_POLICY_DEFAULT_CMP0144 NEW)
    set(CMAKE_POLICY_DEFAULT_CMP0167 NEW)
    find_package(PCL QUIET COMPONENTS common io visualization registration)
    
    if(PCL_FOUND)
        target_link_libraries(align_two_splats
            ${PCL_LIBRARIES}
        )
        target_include_directories(align_two_splats PRIVATE ${PCL_INCLUDE_DIRS})
        target_compile_definitions(align_two_splats PRIVATE COLOR_MATCHER_USE_PCL_VIZ)

        add_executable(example_color_kiss_matcher examples/example_color_kiss_matcher.cc)
        target_link_libraries(example_color_kiss_matcher
            3dgs_matcher
            ${PCL_LIBRARIES}
        )
        target_include_directories(example_color_kiss_matcher PRIVATE ${PCL_INCLUDE_DIRS})

        add_executable(example_delta_e_splat_visualizer examples/example_delta_e_splat_visualizer.cc)
        target_link_libraries(example_delta_e_splat_visualizer
            3dgs_matcher
            ${PCL_LIBRARIES}
        )
        target_include_directories(example_delta_e_splat_visualizer PRIVATE ${PCL_INCLUDE_DIRS})
        
        add_executable(test_color_splat examples/test_color_splat.cc)
        target_link_libraries(test_color_splat
            3dgs_matcher
            ${PCL_LIBRARIES}
        )
        target_include_directories(test_color_splat PRIVATE ${PCL_INCLUDE_DIRS})
        
        add_executable(benchmark_color_vs_vanilla examples/benchmark_color_vs_vanilla.cc)
        target_link_libraries(benchmark_color_vs_vanilla
            3dgs_matcher
            ${PCL_LIBRARIES}
        )
        target_include_directories(benchmark_color_vs_vanilla PRIVATE ${PCL_INCLUDE_DIRS})
        
        # Color ICP example requires small_gicp header-only library
        # TODO: Add small_gicp as a git submodule or copy from KISS-Matcher
        # add_executable(example_color_icp examples/example_color_icp.cc)
        # target_link_libraries(example_color_icp
        #     3dgs_matcher
        # )

        message(STATUS "Building examples (PCL found)")
    else()
        message(STATUS "PCL not found - examples will not be built")
    endif()

    # Submap benchmark: vanilla KISS / 3DGS color+colorPFH / GICP / Color-GICP
    # Uses small_gicp (header-only) for ICP — no PCL required.
    # Location may be overridden with -DSMALL_GICP_DIR=/path/to/small_gicp or the
    # SMALL_GICP_ROOT environment variable; otherwise a sibling checkout and the
    # standard system prefixes are searched.
    set(SMALL_GICP_DIR "" CACHE PATH
        "Root of a small_gicp checkout (the directory containing include/small_gicp)")
    find_path(SMALL_GICP_INCLUDE_DIR
        NAMES small_gicp/registration/registration_helper.hpp
        HINTS
            ${SMALL_GICP_DIR}/include
            $ENV{SMALL_GICP_ROOT}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../small_gicp/include
        PATHS
            /usr/local/include
            /usr/include
    )
    if(SMALL_GICP_INCLUDE_DIR)
        # small_gicp::align helper needs its .cpp compiled in (not header-only)
        get_filename_component(SMALL_GICP_SRC_DIR "${SMALL_GICP_INCLUDE_DIR}/../src/small_gicp" ABSOLUTE)
        add_executable(benchmark_submap
            examples/benchmark_submap.cc
            ${SMALL_GICP_SRC_DIR}/registration/registration_helper.cpp
        )
        target_link_libraries(benchmark_submap 3dgs_matcher TBB::tbb)
        target_include_directories(benchmark_submap PRIVATE ${SMALL_GICP_INCLUDE_DIR})
        message(STATUS "benchmark_submap: small_gicp found at ${SMALL_GICP_INCLUDE_DIR}")

        # Budget-sensitivity control for Table II refinement rows: same
        # small_gicp::align path, warm-started from stored coarse poses.
        add_executable(gicp_budget_control
            examples/gicp_budget_control.cc
            ${SMALL_GICP_SRC_DIR}/registration/registration_helper.cpp
        )
        target_link_libraries(gicp_budget_control 3dgs_matcher TBB::tbb)
        target_include_directories(gicp_budget_control PRIVATE ${SMALL_GICP_INCLUDE_DIR})

        # Give align_two_splats the same fine-registration capability (GICP / colour-GICP)
        if(TARGET align_two_splats)
            target_sources(align_two_splats PRIVATE
                ${SMALL_GICP_SRC_DIR}/registration/registration_helper.cpp)
            target_link_libraries(align_two_splats TBB::tbb)
            target_include_directories(align_two_splats PRIVATE ${SMALL_GICP_INCLUDE_DIR})
            target_compile_definitions(align_two_splats PRIVATE COLOR_MATCHER_HAS_GICP)
            message(STATUS "align_two_splats: fine registration (GICP/colour-GICP) enabled")
        endif()
    else()
        message(WARNING "benchmark_submap: small_gicp not found — skipping")
    endif()
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "==================================")
message(STATUS "  3DGS-Matcher Configuration")
message(STATUS "==================================")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build examples: ${BUILD_EXAMPLES}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "==================================")
message(STATUS "")

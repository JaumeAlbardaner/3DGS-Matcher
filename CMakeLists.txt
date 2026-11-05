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

# Matcher CLI tool
if(BUILD_EXAMPLES)
    # align_two_splats: the standalone 3DGS matcher (coarse 3DGS-Matcher + optional
    # GICP / color-GICP refinement). Optional PCL visualization via --viz.
    add_executable(align_two_splats examples/align_two_splats.cc)
    target_link_libraries(align_two_splats 3dgs_matcher)

    # Optional PCL visualization (--viz). Find MPI first to avoid VTK link errors.
    find_package(MPI)
    set(CMAKE_POLICY_DEFAULT_CMP0144 NEW)
    set(CMAKE_POLICY_DEFAULT_CMP0167 NEW)
    find_package(PCL QUIET COMPONENTS common io visualization registration)
    if(PCL_FOUND)
        target_link_libraries(align_two_splats ${PCL_LIBRARIES})
        target_include_directories(align_two_splats PRIVATE ${PCL_INCLUDE_DIRS})
        target_compile_definitions(align_two_splats PRIVATE COLOR_MATCHER_USE_PCL_VIZ)
        message(STATUS "align_two_splats: PCL visualization (--viz) enabled")
    else()
        message(STATUS "align_two_splats: PCL not found — building without --viz")
    endif()

    # Optional fine registration (GICP / color-GICP) via small_gicp (header-only + one .cpp).
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
        get_filename_component(SMALL_GICP_SRC_DIR "${SMALL_GICP_INCLUDE_DIR}/../src/small_gicp" ABSOLUTE)
        target_sources(align_two_splats PRIVATE
            ${SMALL_GICP_SRC_DIR}/registration/registration_helper.cpp)
        target_link_libraries(align_two_splats TBB::tbb)
        target_include_directories(align_two_splats PRIVATE ${SMALL_GICP_INCLUDE_DIR})
        target_compile_definitions(align_two_splats PRIVATE COLOR_MATCHER_HAS_GICP)
        message(STATUS "align_two_splats: fine registration (GICP/color-GICP) enabled")
    else()
        message(WARNING "align_two_splats: small_gicp not found — coarse only (--refine disabled)")
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

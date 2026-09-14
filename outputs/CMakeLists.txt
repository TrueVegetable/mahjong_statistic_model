cmake_minimum_required(VERSION 3.24)
project(hand_efficiency_sim LANGUAGES CXX)

option(HAND_EFFICIENCY_USE_TILE_MASKS "Use GPL necessary/discard masks for one-step evaluation" ON)
option(HAND_EFFICIENCY_USE_NYANTEN "Use the MIT Nyanten shanten backend when tile masks are disabled" ON)

add_executable(hand_efficiency_sim_cpp hand_efficiency_sim.cpp)
target_compile_features(hand_efficiency_sim_cpp PRIVATE cxx_std_17)

if(HAND_EFFICIENCY_USE_TILE_MASKS)
  include(FetchContent)
  set(ENABLE_NYANTEN ON CACHE BOOL "Use compressed necessary-tile lookup tables" FORCE)
  set(FIX_RANDOM_SEED OFF CACHE BOOL "Do not alter dependency examples" FORCE)
  FetchContent_Declare(
    necessary_tiles
    GIT_REPOSITORY https://github.com/tomohxx/necessary-and-unnecessary-tiles.git
    GIT_TAG e5cd6e124218808da07d76dc852bdcc4877de791
    GIT_SHALLOW FALSE
    PATCH_COMMAND
      git apply --ignore-whitespace
      "${CMAKE_CURRENT_SOURCE_DIR}/patches/necessary_tiles_sequential_policy.patch"
  )
  FetchContent_MakeAvailable(necessary_tiles)
  target_link_libraries(hand_efficiency_sim_cpp PRIVATE calsht_dw)
  target_compile_definitions(hand_efficiency_sim_cpp PRIVATE HAND_EFFICIENCY_USE_TILE_MASKS=1)
  add_dependencies(hand_efficiency_sim_cpp index_files)
elseif(HAND_EFFICIENCY_USE_NYANTEN)
  include(FetchContent)
  FetchContent_Declare(
    nyanten
    GIT_REPOSITORY https://github.com/Cryolite/nyanten.git
    GIT_TAG 581aa72ce53e0ffeb0ef8d1ed14df2521ca5fb9b
    GIT_SHALLOW FALSE
    # Nyanten is header-only. Avoid importing its developer/test build flags.
    SOURCE_SUBDIR __header_only_dependency
  )
  FetchContent_MakeAvailable(nyanten)
  target_include_directories(hand_efficiency_sim_cpp PRIVATE "${nyanten_SOURCE_DIR}")
  target_compile_definitions(hand_efficiency_sim_cpp PRIVATE HAND_EFFICIENCY_USE_NYANTEN=1)
endif()

if(MSVC)
  target_compile_options(hand_efficiency_sim_cpp PRIVATE /O2 /W4)
else()
  target_compile_options(hand_efficiency_sim_cpp PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

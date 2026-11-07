cmake_minimum_required(VERSION 3.15)
project(AnkleIKProject CXX)

set(CMAKE_CXX_STANDARD 20)          # needed for operator<=> in kin.h
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenSim REQUIRED)      # found through OpenSim_DIR in ~/.bashrc
message(STATUS "Using OpenSim from: ${OpenSim_DIR}")

add_executable(check_model check_model.cpp)
target_link_libraries(check_model PRIVATE ${OpenSim_LIBRARIES})

add_executable(ankle_ik src/main.cpp)
target_link_libraries(ankle_ik PRIVATE ${OpenSim_LIBRARIES})

add_executable(parseTRC_test src/kin/parseTRC_test.cpp src/kin/parseTRC.cpp)
target_include_directories(parseTRC_test PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src)

add_executable(fk_check src/kin/fk_check.cpp)
target_link_libraries(fk_check PRIVATE ${OpenSim_LIBRARIES})


add_executable(bspline_test src/kin/bspline_test.cpp src/kin/bspline.cpp)
target_include_directories(bspline_test PRIVATE ${CMAKE_SOURCE_DIR}/src/kin)

find_package(Ceres REQUIRED)
add_executable(ceres_hello src/kin/ceres_hello.cpp)
target_link_libraries(ceres_hello PRIVATE Ceres::ceres)


add_executable(toy_ik_test src/kin/toy_ik_test.cpp src/kin/frame_residual.cpp src/kin/bspline.cpp)
target_include_directories(toy_ik_test PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src/kin)
target_link_libraries(toy_ik_test PRIVATE Ceres::ceres)

add_executable(spline_ik src/kin/spline_ik.cpp src/kin/frame_residual.cpp src/kin/bspline.cpp
                         src/kin/opensim_fk.cpp src/kin/parseTRC.cpp src/kin/event_knots.cpp)
target_include_directories(spline_ik PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src/kin)
target_link_libraries(spline_ik PRIVATE Ceres::ceres ${OpenSim_LIBRARIES})

add_executable(event_knots_test src/kin/event_knots_test.cpp src/kin/event_knots.cpp src/kin/bspline.cpp)
target_include_directories(event_knots_test PRIVATE ${CMAKE_SOURCE_DIR}/src/kin)

add_executable(synth_markers src/kin/synth_markers.cpp src/kin/opensim_fk.cpp src/kin/bspline.cpp)
target_include_directories(synth_markers PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src/kin)
target_link_libraries(synth_markers PRIVATE ${OpenSim_LIBRARIES})

add_executable(ik_cost src/kin/ik_cost.cpp src/kin/opensim_fk.cpp src/kin/parseTRC.cpp)
target_include_directories(ik_cost PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src/kin)
target_link_libraries(ik_cost PRIVATE ${OpenSim_LIBRARIES})

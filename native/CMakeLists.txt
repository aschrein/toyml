find_package(pybind11 REQUIRED)
set(embree_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/../bindeps/embree/lib/cmake/embree-4.3.2)
set(tbb_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/../bindeps/tbb/oneapi-tbb-2021.13.0/lib/cmake/tbb)
# message("embree_DIR : ${embree_DIR}")
find_package(embree 4.0 REQUIRED)

function(_add_module module_name)
    pybind11_add_module(${module_name} ${ARGN})
    install(TARGETS ${module_name} LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/${module_name})
endfunction()

_add_module(embree_wrapper embree_wrapper.cpp)

target_include_directories(embree_wrapper PRIVATE ${EMBREE_INCLUDE_DIRS})
target_link_libraries(embree_wrapper PRIVATE embree)
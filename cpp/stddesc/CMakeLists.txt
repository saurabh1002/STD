add_library(stddesc STATIC STDdesc.cpp)
target_include_directories(stddesc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(stddesc PUBLIC ${PCL_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})
set_global_target_properties(stddesc)

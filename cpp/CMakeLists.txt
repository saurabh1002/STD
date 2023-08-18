add_library(stdesc STATIC STDesc.cpp)
target_include_directories(stdesc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(stdesc PUBLIC ${PCL_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})
target_compile_options(stdesc PUBLIC -fPIC)

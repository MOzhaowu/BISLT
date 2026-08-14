find_package(Eigen3 REQUIRED NO_MODULE)

ADD_EXECUTABLE(eg_BIT eg_BIT.cpp)
target_link_libraries(eg_BIT UTILS_LIB INTERFACE_LIB Eigen3::Eigen glfw glog::glog)

ADD_EXECUTABLE(gen_references gen_references.cpp)
target_link_libraries(gen_references UTILS_LIB INTERFACE_LIB Eigen3::Eigen glfw glog::glog)
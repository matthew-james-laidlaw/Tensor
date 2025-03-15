add_executable(unit_tests TestTensor.cpp)
target_link_libraries(unit_tests PUBLIC gtest_main tensor)

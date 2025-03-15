add_executable(unit_tests
    TestDispatch.cpp
    TestTensor.cpp
    TestThreadPool.cpp
)

target_link_libraries(unit_tests
    PUBLIC
        gtest_main
        dispatch
        tensor
)

gtest_discover_tests(unit_tests)

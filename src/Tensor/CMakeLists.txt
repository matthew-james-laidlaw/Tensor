add_library(tensor INTERFACE)
target_include_directories(tensor INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tensor INTERFACE expect)

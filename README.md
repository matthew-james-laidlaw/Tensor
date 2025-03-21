# Tensor

### (See [user guide](./docs/user-guide.md) for detailed usage instructions).

Tensor is a C++ library for N-dimensional tensor operations, including indexing, slicing, and parallel computation. Designed as a template-header library which can be integrated into a project via Git submodules and CMake.

## Integration

To integrate Tensor into your project, follow these steps:

1. **Add Tensor as a Git Submodule:**

    ```sh
    git submodule add https://github.com/matthew-james-laidlaw/Tensor.git extern/Tensor
    git submodule update --init --recursive
    ```

2. **Include Tensor in Your CMake Project:**

    Add the following lines to your CMakeLists.txt:

    ```cmake
    add_subdirectory(extern/Tensor)
    target_link_libraries(your_target PUBLIC tensor)
    ```

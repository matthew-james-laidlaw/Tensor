# Tensor

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

## Usage

### Tensor Class

The `Tensor` class provides N-dimensional tensor operations, including indexing and slicing.

#### Example

```cpp
#include <Tensor.hpp>
#include <iostream>
#include <numeric>

int main()
{
    // construct a 4x4 matrix (2D tensor)
    Tensor<int, 2> matrix({4, 4});

    std::iota(matrix.Data(), matrix.Data() + 16, 0);
    // [  0  1  2  3 ]
    // [  4  5  6  7 ]
    // [  8  9 10 11 ]
    // [ 12 13 14 15 ]

    auto s1 = matrix.Slice(1, Range{1, 3});
    // row index 1 and column indices [1..2]
    // returns [ 5 6 ]

    auto s2 = matrix.Slice(Range{1, 3}, Range{2, 4});
    // row indices [1..2] and column indices [2..3]
    // returns [  6  7 ]
    //         [ 10 11 ]

    auto n1 = s4({0, 0});
    // direct index at (0, 0)
    // returns 0

    // this library supports rough printing of tensors and views
    std::cout << matrix << std::endl;
    std::cout << s1 << std::endl << std::endl;
    std::cout << s2 << std::endl << std::endl;
    std::cout << n1 << std::endl << std::endl;

    return 0;
}
```

### ThreadPool and Dispatch Functions

The library includes a `ThreadPool` class and `Dispatch` functions for parallel computation.

#### Example

```cpp
#include <Dispatch.hpp>
#include <vector>
#include <iostream>

int main()
{
    size_t height = 512;
    size_t width = 512;

    // use 2D vector as stand-in for matrix in this case
    // initialize as 
    // m1 [ 1 1 1 ... 1 1 1 ]
    // m2 [ 2 2 2 ... 2 2 2 ]
    // m3 [ 0 0 0 ... 0 0 0 ]
    std::vector<std::vector<int>> m1(height, std::vector<int>(width, 1));
    std::vector<std::vector<int>> m2(height, std::vector<int>(width, 2));
    std::vector<std::vector<int>> m3(height, std::vector<int>(width, 0));

    // provide a per-element kernel (functor that takes a (y, x) position)
    // DispatchElement gives all threads some number of these indices to execute the kernel on
    DispatchElement(height, width, [&](size_t y, size_t x) {
        // vector add m3 = m1 + m2
        m3[y][x] = m1[y][x] + m2[y][x];
    });

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            std::cout << m3[y][x] << " ";
        }
        std::cout << std::endl;
    }

    // m3 [ 3 3 3 ... 3 3 3 ]

    return 0;
}
```

## Main Components

- **Tensor Class:** Supports N-dimensional indexing and slicing.
- **ThreadPool:** Manages a pool of threads for parallel execution.
- **Dispatch Functions:** Provides parallel execution of tasks over tensor elements or rows.

## Todo

- Tensor initializer list
- Tensor arithmetic + broadcasting
- Expression templates
- Pretty printing

For more details, refer to the source code and examples provided in the repository.
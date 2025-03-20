#include <Tensor.hpp>
#include <iostream>
#include <numeric>

int main()
{
    // construct 4x4 matrix (2D tensor)
    Tensor<int, 2> matrix({4, 4}, 42);

    // fill with sequential values
    std::iota(matrix.Data(), matrix.Data() + 16, 0);

    // print the matrix
    std::cout << "Matrix:" << std::endl << matrix << std::endl;

    // index into the matrix
    std::cout << "Element at (2, 3): " << matrix({2, 3}) << std::endl << std::endl;

    // slice row from matrix
    auto rowSlice = matrix.Slice(1, Range{1, 3});
    std::cout << "Row slice (1, [1, 3)): " << rowSlice << std::endl;

    // slice block from matrix
    auto blockSlice = matrix.Slice(Range{1, 3}, Range{2, 4});
    std::cout << "Block slice ([1, 3), [2, 4)):\n" << blockSlice << std::endl;

    return 0;
}

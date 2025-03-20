#include <Tensor.hpp>
#include <iostream>
#include <numeric>

int main()
{
    // Construct a 4x4 matrix (2D tensor)
    Tensor<int, 2> matrix({4, 4});

    // Fill the matrix with sequential values
    std::iota(matrix.Data(), matrix.Data() + 16, 0);

    // Print the matrix
    std::cout << "Matrix:\n" << matrix << "\n";

    // Indexing
    std::cout << "Element at (2, 3): " << matrix({2, 3}) << "\n";

    // Slicing
    auto rowSlice = matrix.Slice(1, Range{1, 3});
    std::cout << "Row slice (1, [1, 3)): " << rowSlice << "\n";

    auto blockSlice = matrix.Slice(Range{1, 3}, Range{2, 4});
    std::cout << "Block slice ([1, 3), [2, 4)):\n" << blockSlice << "\n";

    return 0;
}

#include <Tensor.hpp>
#include <iostream>
#include <numeric>

int main()
{
    // construct 4x4 matrix and fill with sequential values
    Tensor<int, 2> matrix({4, 4});
    std::iota(matrix.Data(), matrix.Data() + 16, 1);

    std::cout << matrix << std::endl;
    // {
    //     Shape = [4, 4]
    //     Data = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    // }

    // index into the matrix via operator(), passing indices as initializer list
    auto element = matrix({2, 3});

    std::cout << "Element at (2, 3): " << element << std::endl
                << std::endl;
    // Element at (2, 3): 12

    // slicing
    // slicing accepts either an index or range for each dimension
    // for example, matrix.Slice(1, {0, 4}) would return the second row (0 indexed) of the matrix
    // as the first dimension is sliced at index 1, and the second dimension is sliced from 0 to 4 (exclusive), i.e. returning the whole row
    // indexing a dimension returns a view with one less dimension than the original tensor
    // slicing a dimension with a range returns a view with the same number of dimensions as the original tensor, but with a constrained range of indices

    // slice a row from the matrix, returning a non-owning view
    auto rowSlice = matrix.Slice(1, Range{0, 4});

    std::cout << "Row slice (1, [0, 4)):" << std::endl
                << rowSlice << std::endl;
    // Row slice (1, [0, 4)):
    // {
    //     Shape = [4]
    //     Data = [5, 6, 7, 8]
    // }

    // slice a column from the matrix
    auto colSlice = matrix.Slice(Range{0, 4}, 1);

    std::cout << "Col slice ([0, 4), 1):" << std::endl
                << colSlice << std::endl;
    // Col slice ([0, 4), 1):
    // {
    //     Shape = [4]
    //     Data = [2, 6, 10, 14]
    // }

    // slice a block from the matrix
    auto blockSlice = matrix.Slice(Range{1, 3}, Range{1, 3});

    std::cout << "Block slice ([1, 3), [1, 3)):" << std::endl
                << blockSlice << std::endl;
    // Block slice ([1, 3), [1, 3)):
    // {
    //     Shape = [2, 2]
    //     Data = [[6, 7], [10, 11]]
    // }

    // copy the block slice view into a new tensor via a special copy constructor for the tensor class
    // this will make a deep copy of the data in the block slice, where the new tensor will own the data
    // and the new tensor will have a contiguous layout
    Tensor<int, 2> copy(blockSlice);
    std::cout << "Copy of block slice:" << std::endl
                << copy << std::endl;
    // Copy of block slice:
    // {
    //     Shape = [2, 2]
    //     Data = [[6, 7], [10, 11]]
    // }

    return 0;
}

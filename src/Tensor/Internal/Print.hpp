#pragma once

#include "ITensor.hpp"

#include <iostream>

// Provide an operator<< for std::array to print the shape nicely.
template <typename T, std::size_t N>
std::ostream& operator<<(std::ostream& os, std::array<T, N> const& arr)
{
    os << "[";
    for (std::size_t i = 0; i < N; ++i)
    {
        os << arr[i];
        if (i + 1 < N)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

// A helper function that recursively prints tensor elements
template <typename Tensor, size_t Order>
void print_tensor_recursive(std::ostream& out, Tensor const& tensor,
                            std::array<size_t, Order> const& shape,
                            std::array<size_t, Order>& indices, size_t dim)
{
    if (dim == Order)
    {
        // Base case: all indices specified, print one element.
        out << tensor(indices);
    }
    else
    {
        out << "[";
        for (size_t i = 0; i < shape[dim]; ++i)
        {
            indices[dim] = i;
            print_tensor_recursive(out, tensor, shape, indices, dim + 1);
            if (i + 1 < shape[dim])
            {
                out << ", ";
            }
        }
        out << "]";
    }
}

// This operator<< works for any ITensor-derived type that provides a Shape() method
// and an operator() that takes a std::array of indices.
template <typename Derived>
auto operator<<(std::ostream& out, ITensor<Derived> const& tensor) -> std::ostream&
{
    // Cast to the derived type to access its interface.
    auto const& derived = static_cast<Derived const&>(tensor);

    // Print shape.
    auto shape = derived.Shape();
    out << "Shape: " << shape << "\nData: ";

    // Create an index array initialized to zero.
    std::array<size_t, Derived::kOrder> indices{}; // All elements are zero-initialized.

    // Recursively print all tensor elements.
    print_tensor_recursive(out, derived, shape, indices, 0);

    return out;
}

#pragma once

#include <array>
#include <iostream>

// Provide an operator<< for std::array to print the shape nicely.
template <typename T, size_t Order>
std::ostream& operator<<(std::ostream& os, std::array<T, Order> const& arr)
{
    os << "[";
    for (size_t i = 0; i < Order; ++i)
    {
        os << arr[i];
        if (i + 1 < Order)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

// A helper function that recursively prints tensor elements
template <typename T, size_t Order>
void print_tensor_recursive(std::ostream& out, T const& tensor,
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

template <typename T, size_t Order>
auto operator<<(std::ostream& out, T const& tensor) -> std::ostream&
{
    out << "{" << std::endl;
    // Print shape.
    auto shape = tensor.Shape();
    out << "    Shape = " << shape << std::endl
        << "    Data = ";

    // Create an index array initialized to zero.
    std::array<size_t, Order> indices{}; // All elements are zero-initialized.

    // Recursively print all tensor elements.
    print_tensor_recursive(out, tensor, shape, indices, 0);

    out << std::endl
        << "}" << std::endl;

    return out;
}

// Printable mixin via CRTP.
// Derived classes must implement:
//   - Shape() returning std::array<size_t, Order>
//   - operator()(std::array<size_t, Order> const&) for element access.
template <typename Derived, size_t Order>
struct Printable
{
    friend std::ostream& operator<<(std::ostream& out, Derived const& tensor)
    {
        out << "{" << std::endl;
        auto shape = tensor.Shape();
        out << "    Shape = " << shape << std::endl;
        out << "    Data = ";
        std::array<size_t, Order> indices{}; // Zero-initialized index array.
        print_tensor_recursive(out, tensor, shape, indices, 0);
        out << std::endl
            << "}" << std::endl;
        return out;
    }
};

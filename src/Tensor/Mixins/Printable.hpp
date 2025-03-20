#pragma once

#include <array>
#include <iostream>

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

template <typename T, size_t Order>
void print_tensor_recursive(std::ostream& out, T const& tensor,
                            std::array<size_t, Order> const& shape,
                            std::array<size_t, Order>& indices, size_t dim)
{
    if (dim == Order)
    {
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

// template <typename T, size_t Order>
// auto operator<<(std::ostream& out, T const& tensor) -> std::ostream&
// {
//     out << "{" << std::endl;

//     auto shape = tensor.Shape();
//     out << "    Shape = " << shape << std::endl
//         << "    Data = ";

//     std::array<size_t, Order> indices{};

//     print_tensor_recursive(out, tensor, shape, indices, 0);

//     out << std::endl
//         << "}" << std::endl;

//     return out;
// }

template <typename Derived>
struct Printable
{

    static constexpr size_t order = TensorTraits<Derived>::order;

    friend std::ostream& operator<<(std::ostream& out, Derived const& tensor)
    {
        out << "{" << std::endl;
        auto shape = tensor.Shape();
        out << "    Shape = " << shape << std::endl;
        out << "    Data = ";
        std::array<size_t, order> indices{};
        print_tensor_recursive(out, tensor, shape, indices, 0);
        out << std::endl
            << "}" << std::endl;
        return out;
    }
};

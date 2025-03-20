#pragma once

#include <array>
#include <iostream>

template <typename T, size_t Order>
auto operator<<(std::ostream& out, std::array<T, Order> const& arr) -> std::ostream&
{
    out << "[";
    for (size_t i = 0; i < Order; ++i)
    {
        out << arr[i];
        if (i + 1 < Order)
        {
            out << ", ";
        }
    }
    out << "]";
    return out;
}

template <typename T, size_t Order>
auto PrintRecursive(std::ostream& out,
                    T const& tensor,
                    std::array<size_t, Order> const& shape,
                    std::array<size_t, Order>& indices,
                    size_t currentDimension) -> void
{
    if (currentDimension == Order)
    {
        out << tensor(indices);
    }
    else
    {
        out << "[";
        for (size_t i = 0; i < shape[currentDimension]; ++i)
        {
            indices[currentDimension] = i;
            PrintRecursive(out, tensor, shape, indices, currentDimension + 1);
            if (i + 1 < shape[currentDimension])
            {
                out << ", ";
            }
        }
        out << "]";
    }
}

/*
 * Printable mixin for tensor-like classes.
 * Provides operator<<.
 */
template <typename Derived>
class Printable
{
public:

    static constexpr size_t order = TensorTraits<Derived>::order;

    friend auto operator<<(std::ostream& out, Derived const& tensor) -> std::ostream&
    {
        out << "{" << std::endl;
        auto shape = tensor.Shape();
        out << "    Shape = " << shape << std::endl;
        out << "    Data = ";
        std::array<size_t, order> indices{};
        PrintRecursive(out, tensor, shape, indices, 0);
        out << std::endl
            << "}" << std::endl;
        return out;
    }
};

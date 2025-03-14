#pragma once

#include "TensorFwd.hpp"

#include "Utilities.hpp"
#include "Slice.hpp"

#include <array>
#include <iostream>
#include <vector>

template <typename T, size_t Order>
class Tensor {
  private:
    std::array<size_t, Order> shape_;
    std::array<size_t, Order> strides_;
    std::vector<T> data_;

  public:
    Tensor() = delete;
    ~Tensor() = default;

    template <typename... Shape>
    requires(ValidShapeOrIndex<Order, Shape...>)
        Tensor(Shape... shape)
        : Tensor<T, Order>(std::array<size_t, Order>{static_cast<size_t>(shape)...}) {
    }

    template <typename... Shape>
    requires(ValidShapeOrIndex<Order, Shape...>)
        Tensor(T const &initializer, Shape... shape)
        : Tensor(std::array<size_t, Order>{static_cast<size_t>(shape)...}) {
        std::fill(data_.begin(), data_.end(), initializer);
    }

    Tensor(TensorView<T, Order> const &view)
        : Tensor(view.Shape()) {
        // Define a recursive lambda that accumulates indices.
        auto assign_recursive = [&](auto &&self, auto... indices) -> void {
            // Base case: when we have indices for all dimensions, assign the element.
            if
                constexpr(sizeof...(indices) == Order) {
                    this->operator()(indices...) = view(indices...);
                }
            else {
                // Determine the current dimension (number of accumulated indices).
                constexpr size_t dim = sizeof...(indices);
                // Loop over the valid indices for this dimension.
                for (size_t i = 0; i < view.Shape()[dim]; ++i) {
                    self(self, indices..., i);
                }
            }
        };

        // Kick off the recursion with no indices.
        assign_recursive(assign_recursive);
    }

    Tensor(Tensor const &other)
        : Tensor(other.shape_) {
        std::copy(other.data_.begin(), other.data_.end(), data_.begin());
    }

    Tensor(Tensor &&other) noexcept
        : Tensor(other.shape_) {
        data_ = std::move(other.data_);
    }

    auto operator=(Tensor const &other) -> Tensor & {
        if (this != &other) {
            shape_ = other.shape_;
            strides_ = other.strides_;
            data_ = other.data_;
        }
        return *this;
    }

    auto operator=(Tensor &&other) noexcept -> Tensor & {
        if (this != &other) {
            shape_ = other.shape_;
            strides_ = other.strides_;
            data_ = std::move(other.data_);
        }
        return *this;
    }

    template <typename... Indices>
    requires(ValidShapeOrIndex<Order, Indices...>) inline auto operator()(Indices... indices) -> T & {
        return data_[LinearIndex(indices...)];
    }

    template <typename... Indices>
    requires(ValidShapeOrIndex<Order, Indices...>) inline auto operator()(Indices... indices) const -> T const & {
        return data_[LinearIndex(indices...)];
    }

    template <typename... Slices>
    auto Slice(Slices &&... slice_pack) {
        return SliceImpl<T, Order>(shape_, strides_, data_.data(), slice_pack...);
    }

    auto Size() const -> size_t {
        return data_.size();
    }

    auto Dimensions() const -> std::array<size_t, Order> const & {
        return shape_;
    }

    auto Data() -> T * {
        return data_.data();
    }

    auto Data() const -> const T * {
        return data_.data();
    }

    friend std::ostream &operator<<(std::ostream &out, Tensor const &tensor) {
        out << "Tensor(shape = [";
        for (size_t i = 0; i < tensor.shape_.size(); ++i) {
            out << tensor.shape_[i];
            if (i < tensor.shape_.size() - 1) {
                out << ", ";
            }
        }
        out << "], data = [";
        for (size_t i = 0; i < tensor.data_.size(); ++i) {
            out << tensor.data_[i];
            if (i < tensor.data_.size() - 1) {
                out << ", ";
            }
        }
        out << "])";
        return out;
    }

  private:
    Tensor(std::array<size_t, Order> const &shape)
        : shape_(shape), strides_(ComputeStrides(shape_)), data_(ComputeSize(shape_)) {
    }

    template <typename... Indices>
    requires(ValidShapeOrIndex<Order, Indices...>) inline auto LinearIndex(Indices... indices) const -> size_t {
        auto indices_array = std::array<size_t, Order>{static_cast<size_t>(indices)...};
        return std::inner_product(strides_.begin(), strides_.end(), indices_array.begin(), size_t(0));
    }
};

#include "View.hpp"

#pragma once

#include <array>
#include <concepts>
#include <cstddef>

template <typename T, size_t Order>
concept TensorLike = requires(T t) {
    // TensorLike::Data() -> T*
    { t.Data() } -> std::convertible_to<typename T::ValueType*>;

    // TensorLike::Data() const -> T const*
    { std::as_const(t).Data() } -> std::convertible_to<typename T::ValueType const*>;

    // TensorLike::Shape() const -> std::array<size_t, Order>
    { std::as_const(t).Shape() } -> std::convertible_to<std::array<size_t, Order>>;

    // TensorLike::Strides() const -> std::array<size_t, Order>
    { std::as_const(t).Strides() } -> std::convertible_to<std::array<size_t, Order>>;

    // TensorLike::Operator(std::array<size_t, Order>) -> T&
    { t(std::array<size_t, Order>{}) } -> std::convertible_to<typename T::ValueType&>;

    // TensorLike::Operator(std::array<size_t, Order>) const -> T
    { std::as_const(t)(std::array<size_t, Order>{}) } -> std::convertible_to<typename T::ValueType>;

};

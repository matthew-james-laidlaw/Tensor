#pragma once

#include "Forward.hpp"

template <typename T>
struct TensorTraits;

template <typename T, size_t N>
struct TensorTraits<Tensor<T, N>>
{
    using value_type = T;
    static constexpr size_t order = N;
    static constexpr bool has_offset = false;
};

template <typename T, size_t N>
struct TensorTraits<View<T, N>>
{
    using value_type = T;
    static constexpr size_t order = N;
    static constexpr bool has_offset = true;
};

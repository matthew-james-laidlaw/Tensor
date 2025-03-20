#pragma once

#include <array>
#include <concepts>
#include <numeric>
#include <tuple>
#include <utility>

template <typename T, size_t Order>
class View;

struct Range
{
    size_t start;
    size_t stop;
};

template <typename... Slices>
constexpr auto CountRangeSlices() -> size_t
{
    return ((std::same_as<std::decay_t<Slices>, Range> ? 1 : 0) + ...);
}

// Here we require that T satisfies TensorLike<T, Order>
template <typename T, size_t Order, typename... Slices>
auto SliceImpl(T& tensor, Slices&&... slice_pack)
{
    // The new tensor order is the number of Range slices.
    constexpr size_t NewOrder = CountRangeSlices<Slices...>();

    // Calculate the offset. If the tensor is already a View, use its offset.
    size_t offset = 0;
    if constexpr (std::same_as<std::remove_cvref_t<T>, View<typename T::ValueType, T::kOrder>>)
    {
        offset = tensor.Offset();
    }

    std::array<size_t, NewOrder> new_shape{};
    std::array<size_t, NewOrder> new_strides{};
    size_t current_output_dimension = 0;

    // Pack slices into a tuple.
    auto slices = std::make_tuple(std::forward<Slices>(slice_pack)...);

    auto process_dimension = [&](auto i)
    {
        constexpr size_t current_input_dimension = decltype(i)::value;
        auto slice = std::get<i>(slices);

        if constexpr (std::convertible_to<decltype(slice), size_t>)
        {
            size_t slice_index = static_cast<size_t>(slice);
            // For an index slice, update the offset.
            offset += slice_index * tensor.Strides()[current_input_dimension];
        }
        else if constexpr (std::same_as<decltype(slice), Range>)
        {
            auto [slice_start, slice_stop] = slice;
            offset += slice_start * tensor.Strides()[current_input_dimension];
            new_shape[current_output_dimension] = slice_stop - slice_start;
            new_strides[current_output_dimension++] = tensor.Strides()[current_input_dimension];
        }
    };

    // Process each input dimension.
    auto index_sequence = [&]<std::size_t... I>(std::index_sequence<I...>)
    {
        (process_dimension(std::integral_constant<size_t, I>{}), ...);
    };
    index_sequence(std::make_index_sequence<T::kOrder>{});

    return View<typename T::ValueType, NewOrder>{tensor.Data(), new_shape, new_strides, offset};
}

template <typename Derived, size_t Order>
class Sliceable
{
public:

    template <typename... Slices>
    auto Slice(Slices&&... slices)
    {
        auto& self = static_cast<Derived&>(*this);
        return SliceImpl<Derived, Order>(self, std::forward<Slices>(slices)...);
    }
};

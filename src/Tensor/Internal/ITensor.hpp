#pragma once

#include <array>
#include <vector>

template <typename Derived>
class ITensor
{
public:

    auto Data() -> decltype(auto)
    {
        return static_cast<Derived&>(*this).Data();
    }

    auto Data() const -> decltype(auto)
    {
        return static_cast<Derived&>(*this).Data();
    }

    auto Shape() const -> decltype(auto)
    {
        return static_cast<Derived const&>(*this).Shape();
    }

    auto Strides() const -> decltype(auto)
    {
        return static_cast<Derived const&>(*this).Strides();
    }

    template <typename IndexArray>
    inline auto operator()(IndexArray&& indices) -> decltype(auto)
    {
        return static_cast<Derived&>(*this).operator()(indices);
    }

    template <typename IndexArray>
    inline auto operator()(IndexArray&& indices) const -> decltype(auto)
    {
        return static_cast<Derived const&>(*this).operator()(indices);
    }

    template <typename... Slices>
    auto Slice(Slices&&... slices) -> decltype(auto)
    {
        return SliceImpl(static_cast<Derived&>(*this), std::forward<Slices>(slices)...);
    }
};

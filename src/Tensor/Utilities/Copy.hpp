#pragma once

#include <Expect.hpp>

#include <array>

/*
 * Copy the contents of the source container into the destination container, element-by-element.
 * For use when one of the containers is non-contiguous, and memory can't be copied directly.
 * Recursively visits every permutation of indices allowable for the given shape.
 */
template <typename T1, typename T2, size_t Order>
void CopyElementwise(const T1& source, T2& destination)
{
    Expect(source.Shape() == destination.Shape());
    std::array<size_t, Order> currentIndices{};

    auto recursiveCopy = [&](auto&& recursiveCopy, size_t currentDimension) -> void
    {
        if (currentDimension == Order) // base case
        {
            destination(currentIndices) = static_cast<typename T2::ValueType>(source(currentIndices));
        }
        else // recursive case
        {
            for (size_t i = 0; i < source.Shape()[currentDimension]; ++i)
            {
                currentIndices[currentDimension] = i;
                recursiveCopy(recursiveCopy, currentDimension + 1);
            }
        }
    };

    recursiveCopy(recursiveCopy, 0);
}

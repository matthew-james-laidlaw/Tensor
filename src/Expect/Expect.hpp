#pragma once

#include <stdexcept>
#include <string>

template <typename Exception = std::runtime_error>
auto Expect(bool assertion, std::string const& message = "internal error") -> void
{
    if (!assertion)
    {
        throw Exception(message);
    }
}

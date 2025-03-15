#include <gtest/gtest.h>

#include <Dispatch.hpp>

TEST(DispatchTests, DispatchElementMatrixAdd)
{
    size_t height = 512;
    size_t width = 512;

    std::vector<std::vector<int>> m1(height, std::vector<int>(width, 1));
    std::vector<std::vector<int>> m2(height, std::vector<int>(width, 2));
    std::vector<std::vector<int>> m3(height, std::vector<int>(width, 0));

    DispatchElement(height, width, [&](size_t y, size_t x)
    {
        m3[y][x] = m1[y][x] + m2[y][x];
    });

    for (size_t y = 0; y < height; ++y)
    {
        for (size_t x = 0; x < width; ++x)
        {
            EXPECT_EQ(m3[y][x], 3);
        }
    }
}

TEST(DispatchTests, DispatchRowMatrixAdd)
{
    size_t height = 512;
    size_t width = 512;

    std::vector<std::vector<int>> m1(height, std::vector<int>(width, 1));
    std::vector<std::vector<int>> m2(height, std::vector<int>(width, 2));
    std::vector<std::vector<int>> m3(height, std::vector<int>(width, 0));

    DispatchRow(height, [&](size_t y)
    {
        for (size_t x = 0; x < width; ++x)
        {
            m3[y][x] = m1[y][x] + m2[y][x];
        }
    });

    for (size_t y = 0; y < height; ++y)
    {
        for (size_t x = 0; x < width; ++x)
        {
            EXPECT_EQ(m3[y][x], 3);
        }
    }
}

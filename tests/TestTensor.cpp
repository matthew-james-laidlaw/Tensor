#include <gtest/gtest.h>

#include <Tensor.hpp>

TEST(TensorTests, DefaultConstructor)
{
    Tensor<int, 2> m1({4, 2});

    EXPECT_EQ(m1.Shape().size(), 2);
    EXPECT_EQ(m1.Shape()[0], 4);
    EXPECT_EQ(m1.Shape()[1], 2);

    EXPECT_EQ(m1.Strides().size(), 2);
    EXPECT_EQ(m1.Strides()[0], 2);
    EXPECT_EQ(m1.Strides()[1], 1);
}

TEST(TensorTests, FillConstructor)
{
    Tensor<int, 2> m1({4, 2}, 42);

    for (size_t y = 0; y < m1.Shape()[0]; ++y)
    {
        for (size_t x = 0; x < m1.Shape()[1]; ++x)
        {
            EXPECT_EQ(m1({y, x}), 42);
        }
    }
}

TEST(TensorTests, CopyConstructor)
{
    Tensor<int, 2> m1({4, 2}, 42);
    Tensor<int, 2> m2(m1);

    EXPECT_EQ(m2.Shape().size(), 2);
    EXPECT_EQ(m2.Shape()[0], 4);
    EXPECT_EQ(m2.Shape()[1], 2);

    EXPECT_EQ(m2.Strides().size(), 2);
    EXPECT_EQ(m2.Strides()[0], 2);
    EXPECT_EQ(m2.Strides()[1], 1);

    for (size_t y = 0; y < m1.Shape()[0]; ++y)
    {
        for (size_t x = 0; x < m1.Shape()[1]; ++x)
        {
            EXPECT_EQ(m1({y, x}), 42);
            EXPECT_EQ(m2({y, x}), 42);
        }
    }
}

TEST(TensorTests, MoveConstructor)
{
    Tensor<int, 2> m1({4, 2}, 42);
    Tensor<int, 2> m2(std::move(m1));

    EXPECT_EQ(m2.Shape().size(), 2);
    EXPECT_EQ(m2.Shape()[0], 4);
    EXPECT_EQ(m2.Shape()[1], 2);

    EXPECT_EQ(m2.Strides().size(), 2);
    EXPECT_EQ(m2.Strides()[0], 2);
    EXPECT_EQ(m2.Strides()[1], 1);

    EXPECT_EQ(m1.Data(), nullptr);

    for (size_t y = 0; y < m1.Shape()[0]; ++y)
    {
        for (size_t x = 0; x < m1.Shape()[1]; ++x)
        {
            EXPECT_EQ(m2({y, x}), 42);
        }
    }
}

TEST(TensorTests, CopyAssignment)
{
    Tensor<int, 2> m1({4, 2}, 42);
    Tensor<int, 2> m2 = m1;

    EXPECT_EQ(m2.Shape().size(), 2);
    EXPECT_EQ(m2.Shape()[0], 4);
    EXPECT_EQ(m2.Shape()[1], 2);

    EXPECT_EQ(m2.Strides().size(), 2);
    EXPECT_EQ(m2.Strides()[0], 2);
    EXPECT_EQ(m2.Strides()[1], 1);

    for (size_t y = 0; y < m1.Shape()[0]; ++y)
    {
        for (size_t x = 0; x < m1.Shape()[1]; ++x)
        {
            EXPECT_EQ(m1({y, x}), 42);
            EXPECT_EQ(m2({y, x}), 42);
        }
    }
}

TEST(TensorTests, MoveAssignment)
{
    Tensor<int, 2> m1({4, 2}, 42);
    Tensor<int, 2> m2 = std::move(m1);

    EXPECT_EQ(m2.Shape().size(), 2);
    EXPECT_EQ(m2.Shape()[0], 4);
    EXPECT_EQ(m2.Shape()[1], 2);

    EXPECT_EQ(m2.Strides().size(), 2);
    EXPECT_EQ(m2.Strides()[0], 2);
    EXPECT_EQ(m2.Strides()[1], 1);

    EXPECT_EQ(m1.Data(), nullptr);

    for (size_t y = 0; y < m1.Shape()[0]; ++y)
    {
        for (size_t x = 0; x < m1.Shape()[1]; ++x)
        {
            EXPECT_EQ(m2({y, x}), 42);
        }
    }
}

TEST(TensorTests, SliceRow)
{
    Tensor<int, 2> m1({4, 4});
    std::iota(m1.Data(), m1.Data() + 16, 1);

    View<int, 1> row = m1.Slice(0, Range{0, 4});

    EXPECT_EQ(row.Shape()[0], 4);

    EXPECT_EQ(row({0}), 1);
    EXPECT_EQ(row({1}), 2);
    EXPECT_EQ(row({2}), 3);
    EXPECT_EQ(row({3}), 4);
}

TEST(TensorTests, SliceCol)
{
    Tensor<int, 2> m1({4, 4});
    std::iota(m1.Data(), m1.Data() + 16, 1);

    View<int, 1> row = m1.Slice(Range{0, 4}, 0);

    EXPECT_EQ(row.Shape()[0], 4);

    EXPECT_EQ(row({0}), 1);
    EXPECT_EQ(row({1}), 5);
    EXPECT_EQ(row({2}), 9);
    EXPECT_EQ(row({3}), 13);
}

TEST(TensorTests, SliceBlock)
{
    Tensor<int, 2> m1({4, 4});
    std::iota(m1.Data(), m1.Data() + 16, 1);

    View<int, 2> block = m1.Slice(Range{1, 3}, Range{1, 3});

    EXPECT_EQ(block.Shape()[0], 2);
    EXPECT_EQ(block.Shape()[1], 2);

    EXPECT_EQ(block({0, 0}), 6);
    EXPECT_EQ(block({0, 1}), 7);
    EXPECT_EQ(block({1, 0}), 10);
    EXPECT_EQ(block({1, 1}), 11);
}

TEST(TensorTests, SlicePlanes)
{
    Tensor<int, 3> t1({3, 1, 1});
    std::iota(t1.Data(), t1.Data() + 3, 1);
    View<int, 2> p1 = t1.Slice(0, Range{0, 1}, Range{0, 1});
    View<int, 2> p2 = t1.Slice(1, Range{0, 1}, Range{0, 1});
    View<int, 2> p3 = t1.Slice(2, Range{0, 1}, Range{0, 1});

    EXPECT_EQ(p1.Shape().size(), 2);
    EXPECT_EQ(p1.Shape()[0], 1);
    EXPECT_EQ(p1.Shape()[1], 1);
    EXPECT_EQ(p1({0, 0}), 1);

    EXPECT_EQ(p2.Shape().size(), 2);
    EXPECT_EQ(p2.Shape()[0], 1);
    EXPECT_EQ(p2.Shape()[1], 1);
    EXPECT_EQ(p2({0, 0}), 2);

    EXPECT_EQ(p3.Shape().size(), 2);
    EXPECT_EQ(p3.Shape()[0], 1);
    EXPECT_EQ(p3.Shape()[1], 1);
    EXPECT_EQ(p3({0, 0}), 3);
}

TEST(TensorTests, SlicePixels)
{
    Tensor<int, 3> t1({1, 1, 3});
    std::iota(t1.Data(), t1.Data() + 3, 1);
    View<int, 2> p1 = t1.Slice(Range{0, 1}, Range{0, 1}, 0);
    View<int, 2> p2 = t1.Slice(Range{0, 1}, Range{0, 1}, 1);
    View<int, 2> p3 = t1.Slice(Range{0, 1}, Range{0, 1}, 2);

    EXPECT_EQ(p1.Shape().size(), 2);
    EXPECT_EQ(p1.Shape()[0], 1);
    EXPECT_EQ(p1.Shape()[1], 1);
    EXPECT_EQ(p1({0, 0}), 1);

    EXPECT_EQ(p2.Shape().size(), 2);
    EXPECT_EQ(p2.Shape()[0], 1);
    EXPECT_EQ(p2.Shape()[1], 1);
    EXPECT_EQ(p2({0, 0}), 2);

    EXPECT_EQ(p3.Shape().size(), 2);
    EXPECT_EQ(p3.Shape()[0], 1);
    EXPECT_EQ(p3.Shape()[1], 1);
    EXPECT_EQ(p3({0, 0}), 3);
}

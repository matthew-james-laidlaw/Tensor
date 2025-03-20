#include <gtest/gtest.h>

#include <Tensor.hpp>

TEST(TensorTests, DefaultConstructor)
{
    size_t height = 2;
    size_t width = 3;

    auto tensor = Tensor<int, 2>({height, width});

    auto expectedShape = std::array<size_t, 2>{height, width};
    auto expectedStrides = std::array<size_t, 2>{width, 1};

    EXPECT_NE(tensor.Data(), nullptr);
    EXPECT_EQ(tensor.Shape(), expectedShape);
    EXPECT_EQ(tensor.Strides(), expectedStrides);
}

TEST(TensorTests, FillConstructor)
{
    size_t height = 2;
    size_t width = 3;

    int fill = 42;

    auto tensor = Tensor<int, 2>({height, width}, fill);

    for (size_t y = 0; y < height; ++y)
    {
        for (size_t x = 0; x < width; ++x)
        {
            EXPECT_EQ(tensor({y, x}), fill);
        }
    }
}

TEST(TensorTests, CopyConstructor)
{
    size_t height = 2;
    size_t width = 3;

    int fill = 42;

    auto t1 = Tensor<int, 2>({height, width}, fill);
    auto t2(t1);

    EXPECT_NE(t1.Data(), nullptr);
    EXPECT_NE(t2.Data(), nullptr);

    for (size_t y = 0; y < height; ++y)
    {
        for (size_t x = 0; x < width; ++x)
        {
            EXPECT_EQ(t1({y, x}), t2({y, x}));
        }
    }
}

TEST(TensorTests, MoveConstructor)
{
    size_t height = 2;
    size_t width = 3;

    int fill = 42;

    auto t1 = Tensor<int, 2>({height, width}, fill);
    auto t2(std::move(t1));

    EXPECT_EQ(t1.Data(), nullptr);
    EXPECT_NE(t2.Data(), nullptr);

    for (size_t y = 0; y < height; ++y)
    {
        for (size_t x = 0; x < width; ++x)
        {
            EXPECT_EQ(t2({y, x}), fill);
        }
    }
}

TEST(TensorTests, CopyAssignment)
{
    size_t height = 2;
    size_t width = 3;

    int fill = 42;

    auto t1 = Tensor<int, 2>({height, width}, fill);
    auto t2 = t1;

    EXPECT_NE(t1.Data(), nullptr);
    EXPECT_NE(t2.Data(), nullptr);

    for (size_t y = 0; y < height; ++y)
    {
        for (size_t x = 0; x < width; ++x)
        {
            EXPECT_EQ(t1({y, x}), t2({y, x}));
        }
    }
}

TEST(TensorTests, MoveAssignment)
{
    size_t height = 2;
    size_t width = 3;

    int fill = 42;

    auto t1 = Tensor<int, 2>({height, width}, fill);
    auto t2 = std::move(t1);

    EXPECT_EQ(t1.Data(), nullptr);
    EXPECT_NE(t2.Data(), nullptr);

    for (size_t y = 0; y < height; ++y)
    {
        for (size_t x = 0; x < width; ++x)
        {
            EXPECT_EQ(t2({y, x}), fill);
        }
    }
}

TEST(TensorTests, Indexing)
{
    size_t height = 2;
    size_t width = 3;

    auto tensor = Tensor<int, 2>({height, width});

    size_t size = height * width;
    std::iota(tensor.Data(), tensor.Data() + size, 0);

    for (size_t y = 0; y < height; ++y)
    {
        for (size_t x = 0; x < width; ++x)
        {
            EXPECT_EQ(tensor({y, x}), y * width + x);
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

TEST(TensorTests, ViewCopyConstructor)
{
    Tensor<int, 2> t1({2, 2}, 42);
    View<int, 2> v1 = t1.Slice(Range{0, 2}, Range{0, 2});
    Tensor<int, 2> t2(v1);

    EXPECT_EQ(t1.Data(), v1.Data());
    EXPECT_NE(t1.Data(), t2.Data());

    for (size_t y = 0; y < 2; ++y)
    {
        for (size_t x = 0; x < 2; ++x)
        {
            EXPECT_EQ(t1({y, x}), v1({y, x}));
            EXPECT_EQ(t1({y, x}), t2({y, x}));
        }
    }
}

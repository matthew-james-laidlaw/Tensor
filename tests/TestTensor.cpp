#include <gtest/gtest.h>

#include <Tensor.hpp>

TEST(TensorTests, DefaultConstructor)
{
    Tensor<int, 2> matrix({4, 2});
    
    EXPECT_EQ(matrix.Shape().size(), 2);
    EXPECT_EQ(matrix.Shape()[0], 4);
    EXPECT_EQ(matrix.Shape()[1], 2);

    EXPECT_EQ(matrix.Strides().size(), 2);
    EXPECT_EQ(matrix.Strides()[0], 2);
    EXPECT_EQ(matrix.Strides()[1], 1);
}



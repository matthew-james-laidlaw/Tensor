# Tensor User Guide

## 1. Tensor Class Overview
The `Tensor` class is a generic, multi-dimensional container that owns a contiguous array of elements. The class maintains shape and stride information, and exposes an `operator()`, for indexing into the underlying buffer via a multi-dimensional index. Each `Tensor` is templated on its value type and its order (i.e. number of dimensions). For example, a vector could be represented as a `Tensor<T, 1>`, and a matrix could be represented as a `Tensor<T, 2>`.

```
template <typename T, size_t N>
class Tensor
{
    ...
};
```

##  2. Constructors

### Shape Constructor
The most basic constructor allows the creation of a `Tensor` with a given shape. The number of elements in the brace initializer must be equal to the dimensionality of the `Tensor`. This constructor allocates the `Tensor`, but does not initialize any elements.

```
auto t1 = Tensor<int, 1>({1});        // shape = 1
auto t2 = Tensor<int, 2>({1, 2});     // shape = 1x2
auto t3 = Tensor<int, 3>({1, 2, 3});  // shape = 1x2x3
auto t4 = Tensor<int, 4>(...);
```

### Fill Constructor
The fill constructor behaves similarly to the shape constructor but initializes all elements to the given value.

```
auto t = Tensor<int, 2>({2, 2}, 42)

t = [ 42, 42 ]
    [ 42, 42 ]
```

### Copy & Move
The copy constructor and copy assignment operator create a deep copy of the given `Tensor` with its own array of elements identical to the copied-from `Tensor`. The move constructor and move assignment operator move the underlying buffer and shape information to the new `Tensor`, leaving the moved-from `Tensor` in an undefined state.

### View Constructor
(See [section 6](#6-view) for a description of the `View` class).

The view constructor allows for creation of a data-owning `Tensor` from a non-owning `View`. The `Tensor` is constructed from the shape information of the `View` and each element in the `View` is copied to the new `Tensor`.

```
auto v = View<int, 2>(...);
auto t = Tensor<int, 2>(v);
```

## 3. Indexing
Multi-dimensional indexing is supported via `operator()`. Similar to the shape constructor, the indexing operator takes in a brace initializer of indices which must have the same number of elements as the dimensionality of the `Tensor`.

```
auto t = Tensor<int, 2>({2, 2});
std::iota(t.Data(), t.Data() + 4, 1);

t = [ 1 2 ]
    [ 3 4 ]

t({0, 0}) == 1
t({0, 1}) == 2
t({1, 0}) == 3
t({1, 1}) == 4

auto t = Tensor<int, 3>({1, 1, 3});
std::iota(t.Data(), t.Data() + 3, 1);

t = [[[1, 2, 3]]]

t({0, 0, 0}) == 1
t({0, 0, 1}) == 2
t({0, 0, 2}) == 3
```

## 4. Slicing
Slicing allows for retrieval of a selection of data elements, rather than a single element, like indexing would. To slice a `Tensor` call the `Slice` method with 1 to N slice types, one for each dimension, If fewer slices are provided than the number of dimensions, a full slice is assumed. Slices come in two forms: `index` and `range`. For a given dimension, an `index` slice selects a single element from that dimension, effectively reducing the dimensionality of the result by one. A `range` slice selects a span of positions along that dimension, preserving the results dimensionality, but constraining which elements to select. An index slice is simply a number, while a `range` slice is a type which can be constucted using aggregate initialization syntax `Range{<start>, <stop>}`. `Slice` returns a non-owning `View` over the original matrix.

```
auto t = Tensor<int, 2>({4, 4});
std::iota(t.Data(), t.Data() + 16, 1);

t = [  1  2  3  4 ]
    [  5  6  7  8 ]
    [  9 10 11 12 ]
    [ 13 14 15 16 ]

auto row = t.Slice(0, Range{0, 4}); // select the first row, all elements
row = [ 1 2 3 4 ]

auto col = t.Slice(Range{0, 4}, 0); // select the first column, all elements
col = [ 1 5 9 13 ]

auto block = t.Slice(Range{1, 3}, Range{1, 3}); // select the central 2x2 block
block = [  6  7 ]
        [ 10 11 ]
```

## 5. Printing
`Tensor` and `View` can be printed via `operator<<`.

```
auto t = Tensor<int, 2>({2, 2});
std::iota(t.Data(), t.Data() + 16, 1);

std::cout << t << std::endl;

// prints
{
    Shape=[2, 2]
    Data=[[1, 2], [3, 4]]
}
```

## 6. View
A `View` is a non-owning wrapper over a `Tensor`'s data that is returned via the slicing functionality. A `Tensor` can be constructed from a `View`, in which the `View`'s contents will be copied over to the new `Tensor`. This does not impact the contents of the `View` object, nor does it impact the contents of the underlying `Tensor` object pointed to by the `View`.

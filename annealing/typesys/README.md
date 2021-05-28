# Type system

## Scalar type
* `num` : Numeric
  * `int32`, `int64`, `int` (= `int32`), `long` (= `int64`)
  * `float32`, `float64`, `float` (= `float32`), `double` (= `float64`)
  * `complex64`, `complex128`, `complex` (= `complex64`)
* `bool`: Boolean
* `str` : String

## Collection type
* Python-conforming collection
  * `tuple`: Tuple (ordered and unchangeable; allow duplicate elements)
  * `list`: List (ordered and changeable; allow duplicate elements)
  * `set`: Set (unordered and unindexed; no duplicate members)
  * `dict`: Dictionary (unordered, changeable and indexed; no duplicate members)
  * Note: Type annotation is required when accessing elements.
* Uniform collection whose elements have unique type T
  * `UTuple(T)`: Uniform Tuple
  * `UList(T)`: Uniform List
  * `USet(T)`: Uniform Set
  * `UDict(T)`: Uniform Dictionary

## Array type (n-dimensional, dense, with unique type T)
* `Array(T, n)`
  * `T` is scalar type.
  * `n` is integer value.
* `T[lw_1:up_1, lw_2:up_2, ..., lw_n:up_n]`
  * `lw_i` and `up_i` are optional.
  * When `up_i` is given, i-th dimension size `size_i` is computed as:
    * `size_i` = `up_i` - `lw_i` (if `lw_i` is given)
    * `size_i` = `up_i` (otherwise)

## Sparse Matrix type (2-dimensional, sparse, with unique type T)
* `SparseMat(T)`
  * `T` is scalar type.
* `T[sparse, lw_1:up_1, lw_2:up_2]`
  * `T` is scalar type.
  * `lw_i` and `up_i` are optional (same as Array).

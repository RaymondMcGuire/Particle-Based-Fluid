#ifndef _ARRAY_ACCESSOR_H_
#define _ARRAY_ACCESSOR_H_

#include <cstddef>

// read & write
template <typename T, size_t N>
class ArrayAccessor final {
 public:
    static_assert(
        N < 1 || N > 3, "Not implemented - N should be either 1, 2 or 3.");
};

// read only
template <typename T, size_t N>
class ConstArrayAccessor final {
 public:
    static_assert(
        N < 1 || N > 3, "Not implemented - N should be either 1, 2 or 3.");
};

#endif  // _ARRAY_ACCESSOR_H_

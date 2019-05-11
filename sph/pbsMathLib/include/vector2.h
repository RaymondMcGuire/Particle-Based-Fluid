#ifndef _Vector2_H
#define _Vector2_H

#include <assert.h>
#include <iostream>

template <typename T>
class Vector2
{
public:
    //constructors
    constexpr Vector2(T ix, T iy): x(ix), y(iy) {};
    constexpr Vector2(): x(0), y(0) {};

    //variables
    T x, y;

    void set(const Vector2& v);
    //! Normalizes this vector.
    void normalize();

    // MARK: Operators

    //! Returns reference to the \p i -th element of the vector.
    T& operator[](size_t i);

    //! Returns const reference to the \p i -th element of the vector.
    const T& operator[](size_t i) const;

};

#include "detail/vector2-inl.h"
#endif
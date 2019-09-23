#ifndef _FUNCTORS_H_
#define _FUNCTORS_H_

#include <functional>

//! Type casting operator.
template <typename T, typename U>
struct TypeCast
{
    constexpr U operator()(const T &a) const;
};

//! Reverse minus operator.
template <typename T>
struct RMinus
{
    constexpr T operator()(const T &a, const T &b) const;
};

//! Reverse divides operator.
template <typename T>
struct RDivides
{
    constexpr T operator()(const T &a, const T &b) const;
};
#include "detail/functors-inl.h"

#endif // _FUNCTORS_H_

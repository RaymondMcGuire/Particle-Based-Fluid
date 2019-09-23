#ifndef _DETAIL_FUNCTORS_INL_H_
#define _DETAIL_FUNCTORS_INL_H_

#include "functors.h"

template <typename T, typename U>
constexpr U TypeCast<T, U>::operator()(const T &a) const
{
    return static_cast<U>(a);
}

template <typename T>
constexpr T RMinus<T>::operator()(const T &a, const T &b) const
{
    return b - a;
}

template <typename T>
constexpr T RDivides<T>::operator()(const T &a, const T &b) const
{
    return b / a;
}

#endif // _DETAIL_FUNCTORS_INL_H_

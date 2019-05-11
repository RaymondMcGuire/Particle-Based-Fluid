#ifndef DETAIL_VECTOR2_INL_H_
#define DETAIL_VECTOR2_INL_H_


template <typename T>
void Vector2<T>::set(const Vector2& v) {
    x = v.x;
    y = v.y;
}

template <typename T>
void Vector2<T>::normalize() {
    T l = length();
    x /= l;
    y /= l;
}

// Operators
template <typename T>
T &Vector2<T>::operator[](size_t i)
{
    assert(i < 2);
    return (&x)[i];
}

template <typename T>
const T& Vector2<T>::operator[](size_t i) const {
    assert(i < 2);
    return (&x)[i];
}


#endif
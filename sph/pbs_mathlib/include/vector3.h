#ifndef _Vector3_H
#define _Vector3_H

#include <assert.h>
#include <iostream>

template <typename T>
class Vector3
{
public:
    //constructors
    constexpr Vector3(T ix, T iy, T iz): x(ix), y(iy), z(iz) {};
    constexpr Vector3(): x(0), y(0), z(0) {};

    //variables
    T x, y, z;

    void set(const Vector3& v);
    //! Normalizes this vector.
    void normalize();

    // MARK: Binary operations: new instance = this (+) v

    //! Computes this + (v, v, v).
    Vector3 add(T v) const;

    //! Computes this + (v.x, v.y, v.z).
    Vector3 add(const Vector3& v) const;

    //! Computes this - (v, v, v).
    Vector3 sub(T v) const;

    //! Computes this - (v.x, v.y, v.z).
    Vector3 sub(const Vector3& v) const;

    //! Computes this * (v, v, v).
    Vector3 mul(T v) const;

    //! Computes this * (v.x, v.y, v.z).
    Vector3 mul(const Vector3& v) const;
    //! Computes this / (v, v, v).
    Vector3 div(T v) const;

    //! Computes this / (v.x, v.y, v.z).
    Vector3 div(const Vector3& v) const;

    //! Computes dot product.
    T dot(const Vector3& v) const;

    //! Comptues cross product.
    Vector3 cross(const Vector3& v) const;

    // MARK: Binary operations: new instance = v (+) this

    //! Computes (v, v, v) - this.
    Vector3 rsub(T v) const;

    //! Computes (v.x, v.y, v.z) - this.
    Vector3 rsub(const Vector3& v) const;

    //! Computes (v, v, v) / this.
    Vector3 rdiv(T v) const;

    //! Computes (v.x, v.y, v.z) / this.
    Vector3 rdiv(const Vector3& v) const;

    //! Computes \p v cross this.
    Vector3 rcross(const Vector3& v) const;

    // MARK: Augmented operations: this (+)= v

    //! Computes this += (v, v, v).
    void iadd(T v);

    //! Computes this += (v.x, v.y, v.z).
    void iadd(const Vector3& v);

    //! Computes this -= (v, v, v).
    void isub(T v);

    //! Computes this -= (v.x, v.y, v.z).
    void isub(const Vector3& v);

    //! Computes this *= (v, v, v).
    void imul(T v);

    //! Computes this *= (v.x, v.y, v.z).
    void imul(const Vector3& v);

    //! Computes this /= (v, v, v).
    void idiv(T v);

    //! Computes this /= (v.x, v.y, v.z).
    void idiv(const Vector3& v);

    //! Returns normalized vector.
    Vector3 normalized() const;

    //! Returns the length of the vector.
    T length() const;

    //! Returns the squared length of the vector.
    T lengthSquared() const;

    // MARK: Operators

    //! Returns reference to the \p i -th element of the vector.
    T& operator[](size_t i);

    //! Returns const reference to the \p i -th element of the vector.
    const T& operator[](size_t i) const;

    //! Set x and y with other vector \p pt.
    Vector3& operator=(const Vector3& v);

    //! Computes this += (v, v)
    Vector3& operator+=(T v);

    //! Computes this += (v.x, v.y)
    Vector3& operator+=(const Vector3& v);

    //! Computes this -= (v, v)
    Vector3& operator-=(T v);

    //! Computes this -= (v.x, v.y)
    Vector3& operator-=(const Vector3& v);

    //! Computes this *= (v, v)
    Vector3& operator*=(T v);

    //! Computes this *= (v.x, v.y)
    Vector3& operator*=(const Vector3& v);

    //! Computes this /= (v, v)
    Vector3& operator/=(T v);

    //! Computes this /= (v.x, v.y)
    Vector3& operator/=(const Vector3& v);

    //! Returns true if \p other is the same as this vector.
    bool operator==(const Vector3& v) const;

    //! Returns true if \p other is the not same as this vector.
    bool operator!=(const Vector3& v) const;
};

//! Positive sign operator.
template <typename T>
Vector3<T> operator+(const Vector3<T>& a);

//! Negative sign operator.
template <typename T>
Vector3<T> operator-(const Vector3<T>& a);

//! Computes (a, a, a) + (b.x, b.y, b.z).
template <typename T>
Vector3<T> operator+(T a, const Vector3<T>& b);

//! Computes (a.x, a.y, a.z) + (b.x, b.y, b.z).
template <typename T>
Vector3<T> operator+(const Vector3<T>& a, const Vector3<T>& b);

//! Computes (a.x, a.y, a.z) - (b, b, b).
template <typename T>
Vector3<T> operator-(const Vector3<T>& a, T b);

//! Computes (a, a, a) - (b.x, b.y, b.z).
template <typename T>
Vector3<T> operator-(T a, const Vector3<T>& b);

//! Computes (a.x, a.y, a.z) - (b.x, b.y, b.z).
template <typename T>
Vector3<T> operator-(const Vector3<T>& a, const Vector3<T>& b);

//! Computes (a.x, a.y, a.z) * (b, b, b).
template <typename T>
Vector3<T> operator*(const Vector3<T>& a, T b);

//! Computes (a, a, a) * (b.x, b.y, b.z).
template <typename T>
Vector3<T> operator*(T a, const Vector3<T>& b);

//! Computes (a.x, a.y, a.z) * (b.x, b.y, b.z).
template <typename T>
Vector3<T> operator*(const Vector3<T>& a, const Vector3<T>& b);

//! Computes (a.x, a.y, a.z) / (b, b, b).
template <typename T>
Vector3<T> operator/(const Vector3<T>& a, T b);

//! Computes (a, a, a) / (b.x, b.y, b.z).
template <typename T>
Vector3<T> operator/(T a, const Vector3<T>& b);

//! Computes (a.x, a.y, a.z) / (b.x, b.y, b.z).
template <typename T>
Vector3<T> operator/(const Vector3<T>& a, const Vector3<T>& b);

#include "detail/vector3-inl.h"
#endif
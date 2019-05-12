#ifndef DETAIL_VECTOR3_INL_H_
#define DETAIL_VECTOR3_INL_H_


template <typename T>
void Vector3<T>::set(const Vector3& v) {
    x = v.x;
    y = v.y;
    z = v.z;
}

template <typename T>
void Vector3<T>::normalize() {
    T l = length();
    x /= l;
    y /= l;
    z /= l;
}

// Binary operators: new instance = this (+) v
template <typename T>
Vector3<T> Vector3<T>::add(T v) const {
    return Vector(x + v, y + v, z + v);
}

template <typename T>
Vector3<T> Vector3<T>::add(const Vector3& v) const {
    return Vector3(x + v.x, y + v.y, z + v.z);
}

template <typename T>
Vector3<T> Vector3<T>::sub(T v) const {
    return Vector3(x - v, y - v, z - v);
}

template <typename T>
Vector3<T> Vector3<T>::sub(const Vector3& v) const {
    return Vector3(x - v.x, y - v.y, z - v.z);
}

template <typename T>
Vector3<T> Vector3<T>::mul(T v) const {
    return Vector3(x * v, y * v, z * v);
}

template <typename T>
Vector3<T> Vector3<T>::mul(const Vector3& v) const {
    return Vector3(x * v.x, y * v.y, z * v.z);
}

template <typename T>
Vector3<T> Vector3<T>::div(T v) const {
    return Vector3(x / v, y / v, z / v);
}

template <typename T>
Vector3<T> Vector3<T>::div(const Vector3& v) const {
    return Vector3(x / v.x, y / v.y, z / v.z);
}

template <typename T>
T Vector3<T>::dot(const Vector3& v) const {
    return x * v.x + y * v.y + z * v.z;
}

template <typename T>
Vector3<T> Vector3<T>::cross(const Vector3& v) const {
    return Vector3(y * v.z - v.y * z, z * v.x - v.z * x, x * v.y - v.x * y);
}

// Binary operators: new instance = v (+) this
template <typename T>
Vector3<T> Vector3<T>::rsub(T v) const {
    return Vector(v - x, v - y, v - z);
}

template <typename T>
Vector3<T> Vector3<T>::rsub(const Vector3& v) const {
    return Vector3(v.x - x, v.y - y, v.z - z);
}

template <typename T>
Vector3<T> Vector3<T>::rdiv(T v) const {
    return Vector3(v / x, v / y, v / z);
}

template <typename T>
Vector3<T> Vector3<T>::rdiv(const Vector3& v) const {
    return Vector(v.x / x, v.y / y, v.z / z);
}

template <typename T>
Vector3<T> Vector3<T>::rcross(const Vector3& v) const {
    return Vector3(v.y * z - y * v.z, v.z * x - z * v.x, v.x * y - x * v.y);
}

// Augmented operators: this (+)= v
template <typename T>
void Vector3<T>::iadd(T v) {
    x += v;
    y += v;
    z += v;
}

template <typename T>
void Vector3<T>::iadd(const Vector3& v) {
    x += v.x;
    y += v.y;
    z += v.z;
}

template <typename T>
void Vector3<T>::isub(T v) {
    x -= v;
    y -= v;
    z -= v;
}

template <typename T>
void Vector3<T>::isub(const Vector3& v) {
    x -= v.x;
    y -= v.y;
    z -= v.z;
}

template <typename T>
void Vector3<T>::imul(T v) {
    x *= v;
    y *= v;
    z *= v;
}

template <typename T>
void Vector3<T>::imul(const Vector3& v) {
    x *= v.x;
    y *= v.y;
    z *= v.z;
}

template <typename T>
void Vector3<T>::idiv(T v) {
    x /= v;
    y /= v;
    z /= v;
}

template <typename T>
void Vector3<T>::idiv(const Vector3& v) {
    x /= v.x;
    y /= v.y;
    z /= v.z;
}

template <typename T>
Vector3<T> Vector3<T>::normalized() const {
    T l = length();
    return Vector(x / l, y / l, z / l);
}

template <typename T>
T Vector3<T>::length() const {
    return std::sqrt(x * x + y * y + z * z);
}

template <typename T>
T Vector3<T>::lengthSquared() const {
    return x * x + y * y + z * z;
}

// Operators
template <typename T>
T &Vector3<T>::operator[](size_t i)
{
    assert(i < 3);
    return (&x)[i];
}

template <typename T>
const T& Vector3<T>::operator[](size_t i) const {
    assert(i < 3);
    return (&x)[i];
}

template <typename T>
Vector3<T>& Vector3<T>::operator=(const Vector3& v) {
    set(v);
    return (*this);
}

template <typename T>
Vector3<T>& Vector3<T>::operator+=(T v) {
    iadd(v);
    return (*this);
}

template <typename T>
Vector3<T>& Vector3<T>::operator+=(const Vector3& v) {
    iadd(v);
    return (*this);
}

template <typename T>
Vector3<T>& Vector3<T>::operator-=(T v) {
    isub(v);
    return (*this);
}

template <typename T>
Vector3<T>& Vector3<T>::operator-=(const Vector3& v) {
    isub(v);
    return (*this);
}

template <typename T>
Vector3<T>& Vector3<T>::operator*=(T v) {
    imul(v);
    return (*this);
}

template <typename T>
Vector3<T>& Vector3<T>::operator*=(const Vector3& v) {
    imul(v);
    return (*this);
}

template <typename T>
Vector3<T>& Vector3<T>::operator/=(T v) {
    idiv(v);
    return (*this);
}

template <typename T>
Vector3<T>& Vector3<T>::operator/=(const Vector3& v) {
    idiv(v);
    return (*this);
}

template <typename T>
Vector3<T> operator+(const Vector3<T>& a) {
    return a;
}

template <typename T>
Vector3<T> operator-(const Vector3<T>& a) {
    return Vector3<T>(-a.x, -a.y, -a.z);
}

template <typename T>
Vector3<T> operator+(const Vector3<T>& a, T b) {
    return a.add(b);
}

template <typename T>
Vector3<T> operator+(T a, const Vector3<T>& b) {
    return b.add(a);
}

template <typename T>
Vector3<T> operator+(const Vector3<T>& a, const Vector3<T>& b) {
    return a.add(b);
}

template <typename T>
Vector3<T> operator-(const Vector3<T>& a, T b) {
    return a.sub(b);
}

template <typename T>
Vector3<T> operator-(T a, const Vector3<T>& b) {
    return b.rsub(a);
}

template <typename T>
Vector3<T> operator-(const Vector3<T>& a, const Vector3<T>& b) {
    return a.sub(b);
}

template <typename T>
Vector3<T> operator*(const Vector3<T>& a, T b) {
    return a.mul(b);
}

template <typename T>
Vector3<T> operator*(T a, const Vector3<T>& b) {
    return b.mul(a);
}

template <typename T>
Vector3<T> operator*(const Vector3<T>& a, const Vector3<T>& b) {
    return a.mul(b);
}

template <typename T>
Vector3<T> operator/(const Vector3<T> &a, T b)
{
    return a.div(b);
}

template <typename T>
Vector3<T> operator/(T a, const Vector3<T>& b) {
    return b.rdiv(a);
}

template <typename T>
Vector3<T> operator/(const Vector3<T>& a, const Vector3<T>& b) {
    return a.div(b);
}

#endif
#ifndef _Vector2_
#define _Vector2_

#include <iostream>

template <typename T>
class Vector2
{
public:
    //constructors
    Vector2(T ix, T iy);
    Vector2(T ia);
    Vector2();

    //variables
    T x, y;

    //operators
    Vector2 operator+(const Vector2 &right) const;
    Vector2 operator-(const Vector2 &right) const;
    Vector2 operator*(const Vector2 &right) const;
    Vector2 operator/(const Vector2 &right) const;

    Vector2 operator+(const T &right) const;
    Vector2 operator-(const T &right) const;
    Vector2 operator*(const T &right) const;
    Vector2 operator/(const T &right) const;

    Vector2 &operator+=(const Vector2 &right);
    Vector2 &operator-=(const Vector2 &right);
    Vector2 &operator*=(const Vector2 &right);
    Vector2 &operator/=(const Vector2 &right);

    Vector2 &operator+=(const T &right);
    Vector2 &operator-=(const T &right);
    Vector2 &operator*=(const T &right);
    Vector2 &operator/=(const T &right);
    //functions
    /**
	 * @brief Change the vector length to exactly one
	 */
    void normalize();
    /**
	 * @brief Rotate the vector clockwize in the z direction
	 * 
	 * @param rotation Rotation angle in radians.
	 */
    void rotate(T rotation);
    /**
	 * @brief Rotate the vector clockwize in the z direction
	 * 
	 * @param rotation Rotation angle in radians.
	 */
    Vector2 rotated(T rotation) const;
    /**
	 * @brief Return a vector that is the normelized vector of this vector
	 * @details [long description]
	 * @return this vector normilized
	 */
    Vector2 normalized() const;
    /**
	 * @brief Calcultes the length of the vector and returns it.
	 * @details [long description]
	 * @return The lenght of this vector
	 */
    T absolute() const;
    /**
	 * @brief Calculate the dot product of two vetors.
	 * @details The dot product is the product of the vector in the
	 * same length.
	 * 
	 * @param left Left hand side of the dot operator.
	 * @param right Right hand side of the dot operator.
	 * 
	 * @return Vector result of the dot operation.
	 */
    static T dotProduct(const Vector2 &left, const Vector2 &right);
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const Vector2<T> &vector2);
#endif
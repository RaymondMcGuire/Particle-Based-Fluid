#ifndef _TYPE_HELPERS_H_
#define _TYPE_HELPERS_H_

//! Returns the type of the value itself.
template <typename T>
struct ScalarType {
    typedef T value;
};

#endif  // _TYPE_HELPERS_H_

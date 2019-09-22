#ifndef _DEFINE_H_
#define _DEFINE_H_

#include <BaseTsd.h>
typedef SSIZE_T ssize_t;

#if defined(DEBUG) || defined(_DEBUG)
#   define BBR_DEBUG_MODE
#   include <cassert>
#   define ASSERT(x) assert(x)
#else
#   define ASSERT(x)
#endif

#endif // _DEFINE_H_
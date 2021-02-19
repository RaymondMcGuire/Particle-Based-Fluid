/*
 * @Author: Xu.WANG
 * @Date: 2021-02-04 12:36:10
 * @LastEditTime: 2021-02-04 15:12:42
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\data\cuda_array.cuh
 */

#ifndef _CUDA_ARRAY_CUH_
#define _CUDA_ARRAY_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI
{
    template <typename T>
    class CudaArray
    {
        static_assert(
            IsSame_Float<T>::value || IsSame_Float2<T>::value || IsSame_Float3<T>::value || IsSame_Float4<T>::value ||
                IsSame_Int<T>::value || IsSame_UInt<T>::value,
            "data type is not correct");

    public:
        explicit CudaArray(const uint len)
            : mLen(len),
              mArray([len]() {
                  T *ptr;
                  KIRI_CUCALL(cudaMalloc((void **)&ptr, sizeof(T) * len));
                  SharedPtr<T> t(new (ptr) T[len], [](T *ptr) { KIRI_CUCALL(cudaFree(ptr)); });
                  return t;
              }())
        {
            this->Clear();
        }

        CudaArray(const CudaArray &) = delete;
        CudaArray &operator=(const CudaArray &) = delete;
        T *Data(const int offset = 0) const
        {
            return mArray.get() + offset;
        }

        uint Length() const { return mLen; }
        void Clear()
        {
            KIRI_CUCALL(cudaMemset(this->Data(), 0, sizeof(T) * this->Length()));
        }

        ~CudaArray() noexcept {}

    private:
        const uint mLen;
        const SharedPtr<T> mArray;
    };
} // namespace KIRI

#endif
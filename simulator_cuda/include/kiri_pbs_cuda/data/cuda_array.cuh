/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2021-02-05 12:33:37
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-22 17:00:54
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\data\cuda_array.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
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
    public:
        explicit CudaArray(const size_t len, const bool unified = false)
            : mLen(len),
              mArray([len, unified]()
                     {
                  T *ptr;
                  if (unified)
                      KIRI_CUCALL(cudaMallocManaged((void **)&ptr, sizeof(T) * len));
                  else
                      KIRI_CUCALL(cudaMalloc((void **)&ptr, sizeof(T) * len));
                  SharedPtr<T> t(new (ptr) T[len], [](T *ptr) { KIRI_CUCALL(cudaFree(ptr)); });
                  return t; }())
        {
            this->clear();
        }

        CudaArray(const CudaArray &) = delete;
        CudaArray &operator=(const CudaArray &) = delete;

        ~CudaArray() noexcept
        {
        }

        T *data(const int offset = 0) const
        {
            return mArray.get() + offset;
        }

        size_t length() const { return mLen; }

        void resize(const size_t len, const bool unified = false)
        {
            mLen = len;
            mArray = SharedPtr<T>([len, unified]()
                                  {
                T *ptr;
                if (unified)
                    KIRI_CUCALL(cudaMallocManaged((void **)&ptr, sizeof(T) * len));
                else
                    KIRI_CUCALL(cudaMalloc((void **)&ptr, sizeof(T) * len));
                SharedPtr<T> t(new (ptr) T[len], [](T *ptr)
                               { KIRI_CUCALL(cudaFree(ptr)); });
                return t; }());
        }

        void clear()
        {
            KIRI_CUCALL(cudaMemset(this->data(), 0, sizeof(T) * this->length()));
        }

        void copyToVec(std::vector<T> *dst) const
        {
            KIRI_CUCALL(cudaMemcpy(&dst->front(), this->data(), mLen * sizeof(T), cudaMemcpyDeviceToHost));
        }

    private:
         size_t mLen;
         SharedPtr<T> mArray;
    };
} // namespace KIRI

#endif
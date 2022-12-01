/*** 
 * @Author: Xu.WANG
 * @Date: 2020-11-04 03:24:07
 * @LastEditTime: 2021-08-18 09:06:34
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriExamples\include\fbs\fbs_helper.h
 */
#include <fbs/generated/basic_types_generated.h>
#include <kiri_pbs_cuda/cuda_helper/helper_math.h>
namespace KIRI
{

    inline FlatBuffers::float2 KiriCUDAToFbs(const float2 &vec)
    {
        return FlatBuffers::float2(vec.x, vec.y);
    }

    inline FlatBuffers::float3 KiriCUDAToFbs(const float3 &vec)
    {
        return FlatBuffers::float3(vec.x, vec.y, vec.z);
    }

    inline FlatBuffers::int3 KiriCUDAToFbs(const int3 &vec)
    {
        return FlatBuffers::int3(vec.x, vec.y, vec.z);
    }

    inline FlatBuffers::uint3 KiriCUDAToFbs(const uint3 &vec)
    {
        return FlatBuffers::uint3(vec.x, vec.y, vec.z);
    }

    inline float2 FbsToKiriCUDA(const FlatBuffers::float2 &vec)
    {
        return make_float2(vec.x(), vec.y());
    }

    inline float3 FbsToKiriCUDA(const FlatBuffers::float3 &vec)
    {
        return make_float3(vec.x(), vec.y(), vec.z());
    }

    inline int3 FbsToKiriCUDA(const FlatBuffers::int3 &vec)
    {
        return make_int3(vec.x(), vec.y(), vec.z());
    }

    inline uint3 FbsToKiriCUDA(const FlatBuffers::uint3 &vec)
    {
        return make_uint3(vec.x(), vec.y(), vec.z());
    }

    inline Vector3F FbsToKiri(const FlatBuffers::float3 &vec)
    {
        return Vector3F(vec.x(), vec.y(), vec.z());
    }

} // namespace KIRI
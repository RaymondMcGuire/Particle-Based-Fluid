/*** 
 * @Author: Xu.WANG
 * @Date: 2020-11-04 03:24:07
 * @LastEditTime: 2020-11-04 04:18:38
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriExamples\include\fbs\fbs_helper.h
 */
#include <fbs/generated/basic_types_generated.h>
#include <kiri_pbs_cuda/cuda_helper/helper_math.h>
namespace KIRI
{

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

    inline Vector3F FbsToKiri(const FlatBuffers::float3& vec)
    {
        return Vector3F(vec.x(), vec.y(), vec.z());
    }

} // namespace KIRI
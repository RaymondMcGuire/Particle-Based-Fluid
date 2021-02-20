/*** 
 * @Author: Xu.WANG
 * @Date: 2020-11-04 03:24:07
 * @LastEditTime: 2020-12-07 01:31:37
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriExamples\include\cuda\cuda_helper.h
 */
#include <kiri_pch.h>
#include <kiri_pbs_cuda/cuda_helper/helper_math.h>

namespace KIRI
{

    inline float3 KiriToCUDA(const Vector3F vec)
    {
        return make_float3(vec.x, vec.y, vec.z);
    }

    inline std::vector<float3> KiriToCUDA(const Array1Vec3F arr)
    {
        std::vector<float3> data;
        for (size_t i = 0; i < arr.size(); i++)
        {
            data.emplace_back(make_float3(arr[i].x, arr[i].y, arr[i].z));
        }
        return data;
    }

    inline std::vector<float4> KiriToCUDA(const Array1Vec4F arr)
    {
        std::vector<float4> data;
        for (size_t i = 0; i < arr.size(); i++)
        {
            data.emplace_back(make_float4(arr[i].x, arr[i].y, arr[i].z, arr[i].w));
        }
        return data;
    }

	inline std::vector<float3> KiriArrVec4FToVecFloat3(const Array1Vec4F arr)
    {
        std::vector<float3> data;
        for (size_t i = 0; i < arr.size(); i++)
        {

            data.emplace_back(make_float3(arr[i].x, arr[i].y, arr[i].z));
        }
        return data;
    }

    inline Array1Vec4F CUDAFloat3ToKiriVector4F(const std::vector<float3> arr)
    {
        Array1Vec4F data;
        for (size_t i = 0; i < arr.size(); i++)
        {
            data.append(Vector4F(arr[i].x, arr[i].y, arr[i].z, 0.1f));
        }
        return data;
    }

    inline std::vector<float3> KiriVertexToVecFloat3(const Array1<VertexFull> arr)
    {
        std::vector<float3> data;
        for (size_t i = 0; i < arr.size(); i++)
        {

            data.emplace_back(make_float3(arr[i].Position[0], arr[i].Position[1], arr[i].Position[2]));
        }
        return data;
    }

    inline std::vector<uint3> KiriIndicesToFaces(const Array1<UInt> arr)
    {
        std::vector<uint3> data;
        for (size_t i = 0; i < arr.size(); i += 3)
        {
            data.emplace_back(make_uint3(arr[i], arr[i + 1], arr[i + 2]));
        }
        return data;
    }

} // namespace KIRI
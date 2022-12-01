/*** 
 * @Author: Xu.WANG
 * @Date: 2020-07-26 17:30:04
 * @LastEditTime: 2020-10-18 02:19:17
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriPBSCuda\include\kiri_pbs_cuda\data\data_container.h
 */

#ifndef _DATA_CONTAINER_H_
#define _DATA_CONTAINER_H_

#include <vector>
#include <kiri_pbs_cuda/cuda_common.cuh>

class DataContainer
{
public:
    typedef std::vector<uint> IntegerDataContainer;
    typedef std::vector<float> ScalarDataContainer;
    typedef std::vector<float3> Vector3DataContainer;
    typedef std::vector<float4> Vector4DataContainer;

protected:
    uint addIntegerData(uint size = 0, uint initialVal = 0);
    uint addScalarData(uint size = 0, float initialVal = 0.f);
    uint addVector3Data(uint size = 0, const float3 &initialVal = make_float3(0.f));
    uint addVector4Data(uint size = 0, const float4 &initialVal = make_float4(0.f));

    std::vector<uint> &DataContainer::integerDataAt(uint idx);
    std::vector<float> &DataContainer::scalarDataAt(uint idx);
    std::vector<float3> &DataContainer::vector3DataAt(uint idx);
    std::vector<float4> &DataContainer::vector4DataAt(uint idx);

private:
    std::vector<IntegerDataContainer> _integerDataList;
    std::vector<ScalarDataContainer> _scalarDataList;
    std::vector<Vector3DataContainer> _vector3DataList;
    std::vector<Vector4DataContainer> _vector4DataList;
};

typedef std::shared_ptr<DataContainer>
    DataContainerPtr;

#endif /* _DATA_CONTAINER_H_ */
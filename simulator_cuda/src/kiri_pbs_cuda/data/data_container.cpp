/*** 
 * @Author: Xu.WANG
 * @Date: 2020-07-26 17:30:04
 * @LastEditTime: 2020-09-29 17:43:06
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriPBSCuda\\src\data\data_container.cpp
 */

#include <kiri_pbs_cuda/data/data_container.h>

uint DataContainer::addIntegerData(uint size, uint initialVal)
{
    uint attrIdx = _integerDataList.size();
    _integerDataList.emplace_back(size, initialVal);
    return attrIdx;
}

uint DataContainer::addScalarData(uint size, float initialVal)
{
    uint attrIdx = _scalarDataList.size();
    _scalarDataList.emplace_back(size, initialVal);
    return attrIdx;
}

uint DataContainer::addVector3Data(uint size, const float3 &initialVal)
{
    uint attrIdx = _vector3DataList.size();
    _vector3DataList.emplace_back(size, initialVal);
    return attrIdx;
}

uint DataContainer::addVector4Data(uint size, const float4 &initialVal)
{
    uint attrIdx = _vector4DataList.size();
    _vector4DataList.emplace_back(size, initialVal);
    return attrIdx;
}

std::vector<uint> &DataContainer::integerDataAt(
    uint idx)
{
    return _integerDataList[idx];
}

std::vector<float> &DataContainer::scalarDataAt(
    uint idx)
{
    return _scalarDataList[idx];
}

std::vector<float3> &DataContainer::vector3DataAt(
    uint idx)
{
    return _vector3DataList[idx];
}

std::vector<float4> &DataContainer::vector4DataAt(
    uint idx)
{
    return _vector4DataList[idx];
}
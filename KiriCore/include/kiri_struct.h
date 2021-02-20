/*** 
 * @Author: Xu.WANG
 * @Date: 2021-02-20 01:47:50
 * @LastEditTime: 2021-02-20 02:02:12
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_struct.h
 */

#ifndef _KIRI_STRUCT_H_
#define _KIRI_STRUCT_H_

#pragma once

enum DataType
{
    Standard = 1,
    Full = 2,
    Quad2 = 3,
    Quad3 = 4
};

struct VertexQuad2
{
    // position
    float Position[3];
    // texCoords
    float TexCoords[2];
};

struct VertexQuad3
{
    // position
    float Position[3];
    // position
    float Color[3];
    // texCoords
    float TexCoords[2];
};

struct VertexStandard
{
    // position
    float Position[3];
    // normal
    float Normal[3];
    // texCoords
    float TexCoords[2];
};

struct VertexFull
{
    // position
    float Position[3];
    // normal
    float Normal[3];
    // texCoords
    float TexCoords[2];
    // tangent
    float Tangent[3];
    //bitangent
    float Bitangent[3];
};

struct InstanceMat4x4
{
    float value[16];
};

enum ShadowType
{
    PointShadow = 1,
    ShadowMapping = 2
};

struct Texture
{
    UInt id;
    String type;
    String path;
};

#endif
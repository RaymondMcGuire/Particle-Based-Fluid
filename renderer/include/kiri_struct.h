/***
 * @Author: Xu.WANG
 * @Date: 2021-02-20 01:47:50
 * @LastEditTime: 2021-04-08 15:50:37
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath: \Kiri\KiriCore\include\kiri_struct.h
 */

#ifndef _KIRI_STRUCT_H_
#define _KIRI_STRUCT_H_

#pragma once

enum DataType
{
    Simple = 1,
    Standard = 2,
    Full = 3,
    Quad2 = 4,
    Quad3 = 5
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
    // color
    float Color[3];
    // texCoords
    float TexCoords[2];
};

struct VertexSimple
{
    // position
    float Position[3];
    // color
    float Color[3];
};

struct VertexStandard
{
    // position
    float Position[3];
    // mNormal
    float Normal[3];
    // texCoords
    float TexCoords[2];
};

struct VertexFull
{
    // position
    float Position[3];
    // mNormal
    float Normal[3];
    // texCoords
    float TexCoords[2];
    // tangent
    float Tangent[3];
    // bitangent
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

struct Rect3
{
    Vector3F original;
    Vector3F size;

    Rect3()
        : original(Vector3F(0.f)), size(Vector3F(0.f)) {}

    Rect3(
        Vector3F original,
        Vector3F size)
        : original(original),
          size(size) {}
};

#endif
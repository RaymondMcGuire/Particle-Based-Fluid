/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:26
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 11:16:09
 * @FilePath: \Kiri\core\include\kiri_utils.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#ifndef _KIRI_UTILS_H_
#define _KIRI_UTILS_H_

#pragma once

#include <kiri_pch.h>
#include <tiny_obj_loader.h>

class KiriUtils
{
public:
    static String UInt2Str4Digit(UInt Input);

    static UInt LoadTexture(char const *, bool = false);
    static UInt LoadTexture(const char *, const String &, bool = false);

    static void ExportBgeoFileFromCPU(String Folder, String FileName, Array1Vec4F Positions);

    static void ExportBgeoFileFromCPU(String Folder, String FileName, Array1Vec4F Positions, Array1<float> Mass);

    static void WriteMultiSizedParticles(String Folder, String FileName, Array1Vec4F Positions, Vec_Float masses);

    static Array1Vec4F ReadBgeoFileForCPU(String Folder, String Name, Vector3F Offset = Vector3F(0.f), bool FlipYZ = false);

    static void ReadParticlesData(Vec_Vec4F &positions, Vec_Float &masses, String Folder, String Name, Vector3F Offset = Vector3F(0.f), bool FlipYZ = false);

    static bool ReadTetFile(const String &filename, Vec_Float &vertices, Vec_Int &mIndices, bool normalize = true, float scale = 1000.f);

    static void ExportXYFile(const Array1Vec2F &points, const String fileName, bool normlization = true);
    static void ExportXYZFile(const Array1Vec3F &points, const String fileName);

    static void LoadVoroFile(
        std::vector<Vector3F> &position,
        std::vector<Vector3F> &mNormal,
        std::vector<int> &mIndices,
        std::string file_name);

    static String GetDefaultExportPath();

    static bool TinyObjWriter(const String &filename, const tinyobj::attrib_t &attributes, const Vector<tinyobj::shape_t> &shapes, const Vector<tinyobj::material_t> &materials, bool coordTransform = false);
};

#endif
/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-18 00:18:57
 * @LastEditTime: 2021-02-20 18:38:53
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_utils.h
 */

#ifndef _KIRI_UTILS_H_
#define _KIRI_UTILS_H_

#pragma once

#include <kiri_pch.h>
#include <kiri_pbs_cuda/cuda_helper/helper_math.h>

class KiriUtils
{
public:
    static String UInt2Str4Digit(UInt Input);

    static UInt loadTexture(char const *, bool = false);
    static UInt loadTexture(const char *, const String &, bool = false);

    static void printMatrix4x4F(Matrix4x4F, String = "");

    static void flipVertically(Int width, Int height, char *data);
    static Int saveScreenshot(const char *filename);
    static const char *createScreenshotBasename();
    static const char *createBasenameForVideo(Int cnt, const char *ext, const char *prefix);
    static Int captureScreenshot(Int cnt);

    static void ExportBgeoFileFromCPU(String Folder, String FileName, Array1Vec4F Positions);
    static void ExportBgeoFileFromGPU(String Folder, String FileName, float4 *Positions, float4 *Colors, uint *Labels, UInt NumOfParticles);
    static std::vector<float4> ReadBgeoFileForGPU(String Folder, String Name, bool FlipYZ = false);
    static std::vector<float4> ReadMultiBgeoFilesForGPU(String Folder, Vec_String Names, bool FlipYZ = false);
    static Array1Vec4F ReadBgeoFileForCPU(String Folder, String Name, Vector3F Offset = Vector3F(0.f), bool FlipYZ = false);
};

#endif
/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:15:33
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\material_equirect2cube.h
 */

#ifndef _KIRI_MATERIAL_EQUIRECT2CUBE_H_
#define _KIRI_MATERIAL_EQUIRECT2CUBE_H_
#pragma once

#include <kiri_core/material/material.h>

class KiriMaterialEquirectangular2CubeMap : public KiriMaterial
{
public:
    KiriMaterialEquirectangular2CubeMap(UInt, Matrix4x4F);

    void Setup() override;
    void Update() override;

private:
    UInt hdrTexture;
    Matrix4x4F mCaptureProjection;
};
typedef SharedPtr<KiriMaterialEquirectangular2CubeMap> KiriMaterialEquirectangular2CubeMapPtr;
#endif
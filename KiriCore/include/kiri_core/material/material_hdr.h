/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:14:59
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\material_hdr.h
 */

#ifndef _KIRI_MATERIAL_HDR_H_
#define _KIRI_MATERIAL_HDR_H_
#pragma once
#include <kiri_core/material/material.h>

class KiriMaterialHDR : public KiriMaterial
{
public:
    KiriMaterialHDR(bool);
    void Update() override;

    void SetSceneBuffer(UInt);
    void SetBloomBuffer(UInt);
    void SetBloom(bool);

    void SetExposure(float);
    void SetHDR(bool);

private:
    void Setup() override;
    bool bloom;

    bool hdr;
    float exposure = 1.0f;
    UInt sceneBuffer;
    UInt bloomBuffer;
};

typedef SharedPtr<KiriMaterialHDR> KiriMaterialHDRPtr;
#endif
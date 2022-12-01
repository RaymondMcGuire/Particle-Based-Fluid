/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:03:18
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\material_ssao.h
 */

#ifndef _KIRI_MATERIAL_SSAO_H_
#define _KIRI_MATERIAL_SSAO_H_
#pragma once
#include <kiri_core/material/material.h>

class KiriMaterialSSAO : public KiriMaterial
{
public:
    KiriMaterialSSAO(Array1Vec3F);
    void Update() override;

private:
    void Setup() override;

    Array1Vec3F mKernel;
};

typedef SharedPtr<KiriMaterialSSAO> KiriMaterialSSAOPtr;
#endif
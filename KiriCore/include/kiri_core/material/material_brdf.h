/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:16:11
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\material_brdf.h
 */

#ifndef _KIRI_MATERIAL_BRDF_H_
#define _KIRI_MATERIAL_BRDF_H_
#pragma once
#include <kiri_core/material/material.h>

class KiriMaterialBRDF : public KiriMaterial
{
public:
    KiriMaterialBRDF();

    void Setup() override;
    void Update() override;
};
typedef SharedPtr<KiriMaterialBRDF> KiriMaterialBRDFPtr;
#endif
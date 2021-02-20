/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:14:53
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\material_ibl.h
 */

#ifndef _KIRI_MATERIAL_IBL_H_
#define _KIRI_MATERIAL_IBL_H_
#pragma once
#include <kiri_core/material/material.h>

class KiriMaterialIBL : public KiriMaterial
{
public:
    KiriMaterialIBL(UInt);

    void Setup() override;
    void Update() override;

private:
    UInt envMap;
};
typedef SharedPtr<KiriMaterialIBL> KiriMaterialIBLPtr;
#endif
/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:14:43
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\material_instancing_obj_demo.h
 */

#ifndef _KIRI_MATERIAL_INSTANCING_OBJ_DEMO_H_
#define _KIRI_MATERIAL_INSTANCING_OBJ_DEMO_H_
#pragma once
#include <kiri_core/material/material.h>

class KiriMaterialInstancingObjDemo : public KiriMaterial
{
public:
    KiriMaterialInstancingObjDemo();

    void Setup() override;
    void Update() override;
};

typedef SharedPtr<KiriMaterialInstancingObjDemo> KiriMaterialInstancingObjDemoPtr;
#endif
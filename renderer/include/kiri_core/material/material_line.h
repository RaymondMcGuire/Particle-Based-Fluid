/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-04-07 17:17:30
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\include\kiri_core\material\material_line.h
 */

#ifndef _KIRI_MATERIAL_LINE_H_
#define _KIRI_MATERIAL_LINE_H_
#pragma once
#include <kiri_core/material/material.h>

class KiriMaterialLine : public KiriMaterial
{
public:
    KiriMaterialLine();

    void Setup() override;
    void Update() override;

private:
    Vector3F mLineColor;
};
typedef SharedPtr<KiriMaterialLine> KiriMaterialLinePtr;
#endif
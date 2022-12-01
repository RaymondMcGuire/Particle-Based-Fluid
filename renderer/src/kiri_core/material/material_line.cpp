/*** 
 * @Author: Xu.WANG
 * @Date: 2021-04-07 14:29:18
 * @LastEditTime: 2021-04-08 15:58:25
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\src\kiri_core\material\material_line.cpp
 */

#include <kiri_core/material/material_line.h>

void KiriMaterialLine::Setup()
{
    KiriMaterial::Setup();
    BindGlobalUniformBufferObjects();
    mShader->Use();
}

void KiriMaterialLine::Update()
{
    mShader->Use();
}

KiriMaterialLine::KiriMaterialLine()
{
    mName = "line";
    Setup();
}
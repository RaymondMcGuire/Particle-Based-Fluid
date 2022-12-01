/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 18:46:08
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\src\kiri_core\material\material_shadow_depth.cpp
 */

#include <kiri_core/material/material_shadow_depth.h>

void KiriMaterialShadowDepth::SetLightSpaceMatrix(Matrix4x4F lightSpaceMatrix)
{
    mLightSpaceMatrix = lightSpaceMatrix;
}

void KiriMaterialShadowDepth::Setup()
{
    KiriMaterial::Setup();
    mShader->Use();
    mShader->SetMat4("mLightSpaceMatrix", mLightSpaceMatrix);
}

void KiriMaterialShadowDepth::Update()
{
    mShader->Use();
    mShader->SetMat4("mLightSpaceMatrix", mLightSpaceMatrix);
}

KiriMaterialShadowDepth::KiriMaterialShadowDepth()
{
    mName = "shadow_depth";
    Setup();
}

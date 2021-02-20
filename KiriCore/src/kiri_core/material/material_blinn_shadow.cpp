/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:16:41
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\src\kiri_core\material\material_blinn_shadow.cpp
 */

#include <kiri_core/material/material_blinn_shadow.h>
#include <kiri_utils.h>
#include <kiri_core/kiri_loadfiles.h>

void KiriMaterialBlinnShadow::Setup()
{
    KiriMaterial::Setup();
    BindGlobalUniformBufferObjects();
    mShader->Use();
    mShader->SetInt("diffuseTexture", 0);
    mShader->SetInt("shadowMap", 1);
}

void KiriMaterialBlinnShadow::Update()
{
    Vector3F mLightPos(-2.0f, 4.0f, -1.0f);
    mShader->Use();
    mShader->SetVec3("mLightPos", mLightPos);
    mShader->SetMat4("mLightSpaceMatrix", shadow->getLightSpaceMatrix());

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, shadow->getDepthMap());
}

KiriMaterialBlinnShadow::KiriMaterialBlinnShadow(KiriShadowMapping *_shadow)
{
    mName = "blinn_shadow";
    shadow = _shadow;
    texture = KiriUtils::loadTexture(KiriLoadFiles::getPath("resources/textures/wood.png").c_str());
    Setup();
}

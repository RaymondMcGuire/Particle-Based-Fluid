/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-19 23:45:29
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\src\kiri_core\material\material.cpp
 */

#include <kiri_core/material/material.h>

void KiriMaterial::Setup()
{
    if (mGeometryShaderEnbale)
		mShader = new KiriShader(mName + ".vs", mName + ".fs", mName + ".gs");
    else
        mShader = new KiriShader(mName + ".vs", mName + ".fs");
}

void KiriMaterial::BindGlobalUniformBufferObjects()
{
    UInt uniformBlockIndex = glGetUniformBlockIndex(mShader->ID, "Matrices");
    glUniformBlockBinding(mShader->ID, uniformBlockIndex, 0);
}

KiriShader *KiriMaterial::GetShader()
{
    return mShader;
}

String KiriMaterial::GetShaderName()
{
    return mName;
}
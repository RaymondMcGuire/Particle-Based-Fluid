/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-20 19:25:10 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-03-24 22:14:30
 */
#include <kiri_core/shadow/point_shadow.h>

KiriPointShadow::KiriPointShadow()
{
    depthMapFBO = depthCubeMap = 0;
}

KiriMaterialPtr KiriPointShadow::getShadowDepthMaterial()
{
    return pointShadowDepthMaterial;
}

UInt KiriPointShadow::getDepthCubeMap()
{
    return depthCubeMap;
}

void KiriPointShadow::enable(Vector3F _pointLight)
{
    pointLight = _pointLight;

    glGenFramebuffers(1, &depthMapFBO);

    // create depth cubemap texture
    glGenTextures(1, &depthCubeMap);
    glBindTexture(GL_TEXTURE_CUBE_MAP, depthCubeMap);
    for (UInt i = 0; i < 6; ++i)
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_DEPTH_COMPONENT, SHADOW_WIDTH, SHADOW_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    // attach depth texture as FBO's depth buffer
    glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depthCubeMap, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // init depth shader
    pointShadowDepthMaterial = std::make_shared<KiriMaterialPointShadowDepth>();
}

void KiriPointShadow::bind()
{
    //reset
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // create depth cubemap transformation matrices
    Matrix4x4F shadowProj = Matrix4x4F::perspectiveMatrix(90.0f, (float)SHADOW_WIDTH / (float)SHADOW_HEIGHT, near_plane, mFarPlane);
    Array1<Matrix4x4F> mShadowTransforms;
    mShadowTransforms.append(shadowProj * Matrix4x4F::viewMatrix(pointLight, pointLight + Vector3F(1.0f, 0.0f, 0.0f), Vector3F(0.0f, -1.0f, 0.0f)));
    mShadowTransforms.append(shadowProj * Matrix4x4F::viewMatrix(pointLight, pointLight + Vector3F(-1.0f, 0.0f, 0.0f), Vector3F(0.0f, -1.0f, 0.0f)));
    mShadowTransforms.append(shadowProj * Matrix4x4F::viewMatrix(pointLight, pointLight + Vector3F(0.0f, 1.0f, 0.0f), Vector3F(0.0f, 0.0f, 1.0f)));
    mShadowTransforms.append(shadowProj * Matrix4x4F::viewMatrix(pointLight, pointLight + Vector3F(0.0f, -1.0f, 0.0f), Vector3F(0.0f, 0.0f, -1.0f)));
    mShadowTransforms.append(shadowProj * Matrix4x4F::viewMatrix(pointLight, pointLight + Vector3F(0.0f, 0.0f, 1.0f), Vector3F(0.0f, -1.0f, 0.0f)));
    mShadowTransforms.append(shadowProj * Matrix4x4F::viewMatrix(pointLight, pointLight + Vector3F(0.0f, 0.0f, -1.0f), Vector3F(0.0f, -1.0f, 0.0f)));

    // render scene to depth cubemap
    glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
    glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
    glClear(GL_DEPTH_BUFFER_BIT);

    pointShadowDepthMaterial->SetFarPlane(mFarPlane);
    pointShadowDepthMaterial->SetLightPos(pointLight);
    pointShadowDepthMaterial->SetShadowTransforms(mShadowTransforms);
    pointShadowDepthMaterial->Update();
}

void KiriPointShadow::release()
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
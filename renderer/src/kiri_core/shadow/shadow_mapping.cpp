/*
 * @Author: Xu.Wang
 * @Date: 2020-03-17 18:00:55
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-03-24 22:25:58
 */
#include <kiri_core/shadow/shadow_mapping.h>

Matrix4x4F KiriShadowMapping::getLightSpaceMatrix()
{
    return mLightSpaceMatrix;
}

UInt KiriShadowMapping::getDepthMap()
{
    return depthMap;
}

KiriShadowMapping::KiriShadowMapping()
{
    depthMapFBO = depthMap = 0;
}

KiriMaterialPtr KiriShadowMapping::GetShadowDepthMaterial()
{
    return shadowDepthMaterial;
}

void KiriShadowMapping::Enable(Vector3F _directionLight)
{
    directionLight = _directionLight;
    // generate depth map FBO
    glGenFramebuffers(1, &depthMapFBO);

    // create depth texture
    glGenTextures(1, &depthMap);
    glBindTexture(GL_TEXTURE_2D, depthMap);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, SHADOW_WIDTH, SHADOW_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    float borderColor[] = {1.0, 1.0, 1.0, 1.0};
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);

    // attach depth texture as FBO's depth buffer
    glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // init depth shader
    shadowDepthMaterial = std::make_shared<KiriMaterialShadowDepth>();
}

void KiriShadowMapping::Bind()
{
    glCullFace(GL_FRONT);
    // reset
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Render depth of scene to texture (from light's perspective)
    Matrix4x4F lightProjection, lightView;
    float mNearPlane = 1.0f, mFarPlane = 7.5f;
    lightProjection = Matrix4x4F::orthoMatrix(-10.0f, 10.0f, -10.0f, 10.0f, mNearPlane, mFarPlane);
    lightView = Matrix4x4F::viewMatrix(directionLight, Vector3F(0.0f), Vector3F(0.0f, 1.0f, 0.0f));
    mLightSpaceMatrix = lightProjection * lightView;

    // Render scene from light's point of view
    shadowDepthMaterial->SetLightSpaceMatrix(mLightSpaceMatrix);
    shadowDepthMaterial->Update();

    // Render depthmap to fb
    glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
    glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
    glClear(GL_DEPTH_BUFFER_BIT);
}

void KiriShadowMapping::Release()
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glCullFace(GL_BACK);
}
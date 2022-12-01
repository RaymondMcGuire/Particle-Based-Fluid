/*
 * @Author: Xu.Wang
 * @Date: 2020-03-20 19:14:39
 * @Last Modified by:   Xu.Wang
 * @Last Modified time: 2020-03-20 19:14:39
 */
#include <kiri_core/kiri_deferred_shading.h>

KiriDeferredShading::KiriDeferredShading(UInt _w, UInt _h)
{
    WINDOW_WIDTH = _w;
    WINDOW_HEIGHT = _h;

    mGBuffer = mGPosition = mGNormal = mGAlbedoSpec = mRBODepth = 0;
    mQuad = NULL;
    mGBuffer = NULL;
    mBlinnDefer = NULL;
    mSSAO = NULL;

    bSSAO = false;
}

void KiriDeferredShading::SetEntities(Array1<KiriEntityPtr> _entities)
{
    mEntities = _entities;
}

void KiriDeferredShading::SetPointLights(Array1<KiriPointLightPtr> _pointLights)
{
    mPointLights = _pointLights;
}

void KiriDeferredShading::SetUseNormalMap(bool _use_normal_map)
{
    mGBufferMat->SetUseNormalMap(_use_normal_map);
}

void KiriDeferredShading::SetUseSSAO(bool _use_ssao)
{
    bSSAO = _use_ssao;
    mBlinnDefer->SetSSAO(bSSAO);
}

void KiriDeferredShading::Enable(bool _b_ssao)
{
    bSSAO = _b_ssao;

    // generate gbuffer
    glGenFramebuffers(1, &mGBuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, mGBuffer);

    // position color buffer
    glGenTextures(1, &mGPosition);
    glBindTexture(GL_TEXTURE_2D, mGPosition);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mGPosition, 0);
    // mNormal color buffer
    glGenTextures(1, &mGNormal);
    glBindTexture(GL_TEXTURE_2D, mGNormal);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, mGNormal, 0);
    // color + specular color buffer
    glGenTextures(1, &mGAlbedoSpec);
    glBindTexture(GL_TEXTURE_2D, mGAlbedoSpec);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, mGAlbedoSpec, 0);
    // tell OpenGL which color mAttachments we'll use (of this framebuffer) for rendering
    UInt mAttachments[3] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2};
    glDrawBuffers(3, mAttachments);

    glGenRenderbuffers(1, &mRBODepth);
    glBindRenderbuffer(GL_RENDERBUFFER, mRBODepth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, WINDOW_WIDTH, WINDOW_HEIGHT);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, mRBODepth);
    // finally check if framebuffer is complete
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "mRBODepth Framebuffer not complete!" << std::endl;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    mGBufferMat = std::make_shared<KiriMaterialGBuffer>();
    mBlinnDefer = std::make_shared<KiriMaterialBlinnDefer>(bSSAO);
    mBlinnDefer->SetPointLights(mPointLights);
    mQuad = std::make_shared<KiriQuad>();
    mQuad->SetMaterial(mBlinnDefer);

    if (bSSAO)
    {
        mSSAO = new KiriSSAO(WINDOW_WIDTH, WINDOW_HEIGHT);
        mSSAO->Enable();
    }
}

void KiriDeferredShading::DawGeometryPass()
{
    BindGeometryPass();
    mEntities.forEach([=](KiriEntityPtr _entity)
                      {
        mGBufferMat->SetOutside(_entity->GetOutside());
        mGBufferMat->SetHaveNormalMap(_entity->GetNormalMap());
        _entity->GetModel()->SetMaterial(mGBufferMat);
        _entity->GetModel()->BindShader();
        _entity->GetModel()->Draw(); });
    Release();
}

void KiriDeferredShading::DrawLightingPass()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    mQuad->BindShader();
    // Bind texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mGPosition);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, mGNormal);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, mGAlbedoSpec);

    if (bSSAO)
    {
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, mSSAO->GetSSAOColorBuffer());
    }

    mQuad->Draw();
    Release();
}

void KiriDeferredShading::Render()
{
    DawGeometryPass();
    if (bSSAO)
    {
        mSSAO->Render(mGPosition, mGNormal);
    }
    DrawLightingPass();
}

void KiriDeferredShading::Release()
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
void KiriDeferredShading::BindGeometryPass()
{
    glBindFramebuffer(GL_FRAMEBUFFER, mGBuffer);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

KiriDeferredShading::~KiriDeferredShading()
{
}
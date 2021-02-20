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

    gBuffer = gPosition = gNormal = gAlbedoSpec = rboDepth = 0;
    quad = NULL;
    mGBuffer = NULL;
    mBlinnDefer = NULL;
    ssao = NULL;

    b_ssao = false;
}

void KiriDeferredShading::SetEntities(Array1<KiriEntityPtr> _entities)
{
    entities = _entities;
}

void KiriDeferredShading::SetPointLights(Array1<KiriPointLightPtr> _pointLights)
{
    pointLights = _pointLights;
}

void KiriDeferredShading::SetUseNormalMap(bool _use_normal_map)
{
    mGBuffer->SetUseNormalMap(_use_normal_map);
}

void KiriDeferredShading::SetUseSSAO(bool _use_ssao)
{
    b_ssao = _use_ssao;
    mBlinnDefer->SetSSAO(b_ssao);
}

void KiriDeferredShading::enable(bool _b_ssao)
{
    b_ssao = _b_ssao;

    //generate gbuffer
    glGenFramebuffers(1, &gBuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, gBuffer);

    // position color buffer
    glGenTextures(1, &gPosition);
    glBindTexture(GL_TEXTURE_2D, gPosition);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gPosition, 0);
    // normal color buffer
    glGenTextures(1, &gNormal);
    glBindTexture(GL_TEXTURE_2D, gNormal);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, gNormal, 0);
    // color + specular color buffer
    glGenTextures(1, &gAlbedoSpec);
    glBindTexture(GL_TEXTURE_2D, gAlbedoSpec);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, gAlbedoSpec, 0);
    // tell OpenGL which color attachments we'll use (of this framebuffer) for rendering
    UInt attachments[3] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2};
    glDrawBuffers(3, attachments);

    glGenRenderbuffers(1, &rboDepth);
    glBindRenderbuffer(GL_RENDERBUFFER, rboDepth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, WINDOW_WIDTH, WINDOW_HEIGHT);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rboDepth);
    // finally check if framebuffer is complete
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "rboDepth Framebuffer not complete!" << std::endl;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    mGBuffer = std::make_shared<KiriMaterialGBuffer>();
    mBlinnDefer = std::make_shared<KiriMaterialBlinnDefer>(b_ssao);
    mBlinnDefer->SetPointLights(pointLights);
    quad = std::make_shared<KiriQuad>();
    quad->SetMaterial(mBlinnDefer);

    if (b_ssao)
    {
        ssao = new KiriSSAO(WINDOW_WIDTH, WINDOW_HEIGHT);
        ssao->enable();
    }
}

void KiriDeferredShading::drawGeometryPass()
{
    bindGeometryPass();
    entities.forEach([=](KiriEntityPtr _entity) {
        mGBuffer->SetOutside(_entity->getOutside());
        mGBuffer->SetHaveNormalMap(_entity->getNormalMap());
        _entity->getModel()->SetMaterial(mGBuffer);
        _entity->getModel()->BindShader();
        _entity->getModel()->Draw();
    });
    release();
}

void KiriDeferredShading::drawLightingPass()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    quad->BindShader();
    //bind texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, gPosition);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, gNormal);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, gAlbedoSpec);

    if (b_ssao)
    {
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, ssao->getSSAOColorBuffer());
    }

    quad->Draw();
    release();
}

void KiriDeferredShading::render()
{
    drawGeometryPass();
    if (b_ssao)
    {
        ssao->render(gPosition, gNormal);
    }
    drawLightingPass();
}

void KiriDeferredShading::release()
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
void KiriDeferredShading::bindGeometryPass()
{
    glBindFramebuffer(GL_FRAMEBUFFER, gBuffer);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

KiriDeferredShading::~KiriDeferredShading()
{
}
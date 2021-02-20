/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-20 19:14:54 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-05-15 01:03:29
 */
#include <kiri_core/kiri_hdr.h>

KiriHDR::KiriHDR(UInt _w, UInt _h, bool _bloom)
{
    WINDOW_WIDTH = _w;
    WINDOW_HEIGHT = _h;
    bloom = _bloom;

    hdrFBO = rboDepth = 0;

    hdrMaterial = NULL;
    gaussBlurMaterial = NULL;
    quad = NULL;
}

void KiriHDR::enable()
{
    glGenFramebuffers(1, &hdrFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, hdrFBO);

    Int useBuffer = bloom ? 2 : 1;

    // create floating point color buffer
    glGenTextures(useBuffer, colorBuffers);

    for (Int i = 0; i < useBuffer; i++)
    {
        glBindTexture(GL_TEXTURE_2D, colorBuffers[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, colorBuffers[i], 0);
        attachments[i] = GL_COLOR_ATTACHMENT0 + i;
    }

    // create depth buffer (renderbuffer)
    glGenRenderbuffers(1, &rboDepth);
    glBindRenderbuffer(GL_RENDERBUFFER, rboDepth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, WINDOW_WIDTH, WINDOW_HEIGHT);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rboDepth);

    // attach buffers
    glDrawBuffers(useBuffer, attachments);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "Framebuffer not complete!" << std::endl;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    //bloom
    if (bloom)
    {
        glGenFramebuffers(2, blurFBO);
        glGenTextures(2, blurColorbuffers);
        for (Int i = 0; i < 2; i++)
        {
            glBindFramebuffer(GL_FRAMEBUFFER, blurFBO[i]);
            glBindTexture(GL_TEXTURE_2D, blurColorbuffers[i]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, blurColorbuffers[i], 0);
            // also check if framebuffers are complete (no need for depth buffer)
            if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
                std::cout << "Framebuffer not complete!" << std::endl;
        }
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    hdrMaterial = std::make_shared<KiriMaterialHDR>(bloom);
    hdrMaterial->SetSceneBuffer(colorBuffers[0]);
    gaussBlurMaterial = std::make_shared<KiriMaterialGaussianBlur>(colorBuffers[1]);
    quad = std::make_shared<KiriQuad>();
}

void KiriHDR::bindHDR()
{
    // render scene into floating point framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, hdrFBO);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void KiriHDR::renderBloom()
{
    if (bloom)
    {
        // blur bright fragments with two-pass Gaussian Blur
        bool horizontal = true, first_iteration = true;
        UInt amount = 10;
        quad->SetMaterial(gaussBlurMaterial);
        quad->BindShader();
        for (UInt i = 0; i < amount; i++)
        {
            glBindFramebuffer(GL_FRAMEBUFFER, blurFBO[horizontal]);
            gaussBlurMaterial->SetHorizontal(horizontal);
            glBindTexture(GL_TEXTURE_2D, first_iteration ? colorBuffers[1] : blurColorbuffers[!horizontal]);
            quad->Draw();
            horizontal = !horizontal;
            if (first_iteration)
                first_iteration = false;
        }

        hdrMaterial->SetBloomBuffer(blurColorbuffers[!horizontal]);
    }
}

void KiriHDR::release()
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void KiriHDR::SetBloom(bool _bloom)
{
    bloom = _bloom;
    hdrMaterial->SetBloom(bloom);
}

void KiriHDR::SetExposure(float _exposure)
{
    hdrMaterial->SetExposure(_exposure);
}

void KiriHDR::SetHDR(bool _hdr)
{
    hdrMaterial->SetHDR(_hdr);
}

void KiriHDR::renderToScreen()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    quad->SetMaterial(hdrMaterial);
    quad->BindShader();
    quad->Draw();
}
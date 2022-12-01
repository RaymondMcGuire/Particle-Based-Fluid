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
    mEnableBloom = _bloom;

    mHDRFBO = mRBODepth = 0;

    mHDRMaterial = NULL;
    mGaussBlurMaterial = NULL;
    mQuad = NULL;
}

void KiriHDR::Enable()
{
    glGenFramebuffers(1, &mHDRFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, mHDRFBO);

    Int useBuffer = mEnableBloom ? 2 : 1;

    // create floating point color buffer
    glGenTextures(useBuffer, mColorBuffers);

    for (Int i = 0; i < useBuffer; i++)
    {
        glBindTexture(GL_TEXTURE_2D, mColorBuffers[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, mColorBuffers[i], 0);
        mAttachments[i] = GL_COLOR_ATTACHMENT0 + i;
    }

    // create depth buffer (renderbuffer)
    glGenRenderbuffers(1, &mRBODepth);
    glBindRenderbuffer(GL_RENDERBUFFER, mRBODepth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, WINDOW_WIDTH, WINDOW_HEIGHT);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, mRBODepth);

    // attach buffers
    glDrawBuffers(useBuffer, mAttachments);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "Framebuffer not complete!" << std::endl;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // mEnableBloom
    if (mEnableBloom)
    {
        glGenFramebuffers(2, mBlurFBO);
        glGenTextures(2, mBlurColorBuffers);
        for (Int i = 0; i < 2; i++)
        {
            glBindFramebuffer(GL_FRAMEBUFFER, mBlurFBO[i]);
            glBindTexture(GL_TEXTURE_2D, mBlurColorBuffers[i]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mBlurColorBuffers[i], 0);
            // also check if framebuffers are complete (no need for depth buffer)
            if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
                std::cout << "Framebuffer not complete!" << std::endl;
        }
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    mHDRMaterial = std::make_shared<KiriMaterialHDR>(mEnableBloom);
    mHDRMaterial->SetSceneBuffer(mColorBuffers[0]);
    mGaussBlurMaterial = std::make_shared<KiriMaterialGaussianBlur>(mColorBuffers[1]);
    mQuad = std::make_shared<KiriQuad>();
}

void KiriHDR::BindHDR()
{
    // Render scene into floating point framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, mHDRFBO);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void KiriHDR::RenderBloom()
{
    if (mEnableBloom)
    {
        // Blur bright fragments with two-pass Gaussian Blur
        bool horizontal = true, first_iteration = true;
        UInt amount = 10;
        mQuad->SetMaterial(mGaussBlurMaterial);
        mQuad->BindShader();
        for (UInt i = 0; i < amount; i++)
        {
            glBindFramebuffer(GL_FRAMEBUFFER, mBlurFBO[horizontal]);
            mGaussBlurMaterial->SetHorizontal(horizontal);
            glBindTexture(GL_TEXTURE_2D, first_iteration ? mColorBuffers[1] : mBlurColorBuffers[!horizontal]);
            mQuad->Draw();
            horizontal = !horizontal;
            if (first_iteration)
                first_iteration = false;
        }

        mHDRMaterial->SetBloomBuffer(mBlurColorBuffers[!horizontal]);
    }
}

void KiriHDR::Release()
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void KiriHDR::SetBloom(bool _bloom)
{
    mEnableBloom = _bloom;
    mHDRMaterial->SetBloom(mEnableBloom);
}

void KiriHDR::SetExposure(float _exposure)
{
    mHDRMaterial->SetExposure(_exposure);
}

void KiriHDR::SetHDR(bool _hdr)
{
    mHDRMaterial->SetHDR(_hdr);
}

void KiriHDR::RenderToScreen()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    mQuad->SetMaterial(mHDRMaterial);
    mQuad->BindShader();
    mQuad->Draw();
}
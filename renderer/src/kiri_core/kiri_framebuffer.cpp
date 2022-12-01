/***
 * @Author: Xu.WANG
 * @Date: 2020-10-24 20:04:09
 * @LastEditTime: 2020-11-25 21:56:03
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath: \Kiri\KiriCore\src\kiri_core\kiri_framebuffer.cpp
 */

#include <kiri_core/kiri_framebuffer.h>

KiriFrameBuffer::KiriFrameBuffer(UInt mWidth, UInt height)
{
    mWidth = mWidth;
    mHeight = height;

    // init framebuffer
    glGenFramebuffers(1, &mFrameBuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, mFrameBuffer);

    // color texture
    glGenTextures(1, &mTextureColorBuffer);
    glBindTexture(GL_TEXTURE_2D, mTextureColorBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, mWidth, mHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mTextureColorBuffer, 0);

    // create a renderbuffer object
    glGenRenderbuffers(1, &mRenderBufferObject);
    glBindRenderbuffer(GL_RENDERBUFFER, mRenderBufferObject);
    // glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, mWidth, mHeight);
    // glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, mRenderBufferObject);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, mWidth, mHeight);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, mRenderBufferObject);

    glGenTextures(1, &mTextureDepthBuffer);
    glBindTexture(GL_TEXTURE_2D, mTextureDepthBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, mWidth, mHeight, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, mTextureDepthBuffer, 0);
    // glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_COMPONENT, GL_TEXTURE_2D, mTextureDepthBuffer, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, mTextureDepthBuffer, 0);

    // GLenum bufs[] = {GL_COLOR_ATTACHMENT0};
    // glDrawBuffers(1, bufs);
    // check
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        KIRI_LOG_ERROR("ERROR::FRAMEBUFFER:: KIRI Framebuffer is not complete!");
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // init screen mQuad
    mScreenShader = std::make_shared<KiriMaterialScreen>();
    float quadVertices[] = {
        // positions   // texCoords
        -1.0f, 1.0f, 0.0f, 1.0f,
        -1.0f, -1.0f, 0.0f, 0.0f,
        1.0f, -1.0f, 1.0f, 0.0f,

        -1.0f, 1.0f, 0.0f, 1.0f,
        1.0f, -1.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 1.0f, 1.0f};
    glGenVertexArrays(1, &mQuadVAO);
    glGenBuffers(1, &mQuadVBO);
    glBindVertexArray(mQuadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, mQuadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));
}

void KiriFrameBuffer::Release()
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void KiriFrameBuffer::Bind()
{
    glBindFramebuffer(GL_FRAMEBUFFER, mFrameBuffer);
}

void KiriFrameBuffer::EnableDepthTest()
{
    glEnable(GL_DEPTH_TEST);
}

void KiriFrameBuffer::DisableDepthTest()
{
    glDisable(GL_DEPTH_TEST);
}

void KiriFrameBuffer::SetPostProcessingType(Int type)
{
    mScreenShader->SetPostProcessingType(type);
}

void KiriFrameBuffer::RenderToScreen()
{
    mScreenShader->Update();
    glBindVertexArray(mQuadVAO);
    glBindTexture(GL_TEXTURE_2D, mTextureColorBuffer);
    // glBindTexture(GL_TEXTURE_2D, mTextureDepthBuffer);

    glDrawArrays(GL_TRIANGLES, 0, 6);
}
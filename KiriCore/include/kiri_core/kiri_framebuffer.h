/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-24 20:04:09
 * @LastEditTime: 2021-02-20 19:40:47
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\kiri_framebuffer.h
 */

#ifndef _KIRI_FRAMEBUFFER_H_
#define _KIRI_FRAMEBUFFER_H_
#pragma once

#include <kiri_core/material/material_screen.h>

class KiriFrameBuffer
{
public:
    KiriFrameBuffer(UInt Width, UInt Height);

    void Bind();
    void Release();

    void EnableDepthTest();
    void DisableDepthTest();

    void RenderToScreen();

    void SetPostProcessingType(Int Type);

    inline UInt TextureColorBuffer() const { return mTextureColorBuffer; }
    inline UInt TextureDepthBuffer() const { return mTextureDepthBuffer; }
    inline UInt FrameBuffer() const { return mFrameBuffer; }

private:
    UInt mWidth, mHeight, mQuadVAO, mQuadVBO;
    KiriMaterialScreenPtr mScreenShader;

    UInt mFrameBuffer;
    UInt mTextureColorBuffer;
    UInt mTextureDepthBuffer;
    UInt mRenderBufferObject;
};

typedef SharedPtr<KiriFrameBuffer> KiriFrameBufferPtr;
#endif
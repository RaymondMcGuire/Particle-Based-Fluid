/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-24 20:04:09
 * @LastEditTime: 2021-02-20 19:30:07
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\texture\texture.h
 */

#ifndef _KIRI_TEXTURE_H_
#define _KIRI_TEXTURE_H_
#pragma once
#include <kiri_pch.h>

class KiriTexture
{
public:
    KiriTexture();
    virtual ~KiriTexture() {}

    KiriTexture(String Path, bool GammaCorrection = false, bool StbVertLoad = false);
    UInt Load();

private:
    String mPath;
    bool mStbVertLoad;
    UInt mTextureID;
    bool mGammaCorrection;

    Int mWidth, mHeight, mChannelComponents;
    UChar *mData;
};

typedef SharedPtr<KiriTexture> KiriTexturePtr;

#endif
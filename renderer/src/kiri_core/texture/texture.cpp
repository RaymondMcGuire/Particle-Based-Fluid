/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:26
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 15:16:33
 * @FilePath: \Kiri\renderer\src\kiri_core\texture\texture.cpp
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#include <kiri_core/texture/texture.h>
#include <stb_image.h>
#include <glad/glad.h>
KiriTexture::KiriTexture()
{
    glGenTextures(1, &mTextureID);
    mGammaCorrection = false;
    mStbVertLoad = false;
    mPath = "";
}

KiriTexture::KiriTexture(String Path, bool GammaCorrection, bool StbVertLoad)
{
    glGenTextures(1, &mTextureID);
    mGammaCorrection = GammaCorrection;
    mStbVertLoad = StbVertLoad;
    mPath = Path;
}

UInt KiriTexture::Load()
{
    if (mPath == "")
    {
        KIRI_LOG_ERROR("KiriTexture not defined the mPath:{0}", mPath);
        return -1;
    }
    stbi_set_flip_vertically_on_load(mStbVertLoad);
    mData = stbi_load(mPath.c_str(), &mWidth, &mHeight, &mChannelComponents, 0);
    if (mData)
    {
        GLenum internalFormat;
        GLenum mDataFormat;
        if (mChannelComponents == 1)
        {
            internalFormat = mDataFormat = GL_RED;
        }
        else if (mChannelComponents == 2)
        {
            internalFormat = mGammaCorrection ? GL_SRGB : GL_RG;
            mDataFormat = GL_RG;
        }
        else if (mChannelComponents == 3)
        {
            internalFormat = mGammaCorrection ? GL_SRGB : GL_RGB;
            mDataFormat = GL_RGB;
        }
        else if (mChannelComponents == 4)
        {
            internalFormat = mGammaCorrection ? GL_SRGB_ALPHA : GL_RGBA;
            mDataFormat = GL_RGBA;
        }

        glBindTexture(GL_TEXTURE_2D, mTextureID);
        glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, mWidth, mHeight, 0, mDataFormat, GL_UNSIGNED_BYTE, mData);
        glGenerateMipmap(GL_TEXTURE_2D);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, mDataFormat == GL_RGBA ? GL_CLAMP_TO_EDGE : GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, mDataFormat == GL_RGBA ? GL_CLAMP_TO_EDGE : GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        stbi_image_free(mData);
    }
    else
    {
        KIRI_LOG_ERROR("KiriTexture failed to load at path:{0}", mPath);
        stbi_image_free(mData);
    }

    return mTextureID;
}
/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-24 20:04:09
 * @LastEditTime: 2020-11-15 19:43:55
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\include\kiri_core\texture\pbr_texture.h
 */

#ifndef _KIRI_PBR_TEXTURE_H_
#define _KIRI_PBR_TEXTURE_H_
#pragma once
#include <kiri_core/texture/texture.h>

class KiriPBRTexture
{
public:
    KiriPBRTexture();
    KiriPBRTexture(String Name, bool GammaCorrection = false, String Folder = "materials", String Extension = ".png");

    inline void SetAlbedoType(String Type)
    {
        mAlbedoType = Type;
    }

    inline void SetMetallicType(String Type)
    {
        mMetallicType = Type;
    }

    inline void SetRoughnessType(String Type)
    {
        mRoughnessType = Type;
    }

    inline void SetAoType(String Type)
    {
        mAoType = Type;
    }

    inline void SetNormalType(String Type)
    {
        mNormalType = Type;
    }

    void Load();

    inline UInt Albedo() const { return mAlbedo; }
    inline UInt Metallic() const { return mMetallic; }
    inline UInt Roughness() const { return mRoughness; }
    inline UInt Ao() const { return mAo; }
    inline UInt Normal() const { return mNormal; }

private:
    String mName;
    String mPath;
    String mExtension;
    String mFolder;

    String mAlbedoType;
    String mMetallicType;
    String mRoughnessType;
    String mAoType;
    String mNormalType;

    UInt mAlbedo;
    UInt mMetallic;
    UInt mRoughness;
    UInt mAo;
    UInt mNormal;

    bool mGammaCorrection;
};

typedef SharedPtr<KiriPBRTexture> KiriPBRTexturePtr;

#endif
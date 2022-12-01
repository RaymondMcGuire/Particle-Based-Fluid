/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:26
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 15:02:36
 * @FilePath: \Kiri\renderer\src\kiri_core\texture\pbr_texture.cpp
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
/***
 * @Author: Xu.WANG
 * @Date: 2020-10-24 20:04:09
 * @LastEditTime: 2021-02-20 19:30:32
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath: \KiriCore\src\kiri_core\texture\pbr_texture.cpp
 */
#include <kiri_core/texture/pbr_texture.h>
#include <root_directory.h>

KiriPBRTexture::KiriPBRTexture() {}

KiriPBRTexture::KiriPBRTexture(String Name, bool GammaCorrection, String Folder, String Extension)
{
    mFolder = Folder;
    mExtension = Extension;
    mName = Name;

    if (RELEASE && PUBLISH)
    {
        // mPath = String(DB_PBR_PATH) + mFolder + "/" + mName + "/" + mName + "-";
        mPath = "./resources/" + mFolder + "/" + mName + "/" + mName + "-";
    }
    else
    {
        mPath = String(DB_PBR_PATH) + mFolder + "/" + mName + "/" + mName + "-";
    }

    mAlbedoType = "";
    mMetallicType = "";
    mRoughnessType = "";
    mAoType = "";
    mNormalType = "";

    mGammaCorrection = GammaCorrection;
}

void KiriPBRTexture::Load()
{
    KiriTexture albedoTex(mPath + "albedo" + mAlbedoType + mExtension, mGammaCorrection);
    KiriTexture metallicTex(mPath + "metallic" + mMetallicType + mExtension, mGammaCorrection);
    KiriTexture roughnessTex(mPath + "roughness" + mRoughnessType + mExtension, mGammaCorrection);
    KiriTexture aoTex(mPath + "ao" + mAoType + mExtension, mGammaCorrection);
    KiriTexture normalTex(mPath + "normal" + mNormalType + mExtension, mGammaCorrection);

    mAlbedo = albedoTex.Load();
    mMetallic = metallicTex.Load();
    mRoughness = roughnessTex.Load();
    mNormal = normalTex.Load();
    mAo = aoTex.Load();
}
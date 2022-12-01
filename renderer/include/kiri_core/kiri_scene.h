/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:26
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 13:48:55
 * @FilePath: \core\include\kiri_core\kiri_scene.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#ifndef _KIRI_SCENE_H_
#define _KIRI_SCENE_H_
#pragma once
#include <kiri_core/ecs/entity.h>
#include <kiri_core/shadow/kiri_shadow_define.h>
#include <kiri_core/kiri_hdr.h>
#include <kiri_core/kiri_deferred_shading.h>
#include <kiri_core/kiri_cube_skybox.h>

#include <kiri_core/camera/camera_fpc.h>
#include <kiri_core/particle/particle_render_system.h>

class KiriScene
{
public:
    KiriScene(){};
    KiriScene(UInt, UInt);
    void Clear();
    void Add(KiriEntityPtr);
    Array1<KiriEntityPtr> GetEntities()
    {
        return mEntities;
    }

    void Add(KiriPointLightPtr);
    Array1<KiriPointLightPtr> GetPointLights()
    {
        return mPointLights;
    }

    void AddDfs(KiriEntityPtr);
    void AddDfs(KiriPointLightPtr);

    void Render();

    void RenderShadow();
    void EnableShadow(ShadowType);
    KiriShadow *GetShadow()
    {
        return mShadow;
    }

    void RenderCubeSkybox();
    void EnableCubeSkybox(bool, String = "");
    KiriCubeSkyboxPtr GetCubeSkybox();

    void SetHDR(bool);
    void SetBloom(bool);
    void SetExposure(float);

    void EnableHDR();
    void BindHDR();
    void RenderHDR();
    KiriHDR *GetHDR();

    void EnableDeferredShading(bool);
    void RenderDF();

    void SetUseNormalMapDF(bool);
    void SetUseSSAO(bool);

private:
    UInt WINDOW_HEIGHT, WINDOW_WIDTH;

    Array1<KiriPointLightPtr> mPointLights;
    Array1<KiriEntityPtr> mEntities;

    KiriShadow *mShadow;
    bool mEnableShadow = false;

    KiriCubeSkyboxPtr mCubeSkybox;
    bool mLoadHDR = false;
    bool mEnableCubeSkybox = false;

    KiriHDR *mHDR;
    bool mEnableHDR = false;

    KiriDeferredShading *mDeferredShading;
    bool mEnableDeferredShading = false;

    Array1<KiriPointLightPtr> mDFSPointLights;
    Array1<KiriEntityPtr> mDFSEntities;
};

typedef SharedPtr<KiriScene> KiriScenePtr;
#endif
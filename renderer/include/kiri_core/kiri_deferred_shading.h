/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:26
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 13:59:38
 * @FilePath: \core\include\kiri_core\kiri_deferred_shading.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#ifndef _KIRI_DEFERRED_SHADING_H_
#define _KIRI_DEFERRED_SHADING_H_
#pragma once
#include <kiri_pch.h>
#include <kiri_core/ecs/entity.h>
#include <kiri_core/light/point_light.h>
#include <kiri_core/model/model_quad.h>
#include <kiri_core/material/material_g_buffer.h>
#include <kiri_core/material/material_blinn_defer.h>

#include <kiri_core/kiri_ssao.h>
class KiriDeferredShading
{
public:
    KiriDeferredShading(UInt, UInt);
    ~KiriDeferredShading();

    void SetEntities(Array1<KiriEntityPtr>);
    void SetPointLights(Array1<KiriPointLightPtr>);

    void Enable(bool);
    void Render();

    void SetUseNormalMap(bool);
    void SetUseSSAO(bool);

    UInt GetGBuffer()
    {
        return mGBuffer;
    }

private:
    UInt WINDOW_WIDTH, WINDOW_HEIGHT;

    UInt mGBuffer;
    UInt mGPosition, mGNormal, mGAlbedoSpec;
    UInt mRBODepth;

    KiriMaterialGBufferPtr mGBufferMat;
    void Release();
    void BindGeometryPass();

    KiriMaterialBlinnDeferPtr mBlinnDefer;
    KiriQuadPtr mQuad;

    Array1<KiriEntityPtr> mEntities;
    Array1<KiriPointLightPtr> mPointLights;
    void DawGeometryPass();
    void DrawLightingPass();

    KiriSSAO *mSSAO;
    bool bSSAO;
    void EnableSSAO();
};
#endif
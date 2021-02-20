/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:40:38
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\kiri_deferred_shading.h
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

    void enable(bool);
    void render();

    void SetUseNormalMap(bool);
    void SetUseSSAO(bool);

    UInt getGBuffer()
    {
        return gBuffer;
    }

private:
    UInt WINDOW_WIDTH, WINDOW_HEIGHT;

    UInt gBuffer;
    UInt gPosition, gNormal, gAlbedoSpec;
    UInt rboDepth;

    KiriMaterialGBufferPtr mGBuffer;
    void release();
    void bindGeometryPass();

    KiriMaterialBlinnDeferPtr mBlinnDefer;
    KiriQuadPtr quad;

    Array1<KiriEntityPtr> entities;
    Array1<KiriPointLightPtr> pointLights;
    void drawGeometryPass();
    void drawLightingPass();

    KiriSSAO *ssao;
    bool b_ssao;
    void enableSSAO();
};
#endif
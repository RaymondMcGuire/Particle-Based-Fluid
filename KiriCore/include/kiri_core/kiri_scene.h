/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-24 20:04:09
 * @LastEditTime: 2021-02-20 19:41:05
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\kiri_scene.h
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
    KiriScene(UInt, UInt, KIRI::KiriCameraPtr camera);
    void Clear();
    void add(KiriEntityPtr);
    Array1<KiriEntityPtr> getEntities()
    {
        return entities;
    }

    void add(KiriPointLightPtr);
    Array1<KiriPointLightPtr> getPointLights()
    {
        return pointLights;
    }

    void addDfs(KiriEntityPtr);
    void addDfs(KiriPointLightPtr);

    void render();

    void renderShadow();
    void enableShadow(ShadowType);
    KiriShadow *getShadow()
    {
        return shadow;
    }

    void renderCubeSkybox();
    void enableCubeSkybox(bool, String = "");
    KiriCubeSkyboxPtr getCubeSkybox();

    void SetHDR(bool);
    void SetBloom(bool);
    void SetExposure(float);

    void enableHDR();
    void bindHDR();
    void renderHDR();
    KiriHDR *getHDR();

    void enableDeferredShading(bool);
    void renderDF();

    void SetUseNormalMapDF(bool);
    void SetUseSSAO(bool);

    // particle system
    void SetParticlesWithRadius(ArrayAccessor1<Vector4F> particles);
    void SetParticles(ArrayAccessor1<Vector3F>, float);
    void SetParticlesVBO(UInt vbo, UInt num, float radius);
    void enableParticleRenderSystem(bool);

private:
    UInt WINDOW_HEIGHT, WINDOW_WIDTH;

    Array1<KiriPointLightPtr> pointLights;
    Array1<KiriEntityPtr> entities;

    KiriShadow *shadow;
    bool enable_shadow = false;

    KiriCubeSkyboxPtr cubeSkybox;
    bool load_hdr = false;
    bool enable_cubeSkybox = false;

    KiriHDR *hdr;
    bool enable_hdr = false;

    KiriDeferredShading *deferredShading;
    bool enable_deferred_shading = false;

    Array1<KiriPointLightPtr> dfsPointLights;
    Array1<KiriEntityPtr> dfsEntities;

    // particle render
    KIRI::KiriParticleRenderSystemPtr _particleRenderSys;
    bool _enable_particle_render = false;
};

typedef SharedPtr<KiriScene> KiriScenePtr;
#endif
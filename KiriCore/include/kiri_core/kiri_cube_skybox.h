/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:40:31
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\kiri_cube_skybox.h
 */

#ifndef _KIRI_CUBE_SKYBOX_H_
#define _KIRI_CUBE_SKYBOX_H_
#pragma once

#include <kiri_define.h>
#include <kiri_pch.h>
#include <root_directory.h>

#include <kiri_core/kiri_loadfiles.h>
#include <kiri_core/material/material.h>
#include <kiri_core/material/material_cube_skybox.h>

#include <kiri_core/material/material_ibl.h>
#include <kiri_core/material/material_equirect2cube.h>
#include <kiri_core/material/material_irradiance_convolution.h>
#include <kiri_core/material/material_spec_cubemap.h>
#include <kiri_core/material/material_brdf.h>

#include <kiri_core/model/model_cube.h>
#include <kiri_core/model/model_quad.h>

class KiriCubeSkybox
{
public:
    KiriCubeSkybox(String = "");
    KiriCubeSkybox(bool, String = "");
    ~KiriCubeSkybox();

    KiriMaterialPtr GetMaterial();
    void Draw();
    void drawIBL();

    UInt getEnvCubeMap()
    {
        return envCubeMap;
    }

    UInt getIrradianceCubeMap()
    {
        return irradianceMap;
    }

    UInt getSpecularCubeMap()
    {
        return specularEnvCubeMap;
    }

    UInt getBrdfLutTexture()
    {
        return brdfLUTTexture;
    }

private:
    const Int HDR_TEX_SIZE = 2048;
    const Int CUBE_TEX_SIZE = 6;
    Array1<String> cubeTexFile;

    bool load_hdr;
    UInt hdrTexture;

    KiriMaterialPtr material;
    UInt mVAO, mVBO;
    UInt envCubeMap;

    void Construct(String);
    void constructHDR(String);

    void BindShader();
    void SetMaterial(KiriMaterialPtr);

    // loads a cube skybox texture from 6 individual texture faces
    // order:
    // +X (right)
    // -X (left)
    // +Y (top)
    // -Y (bottom)
    // +Z (front)
    // -Z (back)
    // -------------------------------------------------------
    void createCubeMap();

    Array1<Matrix4x4F> captureViews;
    Matrix4x4F mCaptureProjection;
    void captureData2CubeMap();

    UInt captureFBO;
    UInt captureRBO;
    KiriCubePtr skyboxCube;
    KiriMaterialIBLPtr m_ibl;
    KiriMaterialEquirectangular2CubeMapPtr m_eq2cube;
    //convert HDR equirectangular environment map to cubemap equivalent
    void convertHDR2CubeMap();

    //generate mipmaps from first mip face (combatting visible dots artifact)
    void createCubeMapMipMap();

    //solve diffuse integral
    UInt irradianceMap;
    KiriMaterialIrradianceConvolutionPtr m_irrconv;
    void createIrradianceCubeMap();

    UInt specularEnvCubeMap;
    KiriMaterialSpecCubeMapPtr m_specubemap;
    void createSpecularCubeMap();
    void renderSpeclarCubeMap();

    UInt brdfLUTTexture;
    KiriQuadPtr brdfLUTQuad;
    KiriMaterialBRDFPtr m_brdf;
    void createBrdfLutTexure();
    void renderBrdfLutTexture();
};
typedef SharedPtr<KiriCubeSkybox> KiriCubeSkyboxPtr;
#endif
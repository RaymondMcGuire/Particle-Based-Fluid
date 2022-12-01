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
    void DrawIBL();

    UInt GetEnvCubeMap()
    {
        return mEnvCubeMapBuffer;
    }

    UInt GetIrradianceCubeMap()
    {
        return mIrradianceMapBuffer;
    }

    UInt GetSpecularCubeMap()
    {
        return mSpecularEnvCubeMapBuffer;
    }

    UInt GetBrdfLutTexture()
    {
        return mBRDFLUTTexture;
    }

private:
    const Int HDR_TEX_SIZE = 2048;
    const Int CUBE_TEX_SIZE = 6;
    Array1<String> mCubeTexFile;

    bool mLoadHDR;
    UInt mHDRTexture;

    KiriMaterialPtr mMat;
    UInt mVAO, mVBO;
    UInt mEnvCubeMapBuffer;

    void Construct(String);
    void ConstructHDR(String);

    void BindShader();
    void SetMaterial(KiriMaterialPtr);

    // order:
    // +X (right)
    // -X (left)
    // +Y (top)
    // -Y (bottom)
    // +Z (front)
    // -Z (back)
    void CreateCubeMap();
    void CaptureData2CubeMap();

    Array1<Matrix4x4F> mCaptureViews;
    Matrix4x4F mCaptureProjection;

    UInt mCaptureFBO;
    UInt mCaptureRBO;
    KiriCubePtr mSkyBox;
    KiriMaterialIBLPtr mIBLMat;
    KiriMaterialEquirectangular2CubeMapPtr mEq2CubeMat;

    void ConvertHDR2CubeMap();
    void CreateCubeMapMipMap();

    UInt mIrradianceMapBuffer;
    KiriMaterialIrradianceConvolutionPtr mIrrConvMat;
    void CreateIrradianceCubeMap();

    UInt mSpecularEnvCubeMapBuffer;
    KiriMaterialSpecCubeMapPtr mSpeCubeMapMat;
    void CreateSpecularCubeMap();
    void RenderSpeclarCubeMap();

    UInt mBRDFLUTTexture;
    KiriQuadPtr mBRDFfLUTQuad;
    KiriMaterialBRDFPtr mBRDFMat;
    void CreateBRDFLutTexure();
    void RenderBRDFLutTexture();
};
typedef SharedPtr<KiriCubeSkybox> KiriCubeSkyboxPtr;
#endif
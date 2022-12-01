/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:26
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 14:16:19
 * @FilePath: \core\src\kiri_core\kiri_cube_skybox.cpp
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#include <kiri_core/kiri_cube_skybox.h>
#include <stb_image.h>
#include <stb_image_write.h>
void KiriCubeSkybox::SetMaterial(KiriMaterialPtr _material)
{
    mMat = _material;
}

KiriMaterialPtr KiriCubeSkybox::GetMaterial() { return mMat; }

void KiriCubeSkybox::BindShader()
{
    mMat->Update();
}

void KiriCubeSkybox::Construct(String path)
{

    String default_path = "resources/mTextures/skybox/";
    if (path != "")
        default_path = path;

    Array1<float> skyboxVertices = {
        -1.0f, 1.0f, -1.0f,
        -1.0f, -1.0f, -1.0f,
        1.0f, -1.0f, -1.0f,
        1.0f, -1.0f, -1.0f,
        1.0f, 1.0f, -1.0f,
        -1.0f, 1.0f, -1.0f,

        -1.0f, -1.0f, 1.0f,
        -1.0f, -1.0f, -1.0f,
        -1.0f, 1.0f, -1.0f,
        -1.0f, 1.0f, -1.0f,
        -1.0f, 1.0f, 1.0f,
        -1.0f, -1.0f, 1.0f,

        1.0f, -1.0f, -1.0f,
        1.0f, -1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, -1.0f,
        1.0f, -1.0f, -1.0f,

        -1.0f, -1.0f, 1.0f,
        -1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, -1.0f, 1.0f,
        -1.0f, -1.0f, 1.0f,

        -1.0f, 1.0f, -1.0f,
        1.0f, 1.0f, -1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        -1.0f, 1.0f, 1.0f,
        -1.0f, 1.0f, -1.0f,

        -1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f, 1.0f,
        1.0f, -1.0f, -1.0f,
        1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f, 1.0f,
        1.0f, -1.0f, 1.0f};

    // Construct cube skybox
    glGenVertexArrays(1, &mVAO);
    glGenBuffers(1, &mVBO);
    glBindVertexArray(mVAO);
    glBindBuffer(GL_ARRAY_BUFFER, mVBO);
    glBufferData(GL_ARRAY_BUFFER, skyboxVertices.size() * sizeof(float), skyboxVertices.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);

    // load cube skybox texture
    mCubeTexFile.append(KiriLoadFiles::GetPath(default_path + "right.jpg"));
    mCubeTexFile.append(KiriLoadFiles::GetPath(default_path + "left.jpg"));
    mCubeTexFile.append(KiriLoadFiles::GetPath(default_path + "top.jpg"));
    mCubeTexFile.append(KiriLoadFiles::GetPath(default_path + "bottom.jpg"));
    mCubeTexFile.append(KiriLoadFiles::GetPath(default_path + "front.jpg"));
    mCubeTexFile.append(KiriLoadFiles::GetPath(default_path + "back.jpg"));

    CreateCubeMap();

    // set cube skybox mMat
    this->SetMaterial(std::make_shared<KiriMaterialCubeSkybox>());
}

void KiriCubeSkybox::ConstructHDR(String name)
{
    // generate pbr framebuffer
    glGenFramebuffers(1, &mCaptureFBO);
    glGenRenderbuffers(1, &mCaptureRBO);

    glBindFramebuffer(GL_FRAMEBUFFER, mCaptureFBO);
    glBindRenderbuffer(GL_RENDERBUFFER, mCaptureRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, HDR_TEX_SIZE, HDR_TEX_SIZE);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, mCaptureRBO);

    String default_name = "Subway_Lights";
    if (name != "")
    {
        default_name = name;
    }

    String path = "";
    float *data;
    Int mWidth, height, nrComponents;
    stbi_set_flip_vertically_on_load(true);
    if (RELEASE && PUBLISH)
    {
        // path = String(DB_PBR_PATH) + "hdr/" + default_name + "/" + default_name + ".hdr";
        path = "./resources/hdr/" + default_name + "/" + default_name + ".hdr";
    }
    else
    {
        path = String(DB_PBR_PATH) + "hdr/" + default_name + "/" + default_name + ".hdr";
    }

    data = stbi_loadf(path.c_str(), &mWidth, &height, &nrComponents, 0);

    if (data)
    {
        glGenTextures(1, &mHDRTexture);
        glBindTexture(GL_TEXTURE_2D, mHDRTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, mWidth, height, 0, GL_RGB, GL_FLOAT, data);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        stbi_image_free(data);
    }
    else
    {
        std::cout << "Failed to load HDR image." << std::endl;
    }
}

KiriCubeSkybox::KiriCubeSkybox(String path)
{
    mLoadHDR = false;
    Construct(path);
}

KiriCubeSkybox::KiriCubeSkybox(bool _load_hdr, String path)
{
    mLoadHDR = _load_hdr;
    if (mLoadHDR)
    {
        mSkyBox = std::make_shared<KiriCube>();

        // pre compute
        ConstructHDR(path);

        CreateCubeMap();

        CaptureData2CubeMap();

        ConvertHDR2CubeMap();

        CreateCubeMapMipMap();

        CreateIrradianceCubeMap();

        CreateSpecularCubeMap();

        RenderSpeclarCubeMap();

        CreateBRDFLutTexure();

        RenderBRDFLutTexture();

        mIBLMat = std::make_shared<KiriMaterialIBL>(mEnvCubeMapBuffer);
        mSkyBox->SetMaterial(mIBLMat);
    }
    else
    {
        Construct(path);
    }
}

void KiriCubeSkybox::Draw()
{
    // Draw skybox as last
    glDepthFunc(GL_LEQUAL); // change depth function so depth test passes when values are equal to depth buffer's content

    this->BindShader();

    // Draw cube skybox
    glBindVertexArray(mVAO);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, mEnvCubeMapBuffer);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glBindVertexArray(0);
    glDepthFunc(GL_LESS); // set depth function back to default
}

void KiriCubeSkybox::DrawIBL()
{

    glDepthFunc(GL_LEQUAL);
    mSkyBox->BindShader();
    mSkyBox->Draw();
    glDepthFunc(GL_LESS);
}

void KiriCubeSkybox::CreateCubeMap()
{
    glGenTextures(1, &mEnvCubeMapBuffer);
    glBindTexture(GL_TEXTURE_CUBE_MAP, mEnvCubeMapBuffer);

    if (mLoadHDR)
    {
        for (Int i = 0; i < CUBE_TEX_SIZE; ++i)
        {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, HDR_TEX_SIZE, HDR_TEX_SIZE, 0, GL_RGB, GL_FLOAT, nullptr);
        }
    }
    else
    {
        Int mWidth, height, nrChannels;
        for (Int i = 0; i < CUBE_TEX_SIZE; i++)
        {
            UChar *data = stbi_load(mCubeTexFile[i].c_str(), &mWidth, &height, &nrChannels, 0);
            if (data)
            {
                glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, mWidth, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
                stbi_image_free(data);
            }
            else
            {
                std::cout << "cubemap texture failed to load at path: " << mCubeTexFile[i] << std::endl;
                stbi_image_free(data);
            }
        }
    }

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    if (mLoadHDR)
    {
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    }
    else
    {
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    }

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

void KiriCubeSkybox::CaptureData2CubeMap()
{
    mCaptureProjection = Matrix4x4F::perspectiveMatrix(90.0f, 1.0f, 0.1f, 10.0f);
    mCaptureViews.append(Matrix4x4F::viewMatrix(Vector3F(0.0f, 0.0f, 0.0f), Vector3F(1.0f, 0.0f, 0.0f), Vector3F(0.0f, -1.0f, 0.0f)));
    mCaptureViews.append(Matrix4x4F::viewMatrix(Vector3F(0.0f, 0.0f, 0.0f), Vector3F(-1.0f, 0.0f, 0.0f), Vector3F(0.0f, -1.0f, 0.0f)));
    mCaptureViews.append(Matrix4x4F::viewMatrix(Vector3F(0.0f, 0.0f, 0.0f), Vector3F(0.0f, 1.0f, 0.0f), Vector3F(0.0f, 0.0f, 1.0f)));
    mCaptureViews.append(Matrix4x4F::viewMatrix(Vector3F(0.0f, 0.0f, 0.0f), Vector3F(0.0f, -1.0f, 0.0f), Vector3F(0.0f, 0.0f, -1.0f)));
    mCaptureViews.append(Matrix4x4F::viewMatrix(Vector3F(0.0f, 0.0f, 0.0f), Vector3F(0.0f, 0.0f, 1.0f), Vector3F(0.0f, -1.0f, 0.0f)));
    mCaptureViews.append(Matrix4x4F::viewMatrix(Vector3F(0.0f, 0.0f, 0.0f), Vector3F(0.0f, 0.0f, -1.0f), Vector3F(0.0f, -1.0f, 0.0f)));
}

void KiriCubeSkybox::ConvertHDR2CubeMap()
{
    mEq2CubeMat = std::make_shared<KiriMaterialEquirectangular2CubeMap>(mHDRTexture, mCaptureProjection);
    mSkyBox->SetMaterial(mEq2CubeMat);

    glViewport(0, 0, HDR_TEX_SIZE, HDR_TEX_SIZE);
    glBindFramebuffer(GL_FRAMEBUFFER, mCaptureFBO);
    for (UInt i = 0; i < 6; ++i)
    {
        mEq2CubeMat->GetShader()->SetMat4("view", mCaptureViews[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, mEnvCubeMapBuffer, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        mSkyBox->Draw();
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void KiriCubeSkybox::CreateCubeMapMipMap()
{
    glBindTexture(GL_TEXTURE_CUBE_MAP, mEnvCubeMapBuffer);
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
}

void KiriCubeSkybox::CreateIrradianceCubeMap()
{
    glGenTextures(1, &mIrradianceMapBuffer);
    glBindTexture(GL_TEXTURE_CUBE_MAP, mIrradianceMapBuffer);
    for (UInt i = 0; i < 6; ++i)
    {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, 32, 32, 0, GL_RGB, GL_FLOAT, nullptr);
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glBindFramebuffer(GL_FRAMEBUFFER, mCaptureFBO);
    glBindRenderbuffer(GL_RENDERBUFFER, mCaptureRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 32, 32);

    // solve mDiffuse integral by convolution to create an irradiance cubemap.
    mIrrConvMat = std::make_shared<KiriMaterialIrradianceConvolution>(mCaptureProjection);
    mSkyBox->SetMaterial(mIrrConvMat);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, mEnvCubeMapBuffer);

    glViewport(0, 0, 32, 32); // don't forget to configure the viewport to the capture dimensions.
    glBindFramebuffer(GL_FRAMEBUFFER, mCaptureFBO);
    for (UInt i = 0; i < 6; ++i)
    {
        mIrrConvMat->GetShader()->SetMat4("view", mCaptureViews[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, mIrradianceMapBuffer, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        mSkyBox->Draw();
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void KiriCubeSkybox::CreateSpecularCubeMap()
{
    glGenTextures(1, &mSpecularEnvCubeMapBuffer);
    glBindTexture(GL_TEXTURE_CUBE_MAP, mSpecularEnvCubeMapBuffer);
    for (UInt i = 0; i < 6; ++i)
    {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, 128, 128, 0, GL_RGB, GL_FLOAT, nullptr);
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // generate mipmaps
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
}

void KiriCubeSkybox::RenderSpeclarCubeMap()
{
    // run a quasi monte-carlo simulation on the environment lighting to create a prefilter cubemap.
    mSpeCubeMapMat = std::make_shared<KiriMaterialSpecCubeMap>(mCaptureProjection);
    mSkyBox->SetMaterial(mSpeCubeMapMat);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, mEnvCubeMapBuffer);

    glBindFramebuffer(GL_FRAMEBUFFER, mCaptureFBO);
    UInt maxMipLevels = 5;
    for (UInt mip = 0; mip < maxMipLevels; ++mip)
    {
        // resize framebuffer according to mip-level size.
        UInt mipWidth = (UInt)(128 * std::pow(0.5, mip));
        UInt mipHeight = (UInt)(128 * std::pow(0.5, mip));
        glBindRenderbuffer(GL_RENDERBUFFER, mCaptureRBO);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, mipWidth, mipHeight);
        glViewport(0, 0, mipWidth, mipHeight);

        float roughness = (float)mip / (float)(maxMipLevels - 1);
        mSpeCubeMapMat->GetShader()->SetFloat("roughness", roughness);
        for (UInt i = 0; i < 6; ++i)
        {
            mSpeCubeMapMat->GetShader()->SetMat4("view", mCaptureViews[i]);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, mSpecularEnvCubeMapBuffer, mip);

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            mSkyBox->Draw();
        }
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void KiriCubeSkybox::CreateBRDFLutTexure()
{
    glGenTextures(1, &mBRDFLUTTexture);

    glBindTexture(GL_TEXTURE_2D, mBRDFLUTTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG16F, HDR_TEX_SIZE, HDR_TEX_SIZE, 0, GL_RG, GL_FLOAT, 0);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

void KiriCubeSkybox::RenderBRDFLutTexture()
{
    mBRDFfLUTQuad = std::make_shared<KiriQuad>();
    mBRDFMat = std::make_shared<KiriMaterialBRDF>();
    mBRDFfLUTQuad->SetMaterial(mBRDFMat);

    // recapture framebuffer object and Render screen-space mQuad with BRDF shader.
    glBindFramebuffer(GL_FRAMEBUFFER, mCaptureFBO);
    glBindRenderbuffer(GL_RENDERBUFFER, mCaptureRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, HDR_TEX_SIZE, HDR_TEX_SIZE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mBRDFLUTTexture, 0);

    glViewport(0, 0, HDR_TEX_SIZE, HDR_TEX_SIZE);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    mBRDFfLUTQuad->Draw();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

KiriCubeSkybox::~KiriCubeSkybox()
{
    glDeleteVertexArrays(1, &mVAO);
    glDeleteBuffers(1, &mVBO);
}

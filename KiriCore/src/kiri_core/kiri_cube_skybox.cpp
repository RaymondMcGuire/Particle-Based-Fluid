/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-20 19:14:45 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-03-24 22:23:42
 */
#include <kiri_core/kiri_cube_skybox.h>
#include <stb_image.h>
#include <stb_image_write.h>
void KiriCubeSkybox::SetMaterial(KiriMaterialPtr _material)
{
    material = _material;
}

KiriMaterialPtr KiriCubeSkybox::GetMaterial() { return material; }

void KiriCubeSkybox::BindShader()
{
    material->Update();
}

void KiriCubeSkybox::Construct(String path)
{

    String default_path = "resources/textures/skybox/";
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

    //load cube skybox texture
    cubeTexFile.append(KiriLoadFiles::getPath(default_path + "right.jpg"));
    cubeTexFile.append(KiriLoadFiles::getPath(default_path + "left.jpg"));
    cubeTexFile.append(KiriLoadFiles::getPath(default_path + "top.jpg"));
    cubeTexFile.append(KiriLoadFiles::getPath(default_path + "bottom.jpg"));
    cubeTexFile.append(KiriLoadFiles::getPath(default_path + "front.jpg"));
    cubeTexFile.append(KiriLoadFiles::getPath(default_path + "back.jpg"));

    createCubeMap();

    //set cube skybox material
    this->SetMaterial(std::make_shared<KiriMaterialCubeSkybox>());
}

void KiriCubeSkybox::constructHDR(String name)
{
    //generate pbr framebuffer
    glGenFramebuffers(1, &captureFBO);
    glGenRenderbuffers(1, &captureRBO);

    glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
    glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, HDR_TEX_SIZE, HDR_TEX_SIZE);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, captureRBO);

    String default_name = "Subway_Lights";
    if (name != "")
    {
        default_name = name;
    }

    String path = "";
    float *data;
    Int width, height, nrComponents;
    stbi_set_flip_vertically_on_load(true);
    if (RELEASE && PUBLISH)
    {
        //path = String(DB_PBR_PATH) + "hdr/" + default_name + "/" + default_name + ".hdr";
        path = "./resources/hdr/" + default_name + "/" + default_name + ".hdr";
    }
    else
    {
        path = String(DB_PBR_PATH) + "hdr/" + default_name + "/" + default_name + ".hdr";
    }

    data = stbi_loadf(path.c_str(), &width, &height, &nrComponents, 0);

    if (data)
    {
        glGenTextures(1, &hdrTexture);
        glBindTexture(GL_TEXTURE_2D, hdrTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, width, height, 0, GL_RGB, GL_FLOAT, data);

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
    load_hdr = false;
    Construct(path);
}

KiriCubeSkybox::KiriCubeSkybox(bool _load_hdr, String path)
{
    load_hdr = _load_hdr;
    if (load_hdr)
    {
        skyboxCube = std::make_shared<KiriCube>();

        //pre compute
        constructHDR(path);

        createCubeMap();

        captureData2CubeMap();

        convertHDR2CubeMap();

        createCubeMapMipMap();

        createIrradianceCubeMap();

        createSpecularCubeMap();

        renderSpeclarCubeMap();

        createBrdfLutTexure();

        renderBrdfLutTexture();

        m_ibl = std::make_shared<KiriMaterialIBL>(envCubeMap);
        skyboxCube->SetMaterial(m_ibl);
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
    glBindTexture(GL_TEXTURE_CUBE_MAP, envCubeMap);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glBindVertexArray(0);
    glDepthFunc(GL_LESS); // set depth function back to default
}

void KiriCubeSkybox::drawIBL()
{

    glDepthFunc(GL_LEQUAL);
    skyboxCube->BindShader();
    skyboxCube->Draw();
    glDepthFunc(GL_LESS);
}

void KiriCubeSkybox::createCubeMap()
{
    glGenTextures(1, &envCubeMap);
    glBindTexture(GL_TEXTURE_CUBE_MAP, envCubeMap);

    if (load_hdr)
    {
        for (Int i = 0; i < CUBE_TEX_SIZE; ++i)
        {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, HDR_TEX_SIZE, HDR_TEX_SIZE, 0, GL_RGB, GL_FLOAT, nullptr);
        }
    }
    else
    {
        Int width, height, nrChannels;
        for (Int i = 0; i < CUBE_TEX_SIZE; i++)
        {
            UChar *data = stbi_load(cubeTexFile[i].c_str(), &width, &height, &nrChannels, 0);
            if (data)
            {
                glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
                stbi_image_free(data);
            }
            else
            {
                std::cout << "cubemap texture failed to load at path: " << cubeTexFile[i] << std::endl;
                stbi_image_free(data);
            }
        }
    }

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    if (load_hdr)
    {
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    }
    else
    {
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    }

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

void KiriCubeSkybox::captureData2CubeMap()
{
    mCaptureProjection = Matrix4x4F::perspectiveMatrix(90.0f, 1.0f, 0.1f, 10.0f);
    captureViews.append(Matrix4x4F::viewMatrix(Vector3F(0.0f, 0.0f, 0.0f), Vector3F(1.0f, 0.0f, 0.0f), Vector3F(0.0f, -1.0f, 0.0f)));
    captureViews.append(Matrix4x4F::viewMatrix(Vector3F(0.0f, 0.0f, 0.0f), Vector3F(-1.0f, 0.0f, 0.0f), Vector3F(0.0f, -1.0f, 0.0f)));
    captureViews.append(Matrix4x4F::viewMatrix(Vector3F(0.0f, 0.0f, 0.0f), Vector3F(0.0f, 1.0f, 0.0f), Vector3F(0.0f, 0.0f, 1.0f)));
    captureViews.append(Matrix4x4F::viewMatrix(Vector3F(0.0f, 0.0f, 0.0f), Vector3F(0.0f, -1.0f, 0.0f), Vector3F(0.0f, 0.0f, -1.0f)));
    captureViews.append(Matrix4x4F::viewMatrix(Vector3F(0.0f, 0.0f, 0.0f), Vector3F(0.0f, 0.0f, 1.0f), Vector3F(0.0f, -1.0f, 0.0f)));
    captureViews.append(Matrix4x4F::viewMatrix(Vector3F(0.0f, 0.0f, 0.0f), Vector3F(0.0f, 0.0f, -1.0f), Vector3F(0.0f, -1.0f, 0.0f)));
}

void KiriCubeSkybox::convertHDR2CubeMap()
{
    m_eq2cube = std::make_shared<KiriMaterialEquirectangular2CubeMap>(hdrTexture, mCaptureProjection);
    skyboxCube->SetMaterial(m_eq2cube);

    glViewport(0, 0, HDR_TEX_SIZE, HDR_TEX_SIZE);
    glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
    for (UInt i = 0; i < 6; ++i)
    {
        m_eq2cube->GetShader()->SetMat4("view", captureViews[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, envCubeMap, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        skyboxCube->Draw();
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void KiriCubeSkybox::createCubeMapMipMap()
{
    glBindTexture(GL_TEXTURE_CUBE_MAP, envCubeMap);
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
}

void KiriCubeSkybox::createIrradianceCubeMap()
{
    glGenTextures(1, &irradianceMap);
    glBindTexture(GL_TEXTURE_CUBE_MAP, irradianceMap);
    for (UInt i = 0; i < 6; ++i)
    {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, 32, 32, 0, GL_RGB, GL_FLOAT, nullptr);
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
    glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 32, 32);

    //solve diffuse integral by convolution to create an irradiance cubemap.
    m_irrconv = std::make_shared<KiriMaterialIrradianceConvolution>(mCaptureProjection);
    skyboxCube->SetMaterial(m_irrconv);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, envCubeMap);

    glViewport(0, 0, 32, 32); // don't forget to configure the viewport to the capture dimensions.
    glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
    for (UInt i = 0; i < 6; ++i)
    {
        m_irrconv->GetShader()->SetMat4("view", captureViews[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, irradianceMap, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        skyboxCube->Draw();
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void KiriCubeSkybox::createSpecularCubeMap()
{
    glGenTextures(1, &specularEnvCubeMap);
    glBindTexture(GL_TEXTURE_CUBE_MAP, specularEnvCubeMap);
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

void KiriCubeSkybox::renderSpeclarCubeMap()
{
    //run a quasi monte-carlo simulation on the environment lighting to create a prefilter cubemap.
    m_specubemap = std::make_shared<KiriMaterialSpecCubeMap>(mCaptureProjection);
    skyboxCube->SetMaterial(m_specubemap);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, envCubeMap);

    glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
    UInt maxMipLevels = 5;
    for (UInt mip = 0; mip < maxMipLevels; ++mip)
    {
        // resize framebuffer according to mip-level size.
        UInt mipWidth = (UInt)(128 * std::pow(0.5, mip));
        UInt mipHeight = (UInt)(128 * std::pow(0.5, mip));
        glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, mipWidth, mipHeight);
        glViewport(0, 0, mipWidth, mipHeight);

        float roughness = (float)mip / (float)(maxMipLevels - 1);
        m_specubemap->GetShader()->SetFloat("roughness", roughness);
        for (UInt i = 0; i < 6; ++i)
        {
            m_specubemap->GetShader()->SetMat4("view", captureViews[i]);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, specularEnvCubeMap, mip);

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            skyboxCube->Draw();
        }
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void KiriCubeSkybox::createBrdfLutTexure()
{
    glGenTextures(1, &brdfLUTTexture);

    glBindTexture(GL_TEXTURE_2D, brdfLUTTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG16F, HDR_TEX_SIZE, HDR_TEX_SIZE, 0, GL_RG, GL_FLOAT, 0);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

void KiriCubeSkybox::renderBrdfLutTexture()
{
    brdfLUTQuad = std::make_shared<KiriQuad>();
    m_brdf = std::make_shared<KiriMaterialBRDF>();
    brdfLUTQuad->SetMaterial(m_brdf);

    //recapture framebuffer object and render screen-space quad with BRDF shader.
    glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
    glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, HDR_TEX_SIZE, HDR_TEX_SIZE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, brdfLUTTexture, 0);

    glViewport(0, 0, HDR_TEX_SIZE, HDR_TEX_SIZE);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    brdfLUTQuad->Draw();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

KiriCubeSkybox::~KiriCubeSkybox()
{
    glDeleteVertexArrays(1, &mVAO);
    glDeleteBuffers(1, &mVBO);
}

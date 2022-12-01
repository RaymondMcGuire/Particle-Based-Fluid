/***
 * @Author: Xu.WANG
 * @Date: 2020-06-16 01:32:28
 * @LastEditTime: 2022-04-10 11:01:58
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath: \Kiri\KiriCore\src\kiri_core\fluid\fluid_render_system.cpp
 */
#include <kiri_core/fluid/fluid_render_system.h>
#include <opengl_helper/opengl_helper.h>
#include <kiri_params.h>
#include <kiri_application.h>

namespace KIRI
{
    KiriFluidRenderSystem::KiriFluidRenderSystem(
        UInt mWidth,
        UInt height,
        KiriCameraPtr camera)
        : mCamera(camera)
    {
        SCREEN_WIDTH = mWidth;
        SCREEN_HEIGHT = height;
        mSkyBoxTex = NULL;

        mNumOfParticles = 0;
        mParticleRadius = 0.1f;
        mSmoothIter = 2;

        bMultiColor = false;
        bDepthAB = false;
        bFluidTransparent = false;
        bSoildSsf = false;

        mDepthShader = std::make_shared<KiriMaterialSSFDepth>();
        mThickShader = std::make_shared<KiriMaterialSSFThick>();
        mNormalShader = std::make_shared<KiriMaterialSSFNormal>();
        mMultiColorShader = std::make_shared<KiriMaterialSSFMultiColor>();
        mFluidShader = std::make_shared<KiriMaterialSSFFluid>();
        mSmoothShader = std::make_shared<KiriMaterialSSFSmooth>();

        InitBuffer();
        InitQuadBuff();
    }

    void KiriFluidRenderSystem::SetSkyBoxTex(UInt skyBox)
    {
        mSkyBoxTex = skyBox;
    }

    void KiriFluidRenderSystem::EnableSoildSsfMode(bool soildSsf)
    {
        bSoildSsf = !soildSsf;
    }

    void KiriFluidRenderSystem::EnableFluidTransparentMode(bool fluidTransparent)
    {
        bFluidTransparent = fluidTransparent;
    }

    void KiriFluidRenderSystem::SetParticles(Array1Vec3F partilces, float radius)
    {
        mNumOfParticles = partilces.size();
        mParticleRadius = radius;

        glBindVertexArray(mParticlesVAO);
        glBindBuffer(GL_ARRAY_BUFFER, mParticlesVBO);
        glBufferData(GL_ARRAY_BUFFER, partilces.size() * 3 * sizeof(float),
                     partilces.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float),
                              static_cast<void *>(0));
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    void KiriFluidRenderSystem::SetParticlesWithRadius(Array1Vec4F pos, Array1Vec4F col, UInt num)
    {
        bMultiColor = true;
        mNumOfParticles = num;
        mParticleRadius = -1.f;

        glBindVertexArray(mParticlesVAO);
        glBindBuffer(GL_ARRAY_BUFFER, mParticlesVBO);
        glBufferData(GL_ARRAY_BUFFER, pos.size() * 4 * sizeof(float),
                     pos.data(), GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                              static_cast<void *>(0));

        glBindBuffer(GL_ARRAY_BUFFER, mParticlesColorVBO);
        glBufferData(GL_ARRAY_BUFFER, col.size() * 4 * sizeof(float),
                     col.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                              static_cast<void *>(0));
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    void KiriFluidRenderSystem::SetParticlesVBO(UInt vbo, UInt num, float radius)
    {
        bMultiColor = false;
        mNumOfParticles = num;
        mParticleRadius = radius;
        mParticlesVBO = vbo;
        glBindVertexArray(mParticlesVAO);
        glBindBuffer(GL_ARRAY_BUFFER, mParticlesVBO);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float),
                              static_cast<void *>(0));
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    void KiriFluidRenderSystem::SetParticlesVBO(UInt pvbo, UInt cvbo, UInt num, float radius)
    {
        bMultiColor = true;
        mNumOfParticles = num;
        mParticleRadius = radius;
        mParticlesVBO = pvbo;
        mParticlesColorVBO = cvbo;
        glBindVertexArray(mParticlesVAO);
        glBindBuffer(GL_ARRAY_BUFFER, mParticlesVBO);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                              static_cast<void *>(0));

        glBindBuffer(GL_ARRAY_BUFFER, mParticlesColorVBO);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                              static_cast<void *>(0));
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    void KiriFluidRenderSystem::SetParticlesVBOWithRadius(UInt pvbo, UInt cvbo, UInt num)
    {
        bMultiColor = true;
        mNumOfParticles = num;
        mParticleRadius = -1.f;
        mParticlesVBO = pvbo;
        mParticlesColorVBO = cvbo;
        glBindVertexArray(mParticlesVAO);
        glBindBuffer(GL_ARRAY_BUFFER, mParticlesVBO);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                              static_cast<void *>(0));

        glBindBuffer(GL_ARRAY_BUFFER, mParticlesColorVBO);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                              static_cast<void *>(0));
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    void KiriFluidRenderSystem::renderFluid(UInt bgTex, UInt bgDepthTex)
    {
        bDepthAB = false;

        glEnable(GL_PROGRAM_POINT_SIZE);
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

        if (bMultiColor)
        {
            RenderMultiColorTex();
        }

        RenderDepthTex();
        RenderThickTex();

        if (!SSF_DEMO_PARAMS.particleView)
        {
            for (Int i = 0; i < mSmoothIter; i++)
            {
                bDepthAB = !bDepthAB;
                RenderSmoothTex(SSF_DEMO_PARAMS.particleView);
            }
        }

        RenderNormalTex();
        RenderFluidTex(bgTex, bgDepthTex);

        glDisable(GL_PROGRAM_POINT_SIZE);
    }

    void KiriFluidRenderSystem::RenderDepthTex()
    {
        glEnable(GL_DEPTH_TEST);
        glBindFramebuffer(GL_FRAMEBUFFER, fluidFBO);

        float inf[] = {100.f}, zero[] = {0.f};
        glClearTexImage(depthATex, 0, GL_RED, GL_FLOAT, inf);
        glClearTexImage(depthBTex, 0, GL_RED, GL_FLOAT, inf);

        GLenum bufs[] = {GL_COLOR_ATTACHMENT0};
        glDrawBuffers(1, bufs);

        mDepthShader->SetParticleRadius(mParticleRadius);
        mDepthShader->SetParticleScale(CalcParticleScale());
        mDepthShader->Update();

        glBindVertexArray(mParticlesVAO);

        glClear(GL_DEPTH_BUFFER_BIT);
        glDrawArrays(GL_POINTS, 0, (GLsizei)mNumOfParticles);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        glDisable(GL_DEPTH_TEST);
    }

    void KiriFluidRenderSystem::RenderThickTex()
    {
        glEnable(GL_BLEND);

        glBindFramebuffer(GL_FRAMEBUFFER, fluidFBO);

        float zero[] = {0.f};
        glClearTexImage(thickTex, 0, GL_RED, GL_FLOAT, zero);

        GLenum bufs[] = {GL_COLOR_ATTACHMENT4 /* thick */};
        glDrawBuffers(1, bufs);

        glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
        glBlendFuncSeparateiARB(0, GL_ONE, GL_ONE, GL_ONE, GL_ONE);

        mThickShader->SetParticleRadius(mParticleRadius);
        mThickShader->SetParticleScale(CalcParticleScale());
        mThickShader->Update();

        glBindVertexArray(mParticlesVAO);
        glDrawArrays(GL_POINTS, 0, (GLsizei)mNumOfParticles);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        glDisable(GL_BLEND);
    }

    void KiriFluidRenderSystem::RenderSmoothTex(bool particle_view)
    {
        // Render mSmooth to fluid fbo
        glBindFramebuffer(GL_FRAMEBUFFER, fluidFBO);

        mSmoothShader->Update();

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, smoothTex());
        glActiveTexture(GL_TEXTURE1);
        glBindImageTexture(1, realDepthTex(), 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, multiColorTex);

        mSmoothShader->SetSmoothTex(0);
        mSmoothShader->SetRealDepthTex(1);
        mSmoothShader->SetMultiColorTex(2);
        mSmoothShader->SetParticleView(particle_view);
        mSmoothShader->SetEnableSSF(bSoildSsf);

        glBindVertexArray(mQuadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void KiriFluidRenderSystem::RenderNormalTex()
    {
        // Render mNormal tex to fluid fbo
        glBindFramebuffer(GL_FRAMEBUFFER, fluidFBO);

        mNormalShader->SetParticleScale(CalcParticleScale());
        mNormalShader->Update();

        float black[] = {0.f, 0.f, 0.f, 0.f};
        GLenum bufs[] = {GL_COLOR_ATTACHMENT2};
        glDrawBuffers(1, bufs);
        glClearTexImage(normTex, 0, GL_RGBA, GL_FLOAT, black);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, realDepthTex());
        mNormalShader->SetDepthTex(0);

        glBindVertexArray(mQuadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void KiriFluidRenderSystem::RenderMultiColorTex()
    {
        glEnable(GL_DEPTH_TEST);
        // Render multi color tex to fluid fbo
        glBindFramebuffer(GL_FRAMEBUFFER, fluidFBO);

        float black[] = {0.f, 0.f, 0.f, 0.f};
        GLenum bufs[] = {GL_COLOR_ATTACHMENT3};
        glDrawBuffers(1, bufs);
        glClearTexImage(multiColorTex, 0, GL_RGBA, GL_FLOAT, black);

        mMultiColorShader->SetParticleRadius(mParticleRadius);
        mMultiColorShader->SetParticleScale(CalcParticleScale());
        mMultiColorShader->SetTransparentMode(bFluidTransparent);
        mMultiColorShader->Update();

        glClear(GL_DEPTH_BUFFER_BIT);
        glBindVertexArray(mParticlesVAO);
        glDrawArrays(GL_POINTS, 0, (GLsizei)mNumOfParticles);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDisable(GL_DEPTH_TEST);
    }

    void KiriFluidRenderSystem::RenderFluidTex(UInt bgTex, UInt _depthTex)
    {
        glEnable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        mFluidShader->SetCameraParams(
            mCamera->GetAspect(),
            mCamera->GetFar(),
            mCamera->GetNear(),
            mCamera->GetFov(),
            mCamera->inverseViewMatrix());

        mFluidShader->Update();

        // transfer params to shader
        mFluidShader->SetParticleView(SSF_DEMO_PARAMS.particleView);
        mFluidShader->SetMultiColor(bMultiColor);
        mFluidShader->SetRenderOpt(SSF_DEMO_PARAMS.renderOpt);

        // Bind texture
        glBindVertexArray(mQuadVAO);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, realDepthTex());
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, normTex);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, thickTex);
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_CUBE_MAP, mSkyBoxTex);
        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_2D, bgTex);
        glActiveTexture(GL_TEXTURE5);
        glBindTexture(GL_TEXTURE_2D, multiColorTex);
        glActiveTexture(GL_TEXTURE6);
        glBindTexture(GL_TEXTURE_2D, _depthTex);

        glBlendFuncSeparateiARB(0, GL_ONE, GL_ZERO, GL_ONE, GL_ZERO);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        mFluidShader->SetDepthTex(0);
        mFluidShader->SetNormalTex(1);
        mFluidShader->SetThickTex(2);
        mFluidShader->SetSkyBoxTex(3);
        mFluidShader->SetBGTex(4);
        mFluidShader->SetMultiColorTex(5);
        mFluidShader->SetBgDepthTex(6);

        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        glDisable(GL_BLEND);
        glDisable(GL_DEPTH_TEST);
    }

    float KiriFluidRenderSystem::CalcParticleScale()
    {
        auto &app = KIRI::KiriApplication::Get();
        auto height = app.GetWindow().GetWindowHeight();

        float aspect = mCamera->GetAspect();
        float fovy = mCamera->GetFov();

        float particleScale = (float)height / tanf(kiri_math::degreesToRadians<float>(fovy) * 0.5f);
        return particleScale;
    }

    void KiriFluidRenderSystem::InitBuffer()
    {
        // particle
        glGenBuffers(1, &mParticlesVBO);
        glGenBuffers(1, &mParticlesColorVBO);
        glGenVertexArrays(1, &mParticlesVAO);

        // depth
        glGenTextures(1, &depthTex);
        glGenTextures(1, &depthATex);
        glGenTextures(1, &depthBTex);

        // thick
        glGenTextures(1, &thickTex);

        // mNormal
        glGenTextures(1, &normTex);

        // multi color
        glGenTextures(1, &multiColorTex);

        // init depth Texture
        glBindTexture(GL_TEXTURE_2D, depthTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, SCREEN_WIDTH, SCREEN_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        glBindTexture(GL_TEXTURE_2D, depthATex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, SCREEN_WIDTH, SCREEN_HEIGHT, 0, GL_RED, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        glBindTexture(GL_TEXTURE_2D, depthBTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, SCREEN_WIDTH, SCREEN_HEIGHT, 0, GL_RED, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        // thick
        glBindTexture(GL_TEXTURE_2D, thickTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, SCREEN_WIDTH, SCREEN_HEIGHT, 0, GL_RED, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // mNormal
        glBindTexture(GL_TEXTURE_2D, normTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SCREEN_WIDTH, SCREEN_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        // multi color
        glBindTexture(GL_TEXTURE_2D, multiColorTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SCREEN_WIDTH, SCREEN_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        // init fluid frame buffer
        glGenFramebuffers(1, &fluidFBO);
        glBindFramebuffer(GL_FRAMEBUFFER, fluidFBO);

        // attach depth texture
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTex, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, depthATex, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, depthBTex, 0);

        // thick
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, GL_TEXTURE_2D, thickTex, 0);

        // multi color
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, multiColorTex, 0);

        // mNormal
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, normTex, 0);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void KiriFluidRenderSystem::InitQuadBuff()
    {
        float quadVertices[] = {
            // positions   // texCoords
            -1.0f, 1.0f, 0.0f, 1.0f,
            -1.0f, -1.0f, 0.0f, 0.0f,
            1.0f, -1.0f, 1.0f, 0.0f,

            -1.0f, 1.0f, 0.0f, 1.0f,
            1.0f, -1.0f, 1.0f, 0.0f,
            1.0f, 1.0f, 1.0f, 1.0f};

        UInt quadVbo;
        glGenVertexArrays(1, &mQuadVAO);
        glGenBuffers(1, &quadVbo);
        glBindVertexArray(mQuadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, quadVbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));

        glBindVertexArray(0);
    }
}
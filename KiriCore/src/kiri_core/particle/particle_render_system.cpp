/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-24 20:04:09
 * @LastEditTime: 2021-02-20 01:16:17
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\src\kiri_core\particle\particle_render_system.cpp
 */

#include <kiri_core/particle/particle_render_system.h>
#include <kiri_application.h>
namespace KIRI
{

    KiriParticleRenderSystem::KiriParticleRenderSystem(KiriCameraPtr camera)
        : mCamera(camera)
    {
        mNumOfParticles = 0;
        mParticleRadius = 1.0f;

        mPointSpriteMaterial = std::make_shared<KiriMaterialParticlePointSprite>();

        glGenBuffers(1, &mParticlesVBO);
        glGenVertexArrays(1, &mParticlesVAO);
    }

    void KiriParticleRenderSystem::RenderParticles()
    {
        glEnable(GL_PROGRAM_POINT_SIZE);
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

        mPointSpriteMaterial->SetParticleRadius(mParticleRadius);
        mPointSpriteMaterial->SetParticleScale(CalcParticleScale());
        mPointSpriteMaterial->Update();

        glBindVertexArray(mParticlesVAO);
        glDrawArrays(GL_POINTS, 0, (GLsizei)mNumOfParticles);
        glBindVertexArray(0);

        glDisable(GL_PROGRAM_POINT_SIZE);
    }

    void KiriParticleRenderSystem::SetParticles(ArrayAccessor1<Vector3F> Particles, float Radius)
    {
        mNumOfParticles = Particles.size();
        mParticleRadius = Radius;

        glBindVertexArray(mParticlesVAO);
        glBindBuffer(GL_ARRAY_BUFFER, mParticlesVBO);
        glBufferData(GL_ARRAY_BUFFER, Particles.size() * 3 * sizeof(float),
                     Particles.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float),
                              static_cast<void *>(0));
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    void KiriParticleRenderSystem::SetParticles(ArrayAccessor1<Vector4F> Particles)
    {
        mNumOfParticles = Particles.size();
        mParticleRadius = -1.f;

        glBindVertexArray(mParticlesVAO);
        glBindBuffer(GL_ARRAY_BUFFER, mParticlesVBO);
        glBufferData(GL_ARRAY_BUFFER, Particles.size() * 4 * sizeof(float),
                     Particles.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                              static_cast<void *>(0));
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    void KiriParticleRenderSystem::SetParticlesVBO(UInt mVBO, UInt Num, float Radius)
    {
        mNumOfParticles = Num;
        mParticleRadius = Radius;
        mParticlesVBO = mVBO;
        glBindVertexArray(mParticlesVAO);
        glBindBuffer(GL_ARRAY_BUFFER, mParticlesVBO);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float),
                              static_cast<void *>(0));
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    float KiriParticleRenderSystem::CalcParticleScale()
    {
        auto &app = KIRI::KiriApplication::Get();
        auto height = app.GetWindow().GetWindowHeight();

        float aspect = mCamera->GetAspect();
        float fovy = mCamera->GetFov();

        float particleScale = (float)height / tanf(kiri_math::degreesToRadians<float>(fovy) * 0.5f);
        return particleScale;
    }
}
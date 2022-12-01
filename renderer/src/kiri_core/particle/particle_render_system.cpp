/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-07-08 14:46:30
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-09-19 15:44:27
 * @FilePath: \Kiri\KiriCore\src\kiri_core\particle\particle_render_system.cpp
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#include <kiri_application.h>
#include <kiri_core/particle/particle_render_system.h>

namespace KIRI {

KiriParticleRenderSystem::KiriParticleRenderSystem(KiriCameraPtr camera)
    : mCamera(camera) {
  mNumOfParticles = 0;
  mParticleRadius = 1.0f;
  KIRI_LOG_DEBUG("init particle system");
  mPointSpriteMaterial = std::make_shared<KiriMaterialParticlePointSprite>();

  glGenBuffers(1, &mParticlesVBO);
  glGenVertexArrays(1, &mParticlesVAO);
}

void KiriParticleRenderSystem::RenderParticles() {
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

void KiriParticleRenderSystem::SetParticles(Array1<Vector4F> particles) {
  mNumOfParticles = particles.size();
  mParticleRadius = -1.f;

  glBindVertexArray(mParticlesVAO);
  glBindBuffer(GL_ARRAY_BUFFER, mParticlesVBO);
  glBufferData(GL_ARRAY_BUFFER, particles.size() * 4 * sizeof(float),
               particles.data(), GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                        static_cast<void *>(0));
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
}

void KiriParticleRenderSystem::SetParticles(Vec_Vec4F particles) {
  mNumOfParticles = particles.size();
  mParticleRadius = -1.f;

  glBindVertexArray(mParticlesVAO);
  glBindBuffer(GL_ARRAY_BUFFER, mParticlesVBO);
  glBufferData(GL_ARRAY_BUFFER, particles.size() * 4 * sizeof(float),
               particles.data(), GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                        static_cast<void *>(0));
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
}

void KiriParticleRenderSystem::SetParticlesVBO(UInt vbo, UInt num,
                                               float radius) {
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

float KiriParticleRenderSystem::CalcParticleScale() {
  auto &app = KIRI::KiriApplication::Get();
  auto height = app.GetWindow().GetWindowHeight();

  float aspect = mCamera->GetAspect();
  float fovy = mCamera->GetFov();

  float particleScale =
      (float)height / tanf(kiri_math::degreesToRadians<float>(fovy) * 0.5f);
  return particleScale;
}
} // namespace KIRI
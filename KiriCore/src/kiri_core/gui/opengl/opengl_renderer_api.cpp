/*** 
 * @Author: Jayden Zhang
 * @Date: 2020-10-26 00:51:36
 * @LastEditTime: 2020-11-02 19:37:25
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\src\kiri_core\gui\opengl\opengl_renderer_api.cpp
 */
#include <kiri_core/gui/opengl/opengl_renderer_api.h>

#include <glad/glad.h>
namespace KIRI
{
    void KiriOpenGLRendererAPI::Init()
    {
        glEnable(GL_MULTISAMPLE);
        glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
    }

    void KiriOpenGLRendererAPI::Clear(bool depth)
    {
        if (depth)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        else
            glClear(GL_COLOR_BUFFER_BIT);
    }

    void KiriOpenGLRendererAPI::SetClearColor(const Vector4F &color)
    {
        glClearColor(color.x, color.y, color.z, color.w);
    }

    void KiriOpenGLRendererAPI::SetViewport(const Vector4F &rect)
    {
        glViewport(rect.x, rect.y, rect.z, rect.w);
    }

    void KiriOpenGLRendererAPI::GlobalUboGenerate()
    {
        //Matrices(projection matrix, view matrix, camera position)
        glGenBuffers(1, &mUboMatrices);
        glBindBuffer(GL_UNIFORM_BUFFER, mUboMatrices);
        glBufferData(GL_UNIFORM_BUFFER, 2 * sizeof(float) * 16 + sizeof(float) * 3, NULL, GL_STATIC_DRAW);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);
        // define the range of the buffer that links to a uniform binding point
        glBindBufferRange(GL_UNIFORM_BUFFER, 0, mUboMatrices, 0, 2 * sizeof(float) * 16 + sizeof(float) * 3);
    }

    void KiriOpenGLRendererAPI::GlobalUboBind(const KiriCameraPtr &camera)
    {
        Matrix4x4F projection = camera->ProjectionMatrix();
        glBindBuffer(GL_UNIFORM_BUFFER, mUboMatrices);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(float) * 16, &projection.data()[0]);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);

        Matrix4x4F view = camera->ViewMatrix();
        glBindBuffer(GL_UNIFORM_BUFFER, mUboMatrices);
        glBufferSubData(GL_UNIFORM_BUFFER, sizeof(float) * 16, sizeof(float) * 16, &view.data()[0]);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);

        Vector3F cam_pos = camera->Position();
        Vec_Float vec_pos{cam_pos.x, cam_pos.y, cam_pos.z};

        glBindBuffer(GL_UNIFORM_BUFFER, mUboMatrices);
        glBufferSubData(GL_UNIFORM_BUFFER, 2 * sizeof(float) * 16, sizeof(float) * 3, &vec_pos.data()[0]);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);
    }

} // namespace KIRI

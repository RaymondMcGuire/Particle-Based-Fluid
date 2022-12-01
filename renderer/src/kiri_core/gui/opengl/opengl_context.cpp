/*** 
 * @Author: Jayden Zhang
 * @Date: 2020-10-25 12:53:53
 * @LastEditTime: 2020-10-26 02:27:17
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\src\kiri_core\gui\opengl\opengl_context.cpp
 */
#include <kiri_core/gui/opengl/opengl_context.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <gl/GL.h>

#include <kiri_define.h>

namespace KIRI
{
    KiriOpenGLContext::KiriOpenGLContext(GLFWwindow *windowHandle)
        : mWindowHandle(windowHandle)
    {
        KIRI_ASSERT(windowHandle);
    }

    void KiriOpenGLContext::Init()
    {
        glfwMakeContextCurrent(mWindowHandle);
        int status = gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
        KIRI_ASSERT(status);
    }

    void KiriOpenGLContext::SwapBuffers()
    {
        glfwSwapBuffers(mWindowHandle);
    }
} // namespace KIRI

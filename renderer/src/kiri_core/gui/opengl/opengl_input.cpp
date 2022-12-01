/*** 
 * @Author: Jayden Zhang
 * @Date: 2020-10-25 14:14:45
 * @LastEditTime: 2020-10-26 17:07:41
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\src\kiri_core\gui\opengl\opengl_input.cpp
 */
#include <kiri_core/gui/opengl/opengl_input.h>

#include <kiri_application.h>
#include <GLFW/glfw3.h>

namespace KIRI
{
    KiriInput *KiriInput::s_Instance = new KiriOpenGLInput();

    bool KiriOpenGLInput::IsKeyDownImpl(int keycode)
    {
        auto window = static_cast<GLFWwindow *>(KiriApplication::Get().GetWindow().GetNativeWindow());
        auto status = glfwGetKey(window, keycode);
        return status == GLFW_PRESS || status == GLFW_REPEAT;
    }

    bool KiriOpenGLInput::IsMouseButtonDownImpl(int mouseButton)
    {
        auto window = static_cast<GLFWwindow *>(KiriApplication::Get().GetWindow().GetNativeWindow());
        auto status = glfwGetMouseButton(window, mouseButton);
        return status == GLFW_PRESS;
    }

    Vector2F KiriOpenGLInput::GetMousePosImpl()
    {
        auto window = static_cast<GLFWwindow *>(KiriApplication::Get().GetWindow().GetNativeWindow());
        double x, y;
        glfwGetCursorPos(window, &x, &y);
        return Vector2F((float)x, (float)y);
    }

    float KiriOpenGLInput::GetMouseXImpl()
    {
        auto [x, y] = GetMousePosImpl();
        return x;
    }

    float KiriOpenGLInput::GetMouseYImpl()
    {
        auto [x, y] = GetMousePosImpl();
        return y;
    }
} // namespace KIRI

/***
 * @Author: Jayden Zhang
 * @Date: 2020-10-25 12:28:19
 * @LastEditTime: 2020-11-02 11:01:31
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath: \Kiri\KiriCore\src\kiri_core\gui\opengl\opengl_window.cpp
 */
#include <kiri_core/gui/opengl/opengl_window.h>

#include <kiri_core/event/application_event.h>
#include <kiri_core/event/mouse_event.h>
#include <kiri_core/event/key_event.h>
#include <kiri_core/gui/opengl/opengl_context.h>

namespace KIRI
{
    static bool sGLFWInitialized = false;

    static void GLFWErrorCallback(int error, const char *description)
    {
        KIRI_LOG_ERROR("GLFW ERROR ({0}) {1}", error, description);
    }

    KiriWindow *KiriWindow::CreateKIRIWindow(const WindowProperty &property)
    {
        return new KiriOpenGLWindow(property);
    }

    KiriOpenGLWindow::KiriOpenGLWindow(const WindowProperty &windowProperty)
    {
        Init(windowProperty);
    }

    void KiriOpenGLWindow::Init(const WindowProperty &windowProperty)
    {
        mWindowData.Title = windowProperty.Title;
        mWindowData.Width = windowProperty.Width;
        mWindowData.Height = windowProperty.Height;

        KIRI_LOG_INFO("Created KiriWindow: {0} ({1},{2})", windowProperty.Title, windowProperty.Width, windowProperty.Height);

        // Make sure to initialize GLFW only once, along with the first window
        if (!sGLFWInitialized)
        {
            int success = glfwInit();
            KIRI_ASSERT(success);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
            glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
            glfwWindowHint(GLFW_SAMPLES, 4);

            glfwSetErrorCallback(GLFWErrorCallback);
            sGLFWInitialized = true;
        }

        mWindow = glfwCreateWindow((int)mWindowData.Width, (int)mWindowData.Height, mWindowData.Title.c_str(), nullptr, nullptr);

        mRenderContext = new KiriOpenGLContext(mWindow);
        mRenderContext->Init();

        glfwSetWindowUserPointer(mWindow, &mWindowData);
        LockFPS(windowProperty.LockFPS);

        // Set GLFW Callbacks
        glfwSetWindowSizeCallback(mWindow, [](GLFWwindow *window, int mWidth, int height)
                                  {
            WindowData &data = *(WindowData *)glfwGetWindowUserPointer(window);
            data.Width = mWidth;
            data.Height = height;

            KiriWindowResizeEvent event(mWidth, height);
            data.EventCallback(event); });

        glfwSetWindowCloseCallback(mWindow, [](GLFWwindow *window)
                                   {
            WindowData &data = *(WindowData *)glfwGetWindowUserPointer(window);
            KiriWindowCloseEvent event;
            data.EventCallback(event); });

        glfwSetKeyCallback(mWindow, [](GLFWwindow *window, int key, int scancode, int action, int mods)
                           {
            WindowData &data = *(WindowData *)glfwGetWindowUserPointer(window);
            switch (action)
            {
            case GLFW_PRESS:
            {
                KiriKeyPressedEvent event(key, 0);
                data.EventCallback(event);
                break;
            }
            case GLFW_RELEASE:
            {
                KiriKeyReleasedEvent event(key);
                data.EventCallback(event);
                break;
            }
            case GLFW_REPEAT:
            {
                KiriKeyPressedEvent event(key, 1);
                data.EventCallback(event);
                break;
            }
            default:
                break;
            } });

        glfwSetCharCallback(mWindow, [](GLFWwindow *window, unsigned int keycode)
                            {
            WindowData &data = *(WindowData *)glfwGetWindowUserPointer(window);
            KiriKeyTypeEvent event(keycode);
            data.EventCallback(event); });

        glfwSetMouseButtonCallback(mWindow, [](GLFWwindow *window, int button, int action, int mods)
                                   {
            WindowData &data = *(WindowData *)glfwGetWindowUserPointer(window);
            switch (action)
            {
            case GLFW_PRESS:
            {
                KiriMouseButtonPressedEvent event(button);
                data.EventCallback(event);
                break;
            }
            case GLFW_RELEASE:
            {
                KiriMouseButtonReleasedEvent event(button);
                data.EventCallback(event);
                break;
            }
            } });

        glfwSetScrollCallback(mWindow, [](GLFWwindow *window, double xOffset, double yOffset)
                              {
            WindowData &data = *(WindowData *)glfwGetWindowUserPointer(window);
            KiriMouseScrollEvent event((float)xOffset, (float)yOffset);
            data.EventCallback(event); });

        glfwSetCursorPosCallback(mWindow, [](GLFWwindow *window, double xPos, double yPos)
                                 {
            WindowData &data = *(WindowData *)glfwGetWindowUserPointer(window);
            KiriMouseMoveEvent event((float)xPos, (float)yPos);
            data.EventCallback(event); });
    }

    KiriOpenGLWindow::~KiriOpenGLWindow()
    {
        ShutDown();
    }

    bool KiriOpenGLWindow::IsFullscreen() const
    {
        return glfwGetWindowMonitor(mWindow) != nullptr;
    }

    void KiriOpenGLWindow::ShutDown()
    {
        glfwDestroyWindow(mWindow);
    }

    void KiriOpenGLWindow::OnUpdate()
    {
        glfwPollEvents();
        mRenderContext->SwapBuffers();
    }

    void KiriOpenGLWindow::LockFPS(bool enabled)
    {
        if (enabled)
            glfwSwapInterval(1);
        else
            glfwSwapInterval(0);

        mWindowData.LockFPS = enabled;
    }

    bool KiriOpenGLWindow::IsLocked() const
    {
        return mWindowData.LockFPS;
    }
} // namespace  KIRI

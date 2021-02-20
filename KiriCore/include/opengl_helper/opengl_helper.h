/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-18 01:04:15
 * @LastEditTime: 2021-02-20 19:41:22
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\opengl_helper\opengl_helper.h
 */

#ifndef _KIRI_OPENGL_HELPER_H_
#define _KIRI_OPENGL_HELPER_H_
#pragma once
#include <kiri_pch.h>

void checkGLErr(const char *func_name)
{
    GLenum err;
    const char *errString;
    if ((err = glGetError()) != GL_NO_ERROR)
    {
        switch (err)
        {
        case GL_INVALID_OPERATION:
            errString = "INVALID_OPERATION";
            break;
        case GL_INVALID_ENUM:
            errString = "INVALID_ENUM";
            break;
        case GL_INVALID_VALUE:
            errString = "INVALID_VALUE";
            break;
        case GL_OUT_OF_MEMORY:
            errString = "OUT_OF_MEMORY";
            break;
        case GL_INVALID_FRAMEBUFFER_OPERATION:
            errString = "INVALID_FRAMEBUFFER_OPERATION";
            break;
        }
        fprintf(stderr, "OpenGL Error #%d: %s in function %s\n", err, errString, func_name);
    }
}

void checkFramebufferComplete()
{
    GLenum err = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    const char *errString = NULL;
    switch (err)
    {
    case GL_FRAMEBUFFER_UNDEFINED:
        errString = "GL_FRAMEBUFFER_UNDEFINED";
        break;
    case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
        errString = "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT";
        break;
    case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
        errString = "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT";
        break;
    case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
        errString = "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER";
        break;
    case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
        errString = "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER";
        break;
    case GL_FRAMEBUFFER_UNSUPPORTED:
        errString = "GL_FRAMEBUFFER_UNSUPPORTED";
        break;
    case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
        errString = "GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE";
        break;
    case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
        errString = "GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS";
        break;
    }

    if (errString)
    {
        fprintf(stderr, "OpenGL Framebuffer Error #%d: %s\n", err, errString);
    }
    else
    {
        printf("Framebuffer complete check ok\n");
    }
}

#endif
/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:26
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 13:44:59
 * @FilePath: \core\include\opengl_helper\opengl_helper.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _KIRI_OPENGL_HELPER_H_
#define _KIRI_OPENGL_HELPER_H_
#pragma once
#include <kiri_pch.h>

void CheckGLErr(const char *func_name)
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

void CheckFramebufferComplete()
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
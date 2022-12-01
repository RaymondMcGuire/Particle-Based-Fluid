/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-18 01:04:15
 * @LastEditTime: 2021-02-20 18:26:39
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\include\kiri_core\kiri_shader.h
 */

#ifndef _KIRI_SHADER_H_
#define _KIRI_SHADER_H_

#pragma once

#include <kiri_pch.h>
#include <glad/glad.h>
class KiriShader
{
public:
    UInt ID;

    KiriShader(const String, const String, const String = "");

    // activate the shader
    // ------------------------------------------------------------------------
    inline void Use() const
    {
        glUseProgram(ID);
    }
    // utility uniform functions
    // ------------------------------------------------------------------------
    inline void SetBool(const String &Name, bool Value) const
    {
        glUniform1i(glGetUniformLocation(ID, Name.c_str()), (Int)Value);
    }
    // ------------------------------------------------------------------------
    inline void SetInt(const String &Name, Int Value) const
    {
        glUniform1i(glGetUniformLocation(ID, Name.c_str()), Value);
    }
    // ------------------------------------------------------------------------
    inline void SetFloat(const String &Name, float Value) const
    {
        glUniform1f(glGetUniformLocation(ID, Name.c_str()), Value);
    }
    // ------------------------------------------------------------------------
    inline void SetVec2(const String &Name, const Vector2F &Value) const
    {
        glUniform2fv(glGetUniformLocation(ID, Name.c_str()), 1, &Value[0]);
    }
    inline void SetVec2(const String &Name, float x, float y) const
    {
        glUniform2f(glGetUniformLocation(ID, Name.c_str()), x, y);
    }
    // ------------------------------------------------------------------------
    inline void SetVec3(const String &Name, float x, float y, float z) const
    {
        glUniform3f(glGetUniformLocation(ID, Name.c_str()), x, y, z);
    }

    inline void SetVec3(const String &Name, const Vector3F &Value) const
    {
        glUniform3fv(glGetUniformLocation(ID, Name.c_str()), 1, &Value[0]);
    }
    // ------------------------------------------------------------------------
    inline void SetVec4(const String &Name, const Vector4F &Value) const
    {
        glUniform4fv(glGetUniformLocation(ID, Name.c_str()), 1, &Value[0]);
    }
    inline void SetVec4(const String &Name, float x, float y, float z, float w)
    {
        glUniform4f(glGetUniformLocation(ID, Name.c_str()), x, y, z, w);
    }
    // ------------------------------------------------------------------------
    inline void SetMat2(const String &Name, const Matrix2x2F &Mat) const
    {
        glUniformMatrix2fv(glGetUniformLocation(ID, Name.c_str()), 1, GL_FALSE, &Mat.data()[0]);
    }
    // ------------------------------------------------------------------------
    inline void SetMat3(const String &Name, const Matrix3x3F &Mat) const
    {
        glUniformMatrix3fv(glGetUniformLocation(ID, Name.c_str()), 1, GL_FALSE, &Mat.data()[0]);
    }
    // ------------------------------------------------------------------------
    inline void SetMat4(const String &Name, const Matrix4x4F &Mat) const
    {
        glUniformMatrix4fv(glGetUniformLocation(ID, Name.c_str()), 1, GL_FALSE, &Mat.data()[0]);
    }

private:
    inline void CheckCompileErrors(GLuint, String);
};
#endif
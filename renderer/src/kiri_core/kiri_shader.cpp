/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-20 19:15:04 
 * @Last Modified by:   Xu.Wang 
 * @Last Modified time: 2020-03-20 19:15:04 
 */
#include <kiri_core/kiri_shader.h>
#include <root_directory.h>

KiriShader::KiriShader(const String VertexPath, const String FragmentPath, const String GeometryPath)
{
    // 1. retrieve the vertex/fragment source code from filePath
    String vertexCode;
    String fragmentCode;
    String geometryCode;
    std::ifstream vShaderFile;
    std::ifstream fShaderFile;
    std::ifstream gShaderFile;
    // ensure ifstream objects can throw exceptions:
    vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    gShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
#ifdef KIRI_DEBUG_MODE
    String vp = String(MSWIN_BUILD_PATH) + "/Debug/shader/" + VertexPath;
    String fp = String(MSWIN_BUILD_PATH) + "/Debug/shader/" + FragmentPath;
#else
    // String vp = String(MSWIN_BUILD_PATH) + "/Release/shader/" + VertexPath;
    // String fp = String(MSWIN_BUILD_PATH) + "/Release/shader/" + FragmentPath;
    String vp = "shader/" + VertexPath;
    String fp = "shader/" + FragmentPath;
#endif

    try
    {
        vShaderFile.open(vp);
    }
    catch (std::ifstream::failure e)
    {
        KIRI_LOG_ERROR("SHADER::FILE:[{0}] NOT SUCCESFULLY READ ", vp);
    }

    try
    {
        fShaderFile.open(fp);
    }
    catch (std::ifstream::failure e)
    {
        KIRI_LOG_ERROR("SHADER::FILE:[{0}] NOT SUCCESFULLY READ ", fp);
    }

    std::stringstream vShaderStream, fShaderStream;
    // read file's buffer contents into streams
    vShaderStream << vShaderFile.rdbuf();
    fShaderStream << fShaderFile.rdbuf();
    // close file handlers
    vShaderFile.close();
    fShaderFile.close();
    // convert stream into String
    vertexCode = vShaderStream.str();
    fragmentCode = fShaderStream.str();

    // if geometry shader path is present, also load a geometry shader
    if (GeometryPath != "")
    {
#ifdef KIRI_DEBUG_MODE
        String gp = String(MSWIN_BUILD_PATH) + "/Debug/shader/" + GeometryPath;
#else
        //String gp = String(MSWIN_BUILD_PATH) + "/Release/shader/" + GeometryPath;
        String gp = "shader/" + GeometryPath;
#endif

        try
        {
            gShaderFile.open(gp);
        }
        catch (std::ifstream::failure e)
        {
            KIRI_LOG_ERROR("SHADER::FILE:[{0}] NOT SUCCESFULLY READ ", gp);
        }

        std::stringstream gShaderStream;
        gShaderStream << gShaderFile.rdbuf();
        gShaderFile.close();
        geometryCode = gShaderStream.str();
    }

    const char *vShaderCode = vertexCode.c_str();
    const char *fShaderCode = fragmentCode.c_str();
    // 2. compile shaders
    UInt vertex, fragment;
    // vertex shader
    vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vShaderCode, NULL);
    glCompileShader(vertex);
    CheckCompileErrors(vertex, "VERTEX");
    // fragment KiriShader
    fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fShaderCode, NULL);
    glCompileShader(fragment);
    CheckCompileErrors(fragment, "FRAGMENT");
    // if geometry shader is given, compile geometry shader
    UInt geometry = -1;
    if (GeometryPath != "")
    {
        const char *gShaderCode = geometryCode.c_str();
        geometry = glCreateShader(GL_GEOMETRY_SHADER);
        glShaderSource(geometry, 1, &gShaderCode, NULL);
        glCompileShader(geometry);
        CheckCompileErrors(geometry, "GEOMETRY");
    }
    // shader Program
    ID = glCreateProgram();
    glAttachShader(ID, vertex);
    glAttachShader(ID, fragment);
    if (GeometryPath != "")
        glAttachShader(ID, geometry);
    glLinkProgram(ID);
    CheckCompileErrors(ID, "PROGRAM");
    // delete the shaders as they're linked into our program now and no longer necessery
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    if (GeometryPath != "")
        glDeleteShader(geometry);
}

void KiriShader::CheckCompileErrors(GLuint shader, String type)
{
    GLint success;
    GLchar infoLog[1024];
    if (type != "PROGRAM")
    {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success)
        {
            glGetShaderInfoLog(shader, 1024, NULL, infoLog);
            KIRI_LOG_ERROR("SHADER_COMPILATION_ERROR of type:{0} ", type);
            KIRI_LOG_ERROR("{0}", infoLog);
            KIRI_LOG_ERROR(" -- --------------------------------------------------- -- ");
        }
    }
    else
    {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success)
        {
            glGetProgramInfoLog(shader, 1024, NULL, infoLog);
            KIRI_LOG_ERROR("SHADER_COMPILATION_ERROR of type:{0} ", type);
            KIRI_LOG_ERROR("{0}", infoLog);
            KIRI_LOG_ERROR(" -- --------------------------------------------------- -- ");
        }
    }
}
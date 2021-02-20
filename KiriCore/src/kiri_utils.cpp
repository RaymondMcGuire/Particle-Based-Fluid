/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-24 20:04:09
 * @LastEditTime: 2021-02-20 02:05:27
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\src\kiri_utils.cpp
 */

#include <kiri_utils.h>
#include <root_directory.h>

#include <partio/Partio.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stb_image.h>
#include <stb_image_write.h>

String KiriUtils::UInt2Str4Digit(UInt Input)
{
    char output[5];
    snprintf(output, 5, "%04d", Input);
    return String(output);
};

UInt KiriUtils::loadTexture(char const *path, bool gammaCorrection)
{
    UInt textureID;
    glGenTextures(1, &textureID);

    Int width, height, nrComponents;
    UChar *data = stbi_load(path, &width, &height, &nrComponents, 0);
    if (data)
    {
        GLenum internalFormat;
        GLenum dataFormat;
        if (nrComponents == 1)
        {
            internalFormat = dataFormat = GL_RED;
        }
        else if (nrComponents == 3)
        {
            internalFormat = gammaCorrection ? GL_SRGB : GL_RGB;
            dataFormat = GL_RGB;
        }
        else if (nrComponents == 4)
        {
            internalFormat = gammaCorrection ? GL_SRGB_ALPHA : GL_RGBA;
            dataFormat = GL_RGBA;
        }

        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, dataFormat, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, dataFormat == GL_RGBA ? GL_CLAMP_TO_EDGE : GL_REPEAT); // for this tutorial: use GL_CLAMP_TO_EDGE to prevent semi-transparent borders. Due to interpolation it takes texels from next repeat
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, dataFormat == GL_RGBA ? GL_CLAMP_TO_EDGE : GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        stbi_image_free(data);
    }
    else
    {
        KIRI_LOG_ERROR("Texture Failed to Load at Path:{0}", path);
        stbi_image_free(data);
    }

    return textureID;
};

UInt KiriUtils::loadTexture(const char *path, const String &directory, bool gammaCorrection)
{
    String filename = String(path);
    filename = directory + '/' + filename;

    UInt textureID;
    glGenTextures(1, &textureID);

    Int width, height, nrComponents;
    UChar *data = stbi_load(filename.c_str(), &width, &height, &nrComponents, 0);
    if (data)
    {
        GLenum internalFormat;
        GLenum dataFormat;
        if (nrComponents == 1)
        {
            internalFormat = dataFormat = GL_RED;
        }
        else if (nrComponents == 3)
        {
            internalFormat = gammaCorrection ? GL_SRGB : GL_RGB;
            dataFormat = GL_RGB;
        }
        else if (nrComponents == 4)
        {
            internalFormat = gammaCorrection ? GL_SRGB_ALPHA : GL_RGBA;
            dataFormat = GL_RGBA;
        }

        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, dataFormat, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        stbi_image_free(data);
    }
    else
    {
        KIRI_LOG_ERROR("Texture Failed to Load at Path:{0}", path);
        stbi_image_free(data);
    }

    return textureID;
};

void KiriUtils::printMatrix4x4F(Matrix4x4F _mat4x4, String _name)
{
    KIRI_LOG_INFO("---------- print Matrix4x4: {0} ----------", _name);

    for (size_t i = 0; i < _mat4x4.rows(); i++)
    {
        for (size_t j = 0; j < _mat4x4.cols(); j++)
        {
            std::cout << _mat4x4[i * _mat4x4.rows() + j] << " ";
        }
        std::cout << std::endl;
    }
}

void KiriUtils::flipVertically(Int width, Int height, char *data)
{
    char rgb[3];

    for (Int y = 0; y < height / 2; ++y)
    {
        for (Int x = 0; x < width; ++x)
        {
            Int top = (x + y * width) * 3;
            Int bottom = (x + (height - y - 1) * width) * 3;

            memcpy(rgb, data + top, sizeof(rgb));
            memcpy(data + top, data + bottom, sizeof(rgb));
            memcpy(data + bottom, rgb, sizeof(rgb));
        }
    }
}

Int KiriUtils::saveScreenshot(const char *filename)
{
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);

    Int x = viewport[0];
    Int y = viewport[1];
    Int width = viewport[2];
    Int height = viewport[3];

    char *data = (char *)malloc((size_t)(width * height * 3)); // 3 components (R, G, B)

    if (!data)
        return 0;

    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(x, y, width, height, GL_RGB, GL_UNSIGNED_BYTE, data);

    flipVertically(width, height, data);

    Int saved = stbi_write_png(filename, width, height, 3, data, 0);

    free(data);

    return saved;
}

const char *KiriUtils::createScreenshotBasename()
{
    static char basename[30];

    time_t t = time(NULL);
    struct tm timeStruct;
    errno_t error;
    error = localtime_s(&timeStruct, &t);

    strftime(basename, 30, "%Y%m%d_%H%M%S.png", &timeStruct);

    return basename;
}

const char *KiriUtils::createBasenameForVideo(Int cnt, const char *ext, const char *prefix)
{
    static char basename[30];

    snprintf(basename, sizeof(basename), "%s%04d.%s", prefix, cnt, ext);

    return basename;
}

Int KiriUtils::captureScreenshot(Int cnt)
{
    char buildRootPath[200];
    strcpy_s(buildRootPath, 200, ROOT_PATH);
    strcat_s(buildRootPath, sizeof(buildRootPath), "/export/screenshots/");
    strcat_s(buildRootPath, sizeof(buildRootPath), createBasenameForVideo(cnt, "png", ""));

    Int saved = saveScreenshot(buildRootPath);

    if (saved)
        KIRI_LOG_INFO("Successfully Saved Image:{0}", buildRootPath);
    else
        KIRI_LOG_ERROR("Failed Saving Image:{0}", buildRootPath);

    return saved;
}

std::vector<float4> KiriUtils::ReadBgeoFileForGPU(String Folder, String Name, bool FlipYZ)
{
    String root_folder = "bgeo";
    String extension = ".bgeo";
    String file_path = String(DB_PBR_PATH) + root_folder + "/" + Folder + "/" + Name + extension;
    Partio::ParticlesDataMutable *data = Partio::read(file_path.c_str());

    Partio::ParticleAttribute pos_attr;
    Partio::ParticleAttribute pscale_attr;
    if (!data->attributeInfo("position", pos_attr) || (pos_attr.type != Partio::FLOAT && pos_attr.type != Partio::VECTOR) || pos_attr.count != 3)
    {
        KIRI_LOG_ERROR("Failed to Get Proper Position Attribute");
    }

    bool pscaleLoaded = data->attributeInfo("pscale", pscale_attr);

    std::vector<float4> pos_array;
    for (Int i = 0; i < data->numParticles(); i++)
    {
        const float *pos = data->data<float>(pos_attr, i);
        if (pscaleLoaded)
        {
            const float *pscale = data->data<float>(pscale_attr, i);
            if (i == 0)
            {
                KIRI_LOG_INFO("pscale={0}", *pscale);
            }

            if (FlipYZ)
            {
                pos_array.push_back(make_float4(pos[0], pos[2], pos[1], *pscale));
            }
            else
            {
                pos_array.push_back(make_float4(pos[0], pos[1], pos[2], *pscale));
            }
        }
        else
        {
            if (FlipYZ)
            {
                pos_array.push_back(make_float4(pos[0], pos[2], pos[1], 0.01f));
            }
            else
            {
                pos_array.push_back(make_float4(pos[0], pos[1], pos[2], 0.01f));
            }
        }
    }

    data->release();

    return pos_array;
}

std::vector<float4> KiriUtils::ReadMultiBgeoFilesForGPU(String Folder, Vec_String Names, bool FlipYZ)
{
    String root_folder = "bgeo";
    String extension = ".bgeo";

    std::vector<float4> pos_array;
    for (size_t n = 0; n < Names.size(); n++)
    {
        String file_path = String(DB_PBR_PATH) + root_folder + "/" + Folder + "/" + Names[n] + extension;
        Partio::ParticlesDataMutable *data = Partio::read(file_path.c_str());

        Partio::ParticleAttribute pos_attr;
        Partio::ParticleAttribute pscale_attr;
        if (!data->attributeInfo("position", pos_attr) || (pos_attr.type != Partio::FLOAT && pos_attr.type != Partio::VECTOR) || pos_attr.count != 3)
        {
            KIRI_LOG_ERROR("File={0}, Failed to Get Proper Position Attribute", Names[n]);
        }

        bool pscaleLoaded = data->attributeInfo("pscale", pscale_attr);

        for (Int i = 0; i < data->numParticles(); i++)
        {
            const float *pos = data->data<float>(pos_attr, i);
            if (pscaleLoaded)
            {
                const float *pscale = data->data<float>(pscale_attr, i);
                if (i == 0)
                {
                    KIRI_LOG_INFO("pscale={0}", *pscale);
                }

                if (FlipYZ)
                {
                    pos_array.push_back(make_float4(pos[0], pos[2], pos[1], *pscale));
                }
                else
                {
                    pos_array.push_back(make_float4(pos[0], pos[1], pos[2], *pscale));
                }
            }
            else
            {
                if (FlipYZ)
                {
                    pos_array.push_back(make_float4(pos[0], pos[2], pos[1], 0.01f));
                }
                else
                {
                    pos_array.push_back(make_float4(pos[0], pos[1], pos[2], 0.01f));
                }
            }
        }

        KIRI_LOG_INFO("Loaded Bgeo File={0}, Number of Particles={1}", Names[n], data->numParticles());

        data->release();
    }

    return pos_array;
}

Array1Vec4F KiriUtils::ReadBgeoFileForCPU(String Folder, String Name, Vector3F Offset, bool FlipYZ)
{
    String root_folder = "bgeo";
    String extension = ".bgeo";
    String file_path = String(DB_PBR_PATH) + root_folder + "/" + Folder + "/" + Name + extension;
    Partio::ParticlesDataMutable *data = Partio::read(file_path.c_str());

    Partio::ParticleAttribute pos_attr;
    Partio::ParticleAttribute pscale_attr;
    if (!data->attributeInfo("position", pos_attr) || (pos_attr.type != Partio::FLOAT && pos_attr.type != Partio::VECTOR) || pos_attr.count != 3)
    {
        KIRI_LOG_ERROR("Failed to Get Proper Position Attribute");
    }

    bool pscaleLoaded = data->attributeInfo("pscale", pscale_attr);

    float max_y = 0.f;
    Array1Vec4F pos_array;
    for (Int i = 0; i < data->numParticles(); i++)
    {
        const float *pos = data->data<float>(pos_attr, i);
        if (pscaleLoaded)
        {
            const float *pscale = data->data<float>(pscale_attr, i);
            if (i == 0)
            {
                KIRI_LOG_INFO("pscale={0}", *pscale);
            }

            if (FlipYZ)
            {
                pos_array.append(Vector4F(pos[0] + Offset.x, pos[2] + Offset.z, pos[1] + Offset.y, *pscale));
            }
            else
            {
                pos_array.append(Vector4F(pos[0] + Offset.x, pos[1] + Offset.y, pos[2] + Offset.z, *pscale));
                if (pos[1] > max_y)
                    max_y = pos[1];
            }
        }
        else
        {
            if (FlipYZ)
            {
                pos_array.append(Vector4F(pos[0] + Offset.x, pos[2] + Offset.z, pos[1] + Offset.y, 0.01f));
            }
            else
            {
                pos_array.append(Vector4F(pos[0] + Offset.x, pos[1] + Offset.y, pos[2] + Offset.z, 0.01f));
            }
        }
    }

    //printf("max_Y=%.3f , offset=(%.3f,%.3f,%.3f) \n", max_y, Offset.x, Offset.y, Offset.z);

    data->release();

    return pos_array;
}

void KiriUtils::ExportBgeoFileFromGPU(String Folder, String FileName, float4 *Positions, float4 *Colors, uint *Labels, UInt NumOfParticles)
{
    String exportPath = String(EXPORT_PATH) + "bgeo/" + Folder + "/" + FileName + ".bgeo";

    Partio::ParticlesDataMutable *p = Partio::create();
    Partio::ParticleAttribute positionAttr = p->addAttribute("position", Partio::VECTOR, 3);
    Partio::ParticleAttribute colorAttr = p->addAttribute("Cd", Partio::FLOAT, 3);
    Partio::ParticleAttribute pScaleAttr = p->addAttribute("pscale", Partio::FLOAT, 1);
    Partio::ParticleAttribute labelAttr = p->addAttribute("label", Partio::INT, 1);

    // transfer GPU data to CPU
    uint f4Bytes = NumOfParticles * sizeof(float4);
    uint uintBytes = NumOfParticles * sizeof(uint);

    float4 *cpuPositions = (float4 *)malloc(f4Bytes);
    float4 *cpuColors = (float4 *)malloc(f4Bytes);
    uint *cpuLabels = (uint *)malloc(uintBytes);

    cudaMemcpy(cpuPositions, Positions, f4Bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpuColors, Colors, f4Bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpuLabels, Labels, uintBytes, cudaMemcpyDeviceToHost);

    for (UInt i = 0; i < NumOfParticles; i++)
    {
        Int particle = p->addParticle();
        float *pos = p->dataWrite<float>(positionAttr, particle);
        float *col = p->dataWrite<float>(colorAttr, particle);
        float *pscale = p->dataWrite<float>(pScaleAttr, particle);
        int *label = p->dataWrite<int>(labelAttr, particle);

        pos[0] = cpuPositions[i].x;
        pos[1] = cpuPositions[i].y;
        pos[2] = cpuPositions[i].z;
        col[0] = cpuColors[i].x;
        col[1] = cpuColors[i].y;
        col[2] = cpuColors[i].z;

        // TODO
        *pscale = cpuPositions[i].w;

        *label = cpuLabels[i];
    }
    Partio::write(exportPath.c_str(), *p);

    p->release();

    free(cpuPositions);
    free(cpuColors);
    free(cpuLabels);

    KIRI_LOG_INFO("Successfully Saved Bgeo File:{0}", exportPath);
}

void KiriUtils::ExportBgeoFileFromCPU(String Folder, String FileName, Array1Vec4F Positions)
{
    String exportPath = String(EXPORT_PATH) + "bgeo/" + Folder + "/" + FileName + ".bgeo";

    Partio::ParticlesDataMutable *p = Partio::create();
    Partio::ParticleAttribute positionAttr = p->addAttribute("position", Partio::VECTOR, 3);
    Partio::ParticleAttribute pScaleAttr = p->addAttribute("pscale", Partio::FLOAT, 1);

    for (UInt i = 0; i < Positions.size(); i++)
    {
        Int particle = p->addParticle();
        float *pos = p->dataWrite<float>(positionAttr, particle);
        float *pscale = p->dataWrite<float>(pScaleAttr, particle);
        pos[0] = Positions[i].x;
        pos[1] = Positions[i].y;
        pos[2] = Positions[i].z;

        // TODO
        *pscale = Positions[i].w;
    }
    Partio::write(exportPath.c_str(), *p);

    p->release();
}

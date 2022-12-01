/***
 * @Author: Xu.WANG
 * @Date: 2020-10-24 20:04:09
 * @LastEditTime: 2021-09-14 18:38:50
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath: \Kiri\KiriCore\src\kiri_utils.cpp
 */

#include <kiri_utils.h>
#include <root_directory.h>

#include <partio/Partio.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stb_image.h>
#include <stb_image_write.h>

#include <filesystem>
#include <sys/types.h>
#include <sys/stat.h>

#include <tuple>

#define TINYOBJLOADER_IMPLEMENTATION

String KiriUtils::UInt2Str4Digit(UInt Input)
{
    char output[5];
    snprintf(output, 5, "%04d", Input);
    return String(output);
};

void KiriUtils::LoadVoroFile(
    std::vector<Vector3F> &position,
    std::vector<Vector3F> &mNormal,
    std::vector<int> &mIndices,
    std::string file_name)
{
    String file_path = String(DB_PBR_PATH) + "voro/" + file_name + ".voro";
    std::ifstream file(file_path.c_str());
    auto size_pos = 0, size_n = 0, size_ind = 0;
    file >> size_pos >> size_n >> size_ind;

    for (int i = 0; i < size_pos; ++i)
    {
        Vector3F pos;
        file >> pos.x >> pos.y >> pos.z;
        position.emplace_back(pos);
    }

    for (int i = 0; i < size_n; ++i)
    {
        Vector3F n;
        file >> n.x >> n.y >> n.z;
        mNormal.emplace_back(n);
    }

    for (int i = 0; i < size_ind; ++i)
    {
        int ind;
        file >> ind;
        mIndices.emplace_back(ind);
    }

    file.close();
};

UInt KiriUtils::LoadTexture(char const *path, bool gammaCorrection)
{
    UInt textureID;
    glGenTextures(1, &textureID);

    Int mWidth, height, nrComponents;
    UChar *data = stbi_load(path, &mWidth, &height, &nrComponents, 0);
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
        glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, mWidth, height, 0, dataFormat, GL_UNSIGNED_BYTE, data);
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

UInt KiriUtils::LoadTexture(const char *path, const String &directory, bool gammaCorrection)
{
    String filename = String(path);
    filename = directory + '/' + filename;

    UInt textureID;
    glGenTextures(1, &textureID);

    Int mWidth, height, nrComponents;
    UChar *data = stbi_load(filename.c_str(), &mWidth, &height, &nrComponents, 0);
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
        glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, mWidth, height, 0, dataFormat, GL_UNSIGNED_BYTE, data);
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

void get_bbox(const std::vector<float> &vertices, float &xmin, float &ymin, float &zmin, float &xmax, float &ymax, float &zmax)
{
    int nb_v = vertices.size() / 3;
    xmin = xmax = vertices[0];
    ymin = ymax = vertices[1];
    zmin = zmax = vertices[2];
    for (int i = 1; i < nb_v; ++i)
    {
        xmin = std::min(xmin, vertices[3 * i]);
        ymin = std::min(ymin, vertices[3 * i + 1]);
        zmin = std::min(zmin, vertices[3 * i + 2]);
        xmax = std::max(xmax, vertices[3 * i]);
        ymax = std::max(ymax, vertices[3 * i + 1]);
        zmax = std::max(zmax, vertices[3 * i + 2]);
    }
    float d = xmax - xmin;
    d = std::max(d, ymax - ymin);
    d = std::max(d, zmax - zmin);
    d = 0.001f * d;
    xmin -= d;
    ymin -= d;
    zmin -= d;
    xmax += d;
    ymax += d;
    zmax += d;
}

bool KiriUtils::ReadTetFile(
    const String &filename,
    Vec_Float &vertices,
    Vec_Int &mIndices,
    bool normalize,
    float scale)
{
    String s;
    int n_vertex, n_tet, temp;

    String root_folder = "tet";
    String file_path = String(DB_PBR_PATH) + root_folder + "/" + filename;
    std::ifstream input(file_path);
    if (input.fail())
        return false;

    String ext = filename.substr(filename.find_last_of('.') + 1);
    if (ext == "tet")
    {
        input >> n_vertex;
        std::getline(input, s);
        input >> n_tet;
        std::getline(input, s);

        vertices.resize(3 * n_vertex);
        mIndices.resize(n_tet << 2);

        for (int i = 0; i < n_vertex; ++i)
            input >> vertices[3 * i] >> vertices[3 * i + 1] >> vertices[3 * i + 2];

        for (int i = 0; i < n_tet; ++i)
        {
            input >> temp >> mIndices[(i << 2)] >> mIndices[(i << 2) + 1] >> mIndices[(i << 2) + 2] >> mIndices[(i << 2) + 3];
            assert(temp == 4);
        }
    }
    else if (ext == "vtk")
    {
        for (int i = 0; i < 4; ++i)
            std::getline(input, s); // skip first 4 lines

        input >> s >> n_vertex >> s;
        vertices.resize(3 * n_vertex);
        for (int i = 0; i < n_vertex; ++i)
            input >> vertices[3 * i] >> vertices[3 * i + 1] >> vertices[3 * i + 2];

        std::getline(input, s);
        input >> s >> n_tet >> s;
        std::cerr << "n_tet:" << n_tet << std::endl;
        mIndices.resize(n_tet << 2);
        for (int i = 0; i < n_tet; ++i)
        {
            input >> temp >> mIndices[(i << 2)] >> mIndices[(i << 2) + 1] >> mIndices[(i << 2) + 2] >> mIndices[(i << 2) + 3];
            assert(temp == 4);
        }
    }
    else
    {
        input.close();
        return false;
    }

    input.close();

    float xmin, ymin, zmin, xmax, ymax, zmax;
    get_bbox(vertices, xmin, ymin, zmin, xmax, ymax, zmax);

    if (normalize) // normalize vertices between [0,scale]^3
    {
        float maxside = std::max(std::max(xmax - xmin, ymax - ymin), zmax - zmin);

        for (int i = 0; i < n_vertex; i++)
        {
            vertices[3 * i] = scale * (vertices[3 * i] - xmin) / maxside;
            vertices[3 * i + 1] = scale * (vertices[3 * i + 1] - ymin) / maxside;
            vertices[3 * i + 2] = scale * (vertices[3 * i + 2] - zmin) / maxside;
        }
        get_bbox(vertices, xmin, ymin, zmin, xmax, ymax, zmax);
        // std::cerr << "bbox [" << xmin << ":" << xmax << "], [" << ymin << ":" << ymax << "], [" << zmin << ":" << zmax << "]" << std::endl;
    }

    return true;
}

Array1Vec4F KiriUtils::ReadBgeoFileForCPU(String Folder, String Name, Vector3F Offset, bool FlipYZ)
{
    Array1Vec4F pos_array;
    String root_folder = "bgeo";
    String extension = ".bgeo";
    String file_path = String(DB_PBR_PATH) + root_folder + "/" + Folder + "/" + Name + extension;
    std::cout << file_path << std::endl;
    Partio::ParticlesDataMutable *data = Partio::read(file_path.c_str());

    Partio::ParticleAttribute pos_attr;
    Partio::ParticleAttribute pscale_attr;
    if (!data->attributeInfo("position", pos_attr) || (pos_attr.type != Partio::FLOAT && pos_attr.type != Partio::VECTOR) || pos_attr.count != 3)
    {
        KIRI_LOG_ERROR("Failed to Get Proper Position Attribute");
    }

    bool pscaleLoaded = data->attributeInfo("pscale", pscale_attr);

    float max_y = 0.f;
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

    // printf("max_Y=%.3f , offset=(%.3f,%.3f,%.3f) \n", max_y, Offset.x, Offset.y, Offset.z);

    data->release();

    return pos_array;
}

void KiriUtils::ReadParticlesData(Vec_Vec4F &positions, Vec_Float &masses, String Folder, String Name, Vector3F Offset, bool FlipYZ)
{
    positions.clear();
    masses.clear();

    String root_folder = "bgeo";
    String extension = ".bgeo";
    String file_path = String(DB_PBR_PATH) + root_folder + "/" + Folder + "/" + Name + extension;
    std::cout << file_path << std::endl;
    Partio::ParticlesDataMutable *data = Partio::read(file_path.c_str());

    Partio::ParticleAttribute pos_attr;
    Partio::ParticleAttribute pscale_attr;
    Partio::ParticleAttribute mass_attr;
    if (!data->attributeInfo("position", pos_attr) || (pos_attr.type != Partio::FLOAT && pos_attr.type != Partio::VECTOR) || pos_attr.count != 3)
    {
        KIRI_LOG_ERROR("Failed to Get Proper Position Attribute");
    }

    bool pscaleLoaded = data->attributeInfo("pscale", pscale_attr);
    bool massLoaded = data->attributeInfo("mass", mass_attr);

    for (Int i = 0; i < data->numParticles(); i++)
    {
        const float *pos = data->data<float>(pos_attr, i);
        const float *mass = data->data<float>(mass_attr, i);
        masses.emplace_back(*mass);

        if (pscaleLoaded)
        {
            const float *pscale = data->data<float>(pscale_attr, i);

            if (FlipYZ)
            {
                positions.emplace_back(Vector4F(pos[0] + Offset.x, pos[2] + Offset.z, pos[1] + Offset.y, *pscale));
            }
            else
            {
                positions.emplace_back(Vector4F(pos[0] + Offset.x, pos[1] + Offset.y, pos[2] + Offset.z, *pscale));
            }
        }
        else
        {
            if (FlipYZ)
            {
                positions.emplace_back(Vector4F(pos[0] + Offset.x, pos[2] + Offset.z, pos[1] + Offset.y, 0.01f));
            }
            else
            {
                positions.emplace_back(Vector4F(pos[0] + Offset.x, pos[1] + Offset.y, pos[2] + Offset.z, 0.01f));
            }
        }
    }

    data->release();
}

String KiriUtils::GetDefaultExportPath() { return String(EXPORT_PATH); }

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

void KiriUtils::WriteMultiSizedParticles(String Folder, String FileName, Array1Vec4F Positions, Vec_Float masses)
{
    String exportPath = String(EXPORT_PATH) + "bgeo/" + Folder + "/" + FileName + ".bgeo";

    Partio::ParticlesDataMutable *p = Partio::create();
    Partio::ParticleAttribute positionAttr = p->addAttribute("position", Partio::VECTOR, 3);
    Partio::ParticleAttribute pScaleAttr = p->addAttribute("pscale", Partio::FLOAT, 1);
    Partio::ParticleAttribute massAttr = p->addAttribute("mass", Partio::FLOAT, 1);

    for (UInt i = 0; i < Positions.size(); i++)
    {
        Int particle = p->addParticle();
        float *pos = p->dataWrite<float>(positionAttr, particle);
        float *pscale = p->dataWrite<float>(pScaleAttr, particle);
        float *mass = p->dataWrite<float>(massAttr, particle);

        pos[0] = Positions[i].x;
        pos[1] = Positions[i].y;
        pos[2] = Positions[i].z;

        *pscale = Positions[i].w;
        *mass = masses[i];
    }
    Partio::write(exportPath.c_str(), *p);

    p->release();
}

void KiriUtils::ExportBgeoFileFromCPU(String Folder, String FileName, Array1Vec4F Positions, Array1<float> Mass)
{
    String exportPath = String(EXPORT_PATH) + "bgeo/" + Folder + "/" + FileName + ".bgeo";

    Partio::ParticlesDataMutable *p = Partio::create();
    Partio::ParticleAttribute positionAttr = p->addAttribute("position", Partio::VECTOR, 3);
    Partio::ParticleAttribute pScaleAttr = p->addAttribute("pscale", Partio::FLOAT, 1);
    Partio::ParticleAttribute massAttr = p->addAttribute("pmass", Partio::FLOAT, 1);

    for (UInt i = 0; i < Positions.size(); i++)
    {
        Int particle = p->addParticle();
        float *pos = p->dataWrite<float>(positionAttr, particle);
        float *pscale = p->dataWrite<float>(pScaleAttr, particle);
        float *mass = p->dataWrite<float>(massAttr, particle);
        pos[0] = Positions[i].x;
        pos[1] = Positions[i].y;
        pos[2] = Positions[i].z;

        // radius
        *pscale = Positions[i].w;

        *mass = Mass[i];
    }
    Partio::write(exportPath.c_str(), *p);

    p->release();
}

void KiriUtils::ExportXYFile(const Array1Vec2F &points, const String fileName, bool normlization)
{
    auto minValue = Huge<float>();
    auto maxValue = Tiny<float>();
    if (normlization)
    {
        points.forEach([&](Vector2F elem)
                       {
                           minValue = std::min(minValue, std::min(elem.x, elem.y));
                           maxValue = std::max(maxValue, std::max(elem.x, elem.y)); });
    }

    String exportPath = String(EXPORT_PATH) + "xy/" + fileName + ".xy";
    std::fstream file;
    file.open(exportPath.c_str(), std::ios_base::out);
    file << points.size() << std::endl;
    for (int i = 0; i < points.size(); i++)
        if (normlization)
            file << points[i].x / maxValue << "  " << points[i].y / maxValue << std::endl;
        else
            file << points[i].x << "  " << points[i].y << std::endl;
    file.close();
}

void KiriUtils::ExportXYZFile(const Array1Vec3F &points, const String fileName)
{
    String exportPath = String(EXPORT_PATH) + "xyz/" + fileName + ".xyz";

    std::fstream file;
    file.open(exportPath.c_str(), std::ios_base::out);
    file << points.size() << std::endl;
    for (int i = 0; i < points.size(); i++)
        file << points[i].x << "  " << points[i].y << "  " << points[i].z << std::endl;
    file.close();
}

bool WriteMat(const std::string &filename, const std::vector<tinyobj::material_t> &materials)
{
    FILE *fp = fopen(filename.c_str(), "w");
    if (!fp)
    {
        fprintf(stderr, "Failed to open file [ %s ] for write.\n", filename.c_str());
        return false;
    }

    for (size_t i = 0; i < materials.size(); i++)
    {

        tinyobj::material_t mat = materials[i];

        fprintf(fp, "newmtl %s\n", mat.name.c_str());
        fprintf(fp, "Ka %f %f %f\n", mat.ambient[0], mat.ambient[1], mat.ambient[2]);
        fprintf(fp, "Kd %f %f %f\n", mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);
        fprintf(fp, "Ks %f %f %f\n", mat.specular[0], mat.specular[1], mat.specular[2]);
        fprintf(fp, "Kt %f %f %f\n", mat.transmittance[0], mat.specular[1], mat.specular[2]);
        fprintf(fp, "Ke %f %f %f\n", mat.emission[0], mat.emission[1], mat.emission[2]);
        fprintf(fp, "Ns %f\n", mat.shininess);
        fprintf(fp, "Ni %f\n", mat.ior);
        fprintf(fp, "illum %d\n", mat.illum);
        fprintf(fp, "\n");
        // @todo { texture }
    }

    fclose(fp);

    return true;
}

bool KiriUtils::TinyObjWriter(const String &filename, const tinyobj::attrib_t &attributes, const Vector<tinyobj::shape_t> &shapes, const Vector<tinyobj::material_t> &materials, bool coordTransform)
{
    String exportPath = String(EXPORT_PATH) + "obj/" + filename + ".obj";

    FILE *fp = fopen(exportPath.c_str(), "w");
    if (!fp)
    {
        fprintf(stderr, "Failed to open file [ %s ] for write.\n", exportPath.c_str());
        return false;
    }

    std::string basename = filename;
    std::string material_filename = basename + ".mtl";

    int prev_material_id = -1;

    fprintf(fp, "mtllib %s\n\n", material_filename.c_str());

    // facevarying vtx
    for (size_t k = 0; k < attributes.vertices.size(); k += 3)
    {
        if (coordTransform)
        {
            fprintf(fp, "v %f %f %f\n",
                    attributes.vertices[k + 0],
                    attributes.vertices[k + 2],
                    -attributes.vertices[k + 1]);
        }
        else
        {
            fprintf(fp, "v %f %f %f\n",
                    attributes.vertices[k + 0],
                    attributes.vertices[k + 1],
                    attributes.vertices[k + 2]);
        }
    }

    fprintf(fp, "\n");

    // facevarying mNormal
    for (size_t k = 0; k < attributes.normals.size(); k += 3)
    {
        if (coordTransform)
        {
            fprintf(fp, "vn %f %f %f\n",
                    attributes.normals[k + 0],
                    attributes.normals[k + 2],
                    -attributes.normals[k + 1]);
        }
        else
        {
            fprintf(fp, "vn %f %f %f\n",
                    attributes.normals[k + 0],
                    attributes.normals[k + 1],
                    attributes.normals[k + 2]);
        }
    }

    fprintf(fp, "\n");

    // facevarying texcoord
    for (size_t k = 0; k < attributes.texcoords.size(); k += 2)
    {
        fprintf(fp, "vt %f %f\n",
                attributes.texcoords[k + 0],
                attributes.texcoords[k + 1]);
    }

    for (size_t i = 0; i < shapes.size(); i++)
    {
        fprintf(fp, "\n");

        if (shapes[i].name.empty())
        {
            fprintf(fp, "g Unknown\n");
        }
        else
        {
            fprintf(fp, "g %s\n", shapes[i].name.c_str());
        }

        bool has_vn = false;
        bool has_vt = false;
        // Assumes normals and mTextures are set shape-wise.
        if (shapes[i].mesh.indices.size() > 0)
        {
            has_vn = shapes[i].mesh.indices[0].normal_index != -1;
            has_vt = shapes[i].mesh.indices[0].texcoord_index != -1;
        }

        // face
        int face_index = 0;
        for (size_t k = 0; k < shapes[i].mesh.indices.size(); k += shapes[i].mesh.num_face_vertices[face_index++])
        {
            // Check Materials
            int material_id = shapes[i].mesh.material_ids[face_index];
            if (material_id != prev_material_id)
            {
                std::string material_name = materials[material_id].name;
                fprintf(fp, "usemtl %s\n", material_name.c_str());
                prev_material_id = material_id;
            }

            unsigned char v_per_f = shapes[i].mesh.num_face_vertices[face_index];
            // Imperformant, but if you want to have variable vertices per face, you need some kind of a dynamic loop.
            fprintf(fp, "f");
            for (int l = 0; l < v_per_f; l++)
            {
                const tinyobj::index_t &ref = shapes[i].mesh.indices[k + l];
                if (has_vn && has_vt)
                {
                    // v0/t0/vn0
                    fprintf(fp, " %d/%d/%d", ref.vertex_index + 1, ref.texcoord_index + 1, ref.normal_index + 1);
                    continue;
                }
                if (has_vn && !has_vt)
                {
                    // v0//vn0
                    fprintf(fp, " %d//%d", ref.vertex_index + 1, ref.normal_index + 1);
                    continue;
                }
                if (!has_vn && has_vt)
                {
                    // v0/vt0
                    fprintf(fp, " %d/%d", ref.vertex_index + 1, ref.texcoord_index + 1);
                    continue;
                }
                if (!has_vn && !has_vt)
                {
                    // v0 v1 v2
                    fprintf(fp, " %d", ref.vertex_index + 1);
                    continue;
                }
            }
            fprintf(fp, "\n");
        }
    }

    fclose(fp);

    //
    // Write material file
    //
    bool ret = WriteMat(material_filename, materials);

    return ret;
}
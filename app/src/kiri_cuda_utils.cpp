/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-08-24 14:09:05
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-09-15 14:43:18
 * @FilePath: \Kiri\KiriExamples\src\kiri_cuda_utils.cpp
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
// clang-format off
#include <kiri_cuda_utils.h>
#include <root_directory.h>

#include <partio/Partio.h>


#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <stb_image.h>
#include <stb_image_write.h>

#include <filesystem>
#include <sys/stat.h>
#include <sys/types.h>

#include <tuple>
// clang-format on

std::vector<float4>
KiriCudaUtils::ReadBgeoFileForGPU(String Folder, String Name, bool FlipYZ) {
  String root_folder = "bgeo";
  String extension = ".bgeo";
  String file_path =
      String(DB_PBR_PATH) + root_folder + "/" + Folder + "/" + Name + extension;
  Partio::ParticlesDataMutable *data = Partio::read(file_path.c_str());

  Partio::ParticleAttribute pos_attr;
  Partio::ParticleAttribute pscale_attr;
  if (!data->attributeInfo("position", pos_attr) ||
      (pos_attr.type != Partio::FLOAT && pos_attr.type != Partio::VECTOR) ||
      pos_attr.count != 3) {
    KIRI_LOG_ERROR("Failed to Get Proper Position Attribute");
  }

  bool pscaleLoaded = data->attributeInfo("pscale", pscale_attr);

  std::vector<float4> pos_array;
  for (Int i = 0; i < data->numParticles(); i++) {
    const float *pos = data->data<float>(pos_attr, i);
    if (pscaleLoaded) {
      const float *pscale = data->data<float>(pscale_attr, i);
      if (i == 0) {
        KIRI_LOG_INFO("pscale={0}", *pscale);
      }

      if (FlipYZ) {
        pos_array.push_back(make_float4(pos[0], pos[2], pos[1], *pscale));
      } else {
        pos_array.push_back(make_float4(pos[0], pos[1], pos[2], *pscale));
      }
    } else {
      if (FlipYZ) {
        pos_array.push_back(make_float4(pos[0], pos[2], pos[1], 0.01f));
      } else {
        pos_array.push_back(make_float4(pos[0], pos[1], pos[2], 0.01f));
      }
    }
  }

  data->release();

  return pos_array;
}

std::vector<float4> KiriCudaUtils::ReadMultiBgeoFilesForGPU(Vec_String Folders,
                                                            Vec_String Names,
                                                            bool FlipYZ) {
  String root_folder = "bgeo";
  String extension = ".bgeo";

  std::vector<float4> pos_array;
  for (size_t n = 0; n < Folders.size(); n++) {
    String file_path = String(DB_PBR_PATH) + root_folder + "/" + Folders[n] +
                       "/" + Names[n] + extension;
    Partio::ParticlesDataMutable *data = Partio::read(file_path.c_str());

    Partio::ParticleAttribute pos_attr;
    Partio::ParticleAttribute pscale_attr;
    if (!data->attributeInfo("position", pos_attr) ||
        (pos_attr.type != Partio::FLOAT && pos_attr.type != Partio::VECTOR) ||
        pos_attr.count != 3) {
      KIRI_LOG_ERROR("File={0}, Failed to Get Proper Position Attribute",
                     Names[n]);
    }

    bool pscaleLoaded = data->attributeInfo("pscale", pscale_attr);

    for (Int i = 0; i < data->numParticles(); i++) {
      const float *pos = data->data<float>(pos_attr, i);
      if (pscaleLoaded) {
        const float *pscale = data->data<float>(pscale_attr, i);

        // KIRI_LOG_INFO("pscale={0};pos={1},{2},{3}", *pscale, pos[0], pos[2],
        //               pos[1]);

        if (FlipYZ) {
          pos_array.push_back(make_float4(pos[0], pos[2], pos[1], *pscale));
        } else {
          pos_array.push_back(make_float4(pos[0], pos[1], pos[2], *pscale));
        }
      } else {
        if (FlipYZ) {
          pos_array.push_back(make_float4(pos[0], pos[2], pos[1], 0.01f));
        } else {
          pos_array.push_back(make_float4(pos[0], pos[1], pos[2], 0.01f));
        }
      }
    }

    KIRI_LOG_INFO("Loaded Bgeo File={0}, Number of Particles={1}", Names[n],
                  data->numParticles());

    data->release();
  }

  return pos_array;
}

std::pair<std::vector<float4>, std::vector<float>>
KiriCudaUtils::ReadBgeoFileWithMassForGPU(String Folder, String Name,
                                          bool FlipYZ) {
  String root_folder = "bgeo";
  String extension = ".bgeo";
  String file_path =
      String(DB_PBR_PATH) + root_folder + "/" + Folder + "/" + Name + extension;
  Partio::ParticlesDataMutable *data = Partio::read(file_path.c_str());

  Partio::ParticleAttribute pos_attr;
  Partio::ParticleAttribute pscale_attr;
  Partio::ParticleAttribute pmass_attr;
  if (!data->attributeInfo("position", pos_attr) ||
      (pos_attr.type != Partio::FLOAT && pos_attr.type != Partio::VECTOR) ||
      pos_attr.count != 3) {
    KIRI_LOG_ERROR("Failed to Get Proper Position Attribute");
  }

  bool pscaleLoaded = data->attributeInfo("pscale", pscale_attr);
  bool massLoaded = data->attributeInfo("mass", pmass_attr);

  std::vector<float4> pos_array;
  std::vector<float> mass_array;
  for (Int i = 0; i < data->numParticles(); i++) {
    const float *pos = data->data<float>(pos_attr, i);
    const float *pmass = data->data<float>(pmass_attr, i);
    mass_array.push_back(*pmass);
    if (pscaleLoaded) {
      const float *pscale = data->data<float>(pscale_attr, i);
      if (i == 0) {
        KIRI_LOG_INFO("pscale={0}", *pscale);
      }

      if (FlipYZ) {
        pos_array.push_back(make_float4(pos[0], pos[2], pos[1], *pscale));
      } else {
        pos_array.push_back(make_float4(pos[0], pos[1], pos[2], *pscale));
      }
    } else {
      if (FlipYZ) {
        pos_array.push_back(make_float4(pos[0], pos[2], pos[1], 0.01f));
      } else {
        pos_array.push_back(make_float4(pos[0], pos[1], pos[2], 0.01f));
      }
    }
  }

  data->release();

  return {pos_array, mass_array};
}

std::vector<float4> KiriCudaUtils::ReadMultiBgeoFilesForGPU(String Folder,
                                                            Vec_String Names,
                                                            bool FlipYZ) {
  String root_folder = "bgeo";
  String extension = ".bgeo";

  std::vector<float4> pos_array;
  for (size_t n = 0; n < Names.size(); n++) {
    String file_path = String(DB_PBR_PATH) + root_folder + "/" + Folder + "/" +
                       Names[n] + extension;
    Partio::ParticlesDataMutable *data = Partio::read(file_path.c_str());

    Partio::ParticleAttribute pos_attr;
    Partio::ParticleAttribute pscale_attr;
    if (!data->attributeInfo("position", pos_attr) ||
        (pos_attr.type != Partio::FLOAT && pos_attr.type != Partio::VECTOR) ||
        pos_attr.count != 3) {
      KIRI_LOG_ERROR("File={0}, Failed to Get Proper Position Attribute",
                     Names[n]);
    }

    bool pscaleLoaded = data->attributeInfo("pscale", pscale_attr);

    for (Int i = 0; i < data->numParticles(); i++) {
      const float *pos = data->data<float>(pos_attr, i);
      if (pscaleLoaded) {
        const float *pscale = data->data<float>(pscale_attr, i);
        // KIRI_LOG_INFO("pscale={0};pos={1},{2},{3}", *pscale, pos[0], pos[2],
        //               pos[1]);

        if (FlipYZ) {
          pos_array.push_back(make_float4(pos[0], pos[2], pos[1], *pscale));
        } else {
          pos_array.push_back(make_float4(pos[0], pos[1], pos[2], *pscale));
        }
      } else {
        if (FlipYZ) {
          pos_array.push_back(make_float4(pos[0], pos[2], pos[1], 0.01f));
        } else {
          pos_array.push_back(make_float4(pos[0], pos[1], pos[2], 0.01f));
        }
      }
    }

    KIRI_LOG_INFO("Loaded Bgeo File={0}, Number of Particles={1}", Names[n],
                  data->numParticles());

    data->release();
  }

  return pos_array;
}

void KiriCudaUtils::ExportBgeoFileFromGPU(String Folder, String FileName,
                                          float3 *Positions, float *Radius,
                                          UInt numOfParticles) {
  String exportPath = String(EXPORT_PATH) + "bgeo/" + Folder;
  String exportFile = exportPath + "/" + FileName + ".bgeo";

  try {
    std::error_code ec;
    std::filesystem::create_directories(exportPath, ec);

    Partio::ParticlesDataMutable *p = Partio::create();
    Partio::ParticleAttribute positionAttr =
        p->addAttribute("position", Partio::VECTOR, 3);

    Partio::ParticleAttribute pScaleAttr =
        p->addAttribute("pscale", Partio::FLOAT, 1);

    // transfer GPU data to CPU
    uint fBytes = numOfParticles * sizeof(float);
    uint f3Bytes = numOfParticles * sizeof(float3);
    uint uintBytes = numOfParticles * sizeof(uint);

    float *cpuRadius = (float *)malloc(fBytes);
    float3 *cpuPositions = (float3 *)malloc(f3Bytes);

    cudaMemcpy(cpuRadius, Radius, fBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpuPositions, Positions, f3Bytes, cudaMemcpyDeviceToHost);

    for (UInt i = 0; i < numOfParticles; i++) {
      Int particle = p->addParticle();
      float *pos = p->dataWrite<float>(positionAttr, particle);
      float *pscale = p->dataWrite<float>(pScaleAttr, particle);

      pos[0] = cpuPositions[i].x;
      pos[1] = cpuPositions[i].y;
      pos[2] = cpuPositions[i].z;

      *pscale = cpuRadius[i];
    }

    Partio::write(exportFile.c_str(), *p);
    p->release();

    free(cpuPositions);

    KIRI_LOG_INFO("Successfully Saved Bgeo File:{0}", exportFile);
  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
  }
}

void KiriCudaUtils::ExportBgeoFileFromGPU(String Folder, String FileName,
                                          float4 *Positions, float4 *Colors,
                                          uint *Labels, UInt numOfParticles) {
  String exportPath = String(EXPORT_PATH) + "bgeo/" + Folder;
  String exportFile = exportPath + "/" + FileName + ".bgeo";

  try {
    std::error_code ec;
    std::filesystem::create_directories(exportPath, ec);

    Partio::ParticlesDataMutable *p = Partio::create();
    Partio::ParticleAttribute positionAttr =
        p->addAttribute("position", Partio::VECTOR, 3);
    Partio::ParticleAttribute colorAttr =
        p->addAttribute("Cd", Partio::FLOAT, 3);
    Partio::ParticleAttribute pScaleAttr =
        p->addAttribute("pscale", Partio::FLOAT, 1);
    Partio::ParticleAttribute labelAttr =
        p->addAttribute("label", Partio::INT, 1);

    // transfer GPU data to CPU
    uint f4Bytes = numOfParticles * sizeof(float4);
    uint uintBytes = numOfParticles * sizeof(uint);

    float4 *cpuPositions = (float4 *)malloc(f4Bytes);
    float4 *cpuColors = (float4 *)malloc(f4Bytes);
    uint *cpuLabels = (uint *)malloc(uintBytes);

    cudaMemcpy(cpuPositions, Positions, f4Bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpuColors, Colors, f4Bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpuLabels, Labels, uintBytes, cudaMemcpyDeviceToHost);

    for (UInt i = 0; i < numOfParticles; i++) {
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

    Partio::write(exportFile.c_str(), *p);
    p->release();

    free(cpuPositions);
    free(cpuColors);
    free(cpuLabels);

    KIRI_LOG_INFO("Successfully Saved Bgeo File:{0}", exportFile);
  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
  }
}

void KiriCudaUtils::ExportBgeoFileCUDA(String FolderPath, String FileName,
                                       float3 *Positions, float3 *Colors,
                                       float *Radius, size_t *Labels,
                                       UInt numOfParticles) {
  String exportFile = FolderPath + "/" + FileName + ".bgeo";

  try {

    struct stat info;

    if (stat(FolderPath.c_str(), &info) != 0) {
      std::error_code ec;
      bool success = std::filesystem::create_directories(FolderPath, ec);
      if (!success) {
        std::cout << ec.message() << std::endl;
      }
    }

    Partio::ParticlesDataMutable *p = Partio::create();
    Partio::ParticleAttribute positionAttr =
        p->addAttribute("position", Partio::VECTOR, 3);
    Partio::ParticleAttribute colorAttr =
        p->addAttribute("Cd", Partio::FLOAT, 3);
    Partio::ParticleAttribute pScaleAttr =
        p->addAttribute("pscale", Partio::FLOAT, 1);
    Partio::ParticleAttribute labelAttr =
        p->addAttribute("label", Partio::INT, 1);

    // transfer GPU data to CPU
    size_t fBytes = numOfParticles * sizeof(float);
    size_t f3Bytes = numOfParticles * sizeof(float3);
    size_t uintBytes = numOfParticles * sizeof(size_t);

    float3 *cpuPositions = (float3 *)malloc(f3Bytes);
    float3 *cpuColors = (float3 *)malloc(f3Bytes);
    float *cpuRadius = (float *)malloc(fBytes);
    size_t *cpuLabels = (size_t *)malloc(uintBytes);

    cudaMemcpy(cpuPositions, Positions, f3Bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpuColors, Colors, f3Bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpuRadius, Radius, fBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpuLabels, Labels, uintBytes, cudaMemcpyDeviceToHost);

    for (UInt i = 0; i < numOfParticles; i++) {
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
      *pscale = cpuRadius[i];

      *label = cpuLabels[i];
    }
    Partio::write(exportFile.c_str(), *p);

    p->release();

    free(cpuPositions);
    free(cpuColors);
    free(cpuLabels);
    free(cpuRadius);

    KIRI_LOG_INFO("Successfully Saved Bgeo File:{0}", exportFile);
  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
  }
}

void KiriCudaUtils::ExportBgeoFileFromGPU(String Folder, String FileName,
                                          float3 *Positions, float3 *Colors,
                                          float *Radius, size_t *Labels,
                                          UInt numOfParticles) {
  String exportPath = String(EXPORT_PATH) + "bgeo/" + Folder;
  String exportFile = exportPath + "/" + FileName + ".bgeo";

  try {
    std::error_code ec;
    std::filesystem::create_directories(exportPath, ec);

    Partio::ParticlesDataMutable *p = Partio::create();
    Partio::ParticleAttribute positionAttr =
        p->addAttribute("position", Partio::VECTOR, 3);
    Partio::ParticleAttribute colorAttr =
        p->addAttribute("Cd", Partio::FLOAT, 3);
    Partio::ParticleAttribute pScaleAttr =
        p->addAttribute("pscale", Partio::FLOAT, 1);
    Partio::ParticleAttribute labelAttr =
        p->addAttribute("label", Partio::INT, 1);

    // transfer GPU data to CPU
    size_t fBytes = numOfParticles * sizeof(float);
    size_t f3Bytes = numOfParticles * sizeof(float3);
    size_t uintBytes = numOfParticles * sizeof(size_t);

    float3 *cpuPositions = (float3 *)malloc(f3Bytes);
    float3 *cpuColors = (float3 *)malloc(f3Bytes);
    float *cpuRadius = (float *)malloc(fBytes);
    size_t *cpuLabels = (size_t *)malloc(uintBytes);

    cudaMemcpy(cpuPositions, Positions, f3Bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpuColors, Colors, f3Bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpuRadius, Radius, fBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpuLabels, Labels, uintBytes, cudaMemcpyDeviceToHost);

    for (UInt i = 0; i < numOfParticles; i++) {
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
      *pscale = cpuRadius[i];

      *label = cpuLabels[i];
    }
    Partio::write(exportFile.c_str(), *p);

    p->release();

    free(cpuPositions);
    free(cpuColors);
    free(cpuLabels);
    free(cpuRadius);

    KIRI_LOG_INFO("Successfully Saved Bgeo File:{0}", exportFile);
  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
  }
}
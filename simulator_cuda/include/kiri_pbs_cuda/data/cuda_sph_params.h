/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-08-11 11:44:38
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-22 16:07:39
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\data\cuda_sph_params.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _CUDA_SPH_PARAMS_H_
#define _CUDA_SPH_PARAMS_H_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI
{
  enum CudaSphSolverType
  {
    SPH_SOLVER,
    WCSPH_SOLVER,
    IISPH_SOLVER,
    DFSPH_SOLVER,
    PBF_SOLVER,
    MSSPH_SOLVER
  };

  struct CudaSphParams
  {
    float rest_mass;
    float rest_density;
    float particle_radius;
    float kernel_radius;
    int3 grid_size;

    bool atf_visc;
    float stiff;
    float visc;
    float nu;
    float bnu;

    bool sta_akinci13;
    float st_gamma;
    float a_beta;

    float3 gravity;
    float dt;

    CudaSphSolverType solver_type;
  };

  enum CudaSphEmitterType
  {
    SQUARE,
    CIRCLE,
    RECTANGLE
  };

  struct CudaSphEmitterParams
  {
    bool enable = false;
    bool run = false;

    float3 emit_pos;
    float3 emit_vel;
    float3 emit_col;

    CudaSphEmitterType emit_type = CudaSphEmitterType::SQUARE;

    float emit_radius = 0.0;
    float emit_width = 0.0;
    float emit_height = 0.0;
  };

  struct CudaSphAppParams
  {
    size_t max_num = 100000;
    bool run = false;
    bool run_offline = false;

    bool move_boundary = false;
    int current_frame = 0;
    int move_boundary_frame = 0;

    int scene_data_idx = 0;
    char bgeo_file_name[32] = "default";
    bool bgeo_export = false;
  };

  extern CudaSphParams CUDA_SPH_PARAMS;
  extern CudaSphEmitterParams CUDA_SPH_EMITTER_PARAMS;
  extern CudaSphAppParams CUDA_SPH_APP_PARAMS;

} // namespace KIRI

#endif
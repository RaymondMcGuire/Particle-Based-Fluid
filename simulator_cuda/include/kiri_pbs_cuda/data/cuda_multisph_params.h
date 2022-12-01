/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2021-03-02 20:30:53
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-22 17:53:02
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\data\cuda_multisph_params.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _CUDA_MULTISPH_PARAMS_H_
#define _CUDA_MULTISPH_PARAMS_H_

#pragma once

#include <kiri_pbs_cuda/data/cuda_sph_params.h>

namespace KIRI
{
    struct Ren14PhaseDataBlock1
    {
        float rest_mix_density;
        float rest_mix_pressure;
        float rest_mix_viscosity;

        float phase_pressure[MULTISPH_MAX_PHASE_NUM];
        float3 gradient_pressures[MULTISPH_MAX_PHASE_NUM];
    };

    struct Ren14PhaseDataBlock2
    {
        float volume_fractions[MULTISPH_MAX_PHASE_NUM];
        float last_volume_fractions[MULTISPH_MAX_PHASE_NUM];
        float3 gradient_volume_fractions[MULTISPH_MAX_PHASE_NUM];
        float delta_volume_fractions[MULTISPH_MAX_PHASE_NUM];
        float3 drift_velocities[MULTISPH_MAX_PHASE_NUM];
    };

    struct Yan16PhaseData
    {
        size_t phase_type;
        size_t last_phase_type;
        size_t phase_types[MULTISPH_MAX_PHASE_NUM];

        float rest_mix_density;
        float rest_mix_pressure;
        float rest_mix_viscosity;

        float phase_pressure[MULTISPH_MAX_PHASE_NUM];
        float3 gradient_pressures[MULTISPH_MAX_PHASE_NUM];

        float volume_fractions[MULTISPH_MAX_PHASE_NUM];
        float last_volume_fractions[MULTISPH_MAX_PHASE_NUM];

        float3 gradient_volume_fractions[MULTISPH_MAX_PHASE_NUM];
        float delta_volume_fractions[MULTISPH_MAX_PHASE_NUM];

        float3 drift_velocities[MULTISPH_MAX_PHASE_NUM];

        // solid phase
        tensor3x3 stress_tensor[MULTISPH_MAX_PHASE_NUM];
        tensor3x3 stress_rate_tensor[MULTISPH_MAX_PHASE_NUM];
        tensor3x3 deviatoric_stress_tensor[MULTISPH_MAX_PHASE_NUM];
        tensor3x3 deviatoric_stress_rate_tensor[MULTISPH_MAX_PHASE_NUM];
    };

    struct CudaMultiSphRen14Params
    {
        size_t phase_num;

        float rho0[MULTISPH_MAX_PHASE_NUM];
        float mass0[MULTISPH_MAX_PHASE_NUM];
        float3 color0[MULTISPH_MAX_PHASE_NUM];
        float visc0[MULTISPH_MAX_PHASE_NUM];

        float particle_radius;
        float kernel_radius;

        float stiff;
        float sound_speed;
        float bnu;

        bool miscible;
        float tou;
        float sigma;

        float3 gravity;
        float dt;

        CudaSphSolverType solver_type;
    };

    struct Yang15PhaseDataBlock1
    {
        float aggregate_density;

        float mass_ratios[MULTISPH_MAX_PHASE_NUM];
        float last_mass_ratio[MULTISPH_MAX_PHASE_NUM];
        float delta_mass_ratio[MULTISPH_MAX_PHASE_NUM];
    };

    struct Yang15PhaseDataBlock2
    {
        float3 mass_ratio_gradient[MULTISPH_MAX_PHASE_NUM];
        float mass_ratio_laplacian[MULTISPH_MAX_PHASE_NUM];
        float chemical_potential[MULTISPH_MAX_PHASE_NUM];
    };

    struct CudaMultiSphYang15Params
    {
        size_t phase_num;

        float rho0[MULTISPH_MAX_PHASE_NUM];
        float mass0[MULTISPH_MAX_PHASE_NUM];
        float3 color0[MULTISPH_MAX_PHASE_NUM];

        float particle_radius;
        float kernel_radius;

        float visc;
        float boundary_visc;

        float sigma;
        float eta;
        float mobilities;
        float alpha;
        float s1;
        float s2;
        float epsilon;

        float3 gravity;
        float dt;

        CudaSphSolverType solver_type;
    };

    struct CudaMultiSphYan16Params
    {
        size_t phase_num;

        size_t phase0[MULTISPH_MAX_PHASE_NUM];
        float rho0[MULTISPH_MAX_PHASE_NUM];
        float mass0[MULTISPH_MAX_PHASE_NUM];
        float3 color0[MULTISPH_MAX_PHASE_NUM];
        float visc0[MULTISPH_MAX_PHASE_NUM];

        float particle_radius;
        float kernel_radius;

        float stiff;
        float sound_speed;
        float bnu;

        // fluid
        bool miscible;
        float tou;
        float sigma;

        // solid
        float Y;
        float G;

        float3 gravity;
        float dt;

        CudaSphSolverType solver_type;
    };

    struct CudaMultiSphAppParams
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

    extern CudaMultiSphRen14Params CUDA_MULTISPH_REN14_PARAMS;
    extern CudaMultiSphYang15Params CUDA_MULTISPH_YANG15_PARAMS;
    extern CudaMultiSphYan16Params CUDA_MULTISPH_YAN16_PARAMS;
    extern CudaMultiSphAppParams CUDA_MULTISPH_APP_PARAMS;

} // namespace KIRI

#endif
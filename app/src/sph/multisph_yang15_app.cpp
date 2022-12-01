/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:27
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 11:58:49
 * @FilePath: \Kiri\app\src\sph\multisph_yang15_app.cpp
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
// clang-format off
#include <imgui/include/imgui.h>
#include <sph/multisph_yang15_app.h>

#include <kiri_pbs_cuda/solver/multisph/cuda_multisph_yang15_solver.cuh>

#include <fbs/generated/cuda_multisph_app_generated.h>
#include <fbs/fbs_helper.h>

#include <kiri_pbs_cuda/emitter/cuda_volume_emitter.cuh>
// clang-format on

namespace KIRI
{
  void KiriMultiSphYang15App::SetupPBSParams()
  {
    KIRI_LOG_DEBUG("MULTISPH APP:SetupPBSParams");

    auto scene_config_data =
        KIRI::FlatBuffers::GetCudaMultiSphApp(mSceneConfigData.data());

    // max number of particles
    CUDA_MULTISPH_APP_PARAMS.max_num = scene_config_data->max_particles_num();

    // multisph data
    auto multisph_data = scene_config_data->multisph_data();
    CUDA_MULTISPH_YANG15_PARAMS.phase_num = multisph_data->phase_num();

    for (size_t i = 0; i < CUDA_MULTISPH_YANG15_PARAMS.phase_num; i++)
    {
      CUDA_MULTISPH_YANG15_PARAMS.rho0[i] = multisph_data->rho0()->Get(i);
      CUDA_MULTISPH_YANG15_PARAMS.mass0[i] = multisph_data->mass0()->Get(i);
      CUDA_MULTISPH_YANG15_PARAMS.color0[i] = FbsToKiriCUDA(
          *(multisph_data->color0()->GetAs<FlatBuffers::float3>(i)));

      KIRI_LOG_INFO("MultiSph Phase {0}: rho={1},mass={2},color=({3},{4},{5})", i,
                    CUDA_MULTISPH_YANG15_PARAMS.rho0[i],
                    CUDA_MULTISPH_YANG15_PARAMS.mass0[i],
                    CUDA_MULTISPH_YANG15_PARAMS.color0[i].x,
                    CUDA_MULTISPH_YANG15_PARAMS.color0[i].y,
                    CUDA_MULTISPH_YANG15_PARAMS.color0[i].z);
    }

    // sph
    CUDA_MULTISPH_YANG15_PARAMS.kernel_radius = multisph_data->kernel_radius();
    CUDA_MULTISPH_YANG15_PARAMS.particle_radius =
        multisph_data->particle_radius();

    CUDA_MULTISPH_YANG15_PARAMS.visc = 0.05f;
    CUDA_MULTISPH_YANG15_PARAMS.boundary_visc = 0.f;
    CUDA_MULTISPH_YANG15_PARAMS.sigma = 0.5f;
    CUDA_MULTISPH_YANG15_PARAMS.eta =
        0.1f * CUDA_MULTISPH_YANG15_PARAMS.kernel_radius;
    CUDA_MULTISPH_YANG15_PARAMS.mobilities = 0.25f;
    CUDA_MULTISPH_YANG15_PARAMS.alpha = 0.001f;
    CUDA_MULTISPH_YANG15_PARAMS.s1 = 0.4f;
    CUDA_MULTISPH_YANG15_PARAMS.s2 = 0.4f;
    CUDA_MULTISPH_YANG15_PARAMS.epsilon = 0.0001f;

    CUDA_MULTISPH_YANG15_PARAMS.gravity =
        FbsToKiriCUDA(*multisph_data->gravity());
    CUDA_MULTISPH_YANG15_PARAMS.dt = multisph_data->fixed_dt();

    // scene data
    auto app_data = scene_config_data->app_data();
    auto scene_data = app_data->scene();
    CUDA_BOUNDARY_PARAMS.lowest_point = FbsToKiriCUDA(*scene_data->world_lower());
    CUDA_BOUNDARY_PARAMS.highest_point =
        FbsToKiriCUDA(*scene_data->world_upper());
    CUDA_BOUNDARY_PARAMS.world_size = FbsToKiriCUDA(*scene_data->world_size());
    CUDA_BOUNDARY_PARAMS.world_center =
        FbsToKiriCUDA(*scene_data->world_center());
    CUDA_BOUNDARY_PARAMS.kernel_radius = multisph_data->kernel_radius();
    CUDA_BOUNDARY_PARAMS.grid_size = make_int3(
        (CUDA_BOUNDARY_PARAMS.highest_point - CUDA_BOUNDARY_PARAMS.lowest_point) /
        CUDA_BOUNDARY_PARAMS.kernel_radius);

    mInitLowestPoint = CUDA_BOUNDARY_PARAMS.lowest_point;
    mInitHighestPoint = CUDA_BOUNDARY_PARAMS.highest_point;

    // bgeo file export & render FPS
    CUDA_MULTISPH_APP_PARAMS.bgeo_export = app_data->bgeo_export_mode_enable();
    if (app_data->render_mode_enable())
      SetRenderFps(app_data->render_mode_fps());
    else
      SetRenderFps(1.f / CUDA_MULTISPH_YANG15_PARAMS.dt);

    // camera data
    auto camera_data = scene_data->camera();
    mCamera->SetYawPitchPos(camera_data->yaw(), camera_data->pitch(),
                            FbsToKiri(*camera_data->position()));

    // boundary sampling
    BoundaryData boundary_data;
    auto boundary_emitter = std::make_shared<CudaBoundaryEmitter>();

    boundary_emitter->buildWorldBoundary(
        boundary_data, CUDA_BOUNDARY_PARAMS.lowest_point,
        CUDA_BOUNDARY_PARAMS.highest_point,
        CUDA_MULTISPH_YANG15_PARAMS.particle_radius);

    auto boundary_particles = std::make_shared<CudaBoundaryParticles>(
        boundary_data.pos.size() * 4, boundary_data.pos, boundary_data.label);

    // sampling SPH init volume particles
    auto volume_emitter = std::make_shared<CudaVolumeEmitter>(true);
    MultiSphRen14VolumeData volume_data;

    // init volume data
    auto init_volume = scene_config_data->init_volume();
    for (size_t i = 0; i < CUDA_MULTISPH_YANG15_PARAMS.phase_num; i++)
    {
      auto init_volume_box_size = FbsToKiriCUDA(*(
          init_volume->GetAs<FlatBuffers::MultiSphInitBoxVolume>(i)->box_size()));
      auto init_volume_box_lower = FbsToKiriCUDA(
          *(init_volume->GetAs<FlatBuffers::MultiSphInitBoxVolume>(i)
                ->box_lower()));

      KIRI_LOG_INFO("Init MultiSph Volume {0}: box_size=({1},{2},{3}), "
                    "box_lower=({4},{5},{6})",
                    i, init_volume_box_size.x, init_volume_box_size.y,
                    init_volume_box_size.z, init_volume_box_lower.x,
                    init_volume_box_lower.y, init_volume_box_lower.z);

      volume_emitter->buildMultiSphRen14Volume(
          volume_data, init_volume_box_lower, init_volume_box_size,
          CUDA_MULTISPH_YANG15_PARAMS.particle_radius,
          CUDA_MULTISPH_YANG15_PARAMS.color0[i],
          CUDA_MULTISPH_YANG15_PARAMS.mass0[i], i);
    }
    KIRI_LOG_INFO("Number of fluid particles = {0}", volume_data.pos.size());

    // emitter
    CudaEmitterPtr emitter = std::make_shared<CudaEmitter>();

    auto fluid_particles = std::make_shared<CudaMultiSphYang15Particles>(
        CUDA_MULTISPH_APP_PARAMS.max_num, volume_data.pos, volume_data.col,
        volume_data.mass, volume_data.phaseLabel,
        CUDA_MULTISPH_YANG15_PARAMS.rho0, CUDA_MULTISPH_YANG15_PARAMS.mass0,
        CUDA_MULTISPH_YANG15_PARAMS.color0);

    auto sph_solver_type = scene_config_data->sph_solver_type();
    bool adaptive_sub_timestep = false;
    CudaMultiSphYang15SolverPtr solver;
    CUDA_MULTISPH_YANG15_PARAMS.solver_type = SPH_SOLVER;
    switch (sph_solver_type)
    {
    case FlatBuffers::CudaSphType::CudaSphType_SPH:
      solver =
          std::make_shared<CudaMultiSphYang15Solver>(fluid_particles->maxSize());
      break;
    default:
      solver =
          std::make_shared<CudaMultiSphYang15Solver>(fluid_particles->maxSize());
      break;
    }

    CudaGNSearcherPtr searcher = std::make_shared<CudaGNSearcher>(
        CUDA_BOUNDARY_PARAMS.lowest_point, CUDA_BOUNDARY_PARAMS.highest_point,
        fluid_particles->maxSize(), CUDA_BOUNDARY_PARAMS.kernel_radius,
        SearcherParticleType::MULTISPH_YANG15);

    CudaGNBoundarySearcherPtr boundary_searcher =
        std::make_shared<CudaGNBoundarySearcher>(
            CUDA_BOUNDARY_PARAMS.lowest_point, CUDA_BOUNDARY_PARAMS.highest_point,
            boundary_particles->maxSize(), CUDA_BOUNDARY_PARAMS.kernel_radius);

    mSystem = std::make_shared<CudaMultiSphYang15System>(
        fluid_particles, boundary_particles, solver, searcher, boundary_searcher,
        emitter, adaptive_sub_timestep);

    mSystem->updateSystemForVBO(mRenderInterval);

    // ssf data
    auto ssf_data = scene_config_data->renderer_data();
    mFluidRenderSystem->EnableFluidTransparentMode(
        ssf_data->fluid_transparent_mode());
    mFluidRenderSystem->EnableSoildSsfMode(ssf_data->soild_particle_mode());

    // render particles
    SetParticleVBOWithRadius(mSystem->positionsVBO(), mSystem->colorsVBO(),
                             mSystem->numOfParticles());
  }

  void KiriMultiSphYang15App::OnPBSUpdate(const KIRI::KiriTimeStep &DeltaTime)
  {

    if (CUDA_MULTISPH_APP_PARAMS.move_boundary)
    {
      auto time = 0.1f * (CUDA_MULTISPH_APP_PARAMS.current_frame -
                          CUDA_MULTISPH_APP_PARAMS.move_boundary_frame);
      auto phi = sin(time);
      auto move_range = make_float3(0.f, 0.f, 0.5f);
      auto move_interval = move_range * phi;
      // move_range.z * phi > 0.f ? move_range * phi : make_float3(0.f);

      mSystem->moveBoundary(mInitLowestPoint, mInitHighestPoint + move_interval);

      mBoundaryModel->ResetBox(Vector3F(CUDA_BOUNDARY_PARAMS.world_center.x,
                                        CUDA_BOUNDARY_PARAMS.world_center.y,
                                        CUDA_BOUNDARY_PARAMS.world_center.z),
                               Vector3F(CUDA_BOUNDARY_PARAMS.world_size.x,
                                        CUDA_BOUNDARY_PARAMS.world_size.y,
                                        CUDA_BOUNDARY_PARAMS.world_size.z));
      mBoundaryEnity->SetModelMatrix(0, mBoundaryModel->GetModelMatrix());
    }

    if (CUDA_MULTISPH_APP_PARAMS.run)
    {
      mSystem->updateSystemForVBO(mRenderInterval);
      SetParticleVBOWithRadius(mSystem->positionsVBO(), mSystem->colorsVBO(),
                               mSystem->numOfParticles());
    }
    else
    {
      CUDA_MULTISPH_APP_PARAMS.move_boundary = false;
    }

    CUDA_MULTISPH_APP_PARAMS.current_frame++;
  }

  void KiriMultiSphYang15App::SetupPBSScene()
  {
    KIRI_LOG_DEBUG("SPH APP:SetupPBSScene");

    float3 world_center = CUDA_BOUNDARY_PARAMS.world_center;
    float3 world_size = CUDA_BOUNDARY_PARAMS.world_size;

    // pre-computed map
    UInt irradiance_cubemap = mScene->GetCubeSkybox()->GetIrradianceCubeMap();
    UInt spec_cubemap = mScene->GetCubeSkybox()->GetSpecularCubeMap();
    UInt brdf_lut_texure = mScene->GetCubeSkybox()->GetBrdfLutTexture();

    uint id = 0;
    // floor
    KiriPlanePtr plane = std::make_shared<KiriPlane>(
        100.0f, CUDA_BOUNDARY_PARAMS.lowest_point.y, Vector3F(0.0f, 1.0f, 0.0f));
    KiriPBRTexturePtr roomtile = std::make_shared<KiriPBRTexture>("tile");
    roomtile->Load();

    // floor material
    KiriMaterialPBRIBLTexPtr m_roomtile = std::make_shared<KiriMaterialPBRIBLTex>(
        irradiance_cubemap, spec_cubemap, brdf_lut_texure, roomtile);
    m_roomtile->SetPointLights(mScene->GetPointLights());
    plane->ResetModelMatrix();
    KiriEntityPtr entity_roomtile =
        std::make_shared<KiriEntity>(id++, plane, m_roomtile);

    // wall(no material)
    mBoundaryModel = std::make_shared<KiriBox>(
        Vector3F(world_center.x, world_center.y, world_center.z),
        Vector3F(world_size.x, world_size.y, world_size.z));
    mBoundaryModel->SetWireFrame(true);
    KiriMaterialLampPtr m_debug = std::make_shared<KiriMaterialLamp>();
    mBoundaryEnity = std::make_shared<KiriEntity>(id++, mBoundaryModel, m_debug);

    mScene->Add(entity_roomtile);
    mScene->Add(mBoundaryEnity);
  }

  void KiriMultiSphYang15App::OnImguiRender()
  {
    static bool p_open = true;
    if (p_open)
    {
      ImGui::SetNextWindowSize(ImVec2(430, 450), ImGuiCond_FirstUseEver);
      if (ImGui::Begin("SPH Examples", &p_open))
      {
        if (ImGui::CollapsingHeader("Fluid Renderer Parameters"))
        {
          ImGui::Checkbox("Enable Particle View", &SSF_DEMO_PARAMS.particleView);
          const char *items[] = {"depth", "thick", "normal", "color", "fluid"};
          ImGui::Combo("Render Mode", &SSF_DEMO_PARAMS.renderOpt, items,
                       IM_ARRAYSIZE(items));
        }

        if (ImGui::CollapsingHeader("Example Scene"))
        {
          const char *items[] = {"multisph_yang2015_two_phase"};

          ImGui::Combo("Scene Config Data File",
                       &CUDA_MULTISPH_APP_PARAMS.scene_data_idx, items,
                       IM_ARRAYSIZE(items));
          ChangeSceneConfigData(items[CUDA_MULTISPH_APP_PARAMS.scene_data_idx]);
        }

        if (ImGui::CollapsingHeader("Simulation"))
        {

          if (ImGui::Button("Move Boundary"))
          {
            CUDA_MULTISPH_APP_PARAMS.move_boundary =
                !CUDA_MULTISPH_APP_PARAMS.move_boundary;
            CUDA_MULTISPH_APP_PARAMS.move_boundary_frame =
                CUDA_MULTISPH_APP_PARAMS.current_frame;
          }

          if (CUDA_SPH_EMITTER_PARAMS.enable)
            ImGui::Checkbox("Emit Particles", &CUDA_SPH_EMITTER_PARAMS.run);

          ImGui::Checkbox("Run", &CUDA_MULTISPH_APP_PARAMS.run);
        }
        ImGui::End();
      }
    }
  }
} // namespace KIRI
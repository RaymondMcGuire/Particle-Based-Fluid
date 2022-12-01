/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:27
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 12:06:51
 * @FilePath: \Kiri\app\src\sph\sph_app.cpp
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
// clang-format off
#include <imgui/include/imgui.h>
#include <sph/sph_app.h>

#include <kiri_pbs_cuda/particle/cuda_iisph_particles.cuh>
#include <kiri_pbs_cuda/solver/sph/cuda_iisph_solver.cuh>
#include <kiri_pbs_cuda/particle/cuda_dfsph_particles.cuh>
#include <kiri_pbs_cuda/solver/sph/cuda_dfsph_solver.cuh>
#include <kiri_pbs_cuda/solver/sph/cuda_multi_sph_solver.cuh>
#include <kiri_pbs_cuda/solver/sph/cuda_wcsph_solver.cuh>

#include <fbs/generated/cuda_sph_app_generated.h>
#include <fbs/fbs_helper.h>

#include <kiri_pbs_cuda/emitter/cuda_volume_emitter.cuh>
// clang-format on
namespace KIRI
{
  void KiriSphApp::SetupPBSParams()
  {
    KIRI_LOG_DEBUG("SPH APP:SetupPBSParams");

    auto scene_config_data =
        KIRI::FlatBuffers::GetCudaSphApp(mSceneConfigData.data());

    // max number of particles
    CUDA_SPH_APP_PARAMS.max_num = scene_config_data->max_particles_num();

    // sph data
    auto sph_data = scene_config_data->sph_data();
    CUDA_SPH_PARAMS.rest_density = sph_data->rest_density();
    CUDA_SPH_PARAMS.rest_mass = sph_data->rest_mass();
    CUDA_SPH_PARAMS.kernel_radius = sph_data->kernel_radius();
    CUDA_SPH_PARAMS.particle_radius = sph_data->particle_radius();

    CUDA_SPH_PARAMS.stiff = sph_data->stiff();
    CUDA_SPH_PARAMS.gravity = FbsToKiriCUDA(*sph_data->gravity());

    CUDA_SPH_PARAMS.atf_visc = sph_data->enable_atf_visc();
    CUDA_SPH_PARAMS.visc = sph_data->visc();
    CUDA_SPH_PARAMS.nu = sph_data->nu();
    CUDA_SPH_PARAMS.bnu = sph_data->bnu();

    CUDA_SPH_PARAMS.sta_akinci13 = sph_data->enable_sta_akinci13();
    CUDA_SPH_PARAMS.st_gamma = sph_data->st_gamma();
    CUDA_SPH_PARAMS.a_beta = sph_data->a_beta();

    CUDA_SPH_PARAMS.dt = sph_data->fixed_dt();

    KIRI_LOG_INFO("SPH DATA: rho={0}, mass={1}, stiff={2}, nu={3}, dt={4}",
                  CUDA_SPH_PARAMS.rest_density, CUDA_SPH_PARAMS.rest_mass,
                  CUDA_SPH_PARAMS.stiff, CUDA_SPH_PARAMS.nu, CUDA_SPH_PARAMS.dt);

    // init volume data
    auto init_volume = scene_config_data->init_volume();
    auto init_volume_box_size = FbsToKiriCUDA(*init_volume->box_size());
    auto init_volume_box_lower = FbsToKiriCUDA(*init_volume->box_lower());
    auto init_volume_box_color = FbsToKiriCUDA(*init_volume->box_color());

    // sph emitter
    auto sph_emitter = scene_config_data->sph_emitter();
    CUDA_SPH_EMITTER_PARAMS.enable = sph_emitter->enable();
    CUDA_SPH_EMITTER_PARAMS.run = false;
    CUDA_SPH_EMITTER_PARAMS.emit_pos = FbsToKiriCUDA(*sph_emitter->emit_pos());
    CUDA_SPH_EMITTER_PARAMS.emit_vel = FbsToKiriCUDA(*sph_emitter->emit_vel());
    CUDA_SPH_EMITTER_PARAMS.emit_col = init_volume_box_color;

    CUDA_SPH_EMITTER_PARAMS.emit_radius = sph_emitter->emit_radius();
    CUDA_SPH_EMITTER_PARAMS.emit_width = sph_emitter->emit_width();
    CUDA_SPH_EMITTER_PARAMS.emit_height = sph_emitter->emit_height();
    switch (sph_emitter->emit_type())
    {
    case FlatBuffers::CudaSphEmitterType::CudaSphEmitterType_SQUARE:
      CUDA_SPH_EMITTER_PARAMS.emit_type = CudaSphEmitterType::SQUARE;
      break;
    case FlatBuffers::CudaSphEmitterType::CudaSphEmitterType_CIRCLE:
      CUDA_SPH_EMITTER_PARAMS.emit_type = CudaSphEmitterType::CIRCLE;
      break;
    case FlatBuffers::CudaSphEmitterType::CudaSphEmitterType_RECTANGLE:
      CUDA_SPH_EMITTER_PARAMS.emit_type = CudaSphEmitterType::RECTANGLE;
      break;
    }

    CudaEmitterPtr emitter = std::make_shared<CudaEmitter>(
        CUDA_SPH_EMITTER_PARAMS.emit_pos, CUDA_SPH_EMITTER_PARAMS.emit_vel,
        CUDA_SPH_EMITTER_PARAMS.enable);

    // scene data
    auto app_data = scene_config_data->app_data();
    auto scene_data = app_data->scene();
    CUDA_BOUNDARY_PARAMS.lowest_point = FbsToKiriCUDA(*scene_data->world_lower());
    CUDA_BOUNDARY_PARAMS.highest_point =
        FbsToKiriCUDA(*scene_data->world_upper());
    CUDA_BOUNDARY_PARAMS.world_size = FbsToKiriCUDA(*scene_data->world_size());
    CUDA_BOUNDARY_PARAMS.world_center =
        FbsToKiriCUDA(*scene_data->world_center());
    CUDA_BOUNDARY_PARAMS.kernel_radius = sph_data->kernel_radius();
    CUDA_BOUNDARY_PARAMS.grid_size = make_int3(
        (CUDA_BOUNDARY_PARAMS.highest_point - CUDA_BOUNDARY_PARAMS.lowest_point) /
        CUDA_BOUNDARY_PARAMS.kernel_radius);

    mInitLowestPoint = CUDA_BOUNDARY_PARAMS.lowest_point;
    mInitHighestPoint = CUDA_BOUNDARY_PARAMS.highest_point;

    // bgeo file export & render FPS
    CUDA_SPH_APP_PARAMS.bgeo_export = app_data->bgeo_export_mode_enable();
    if (app_data->render_mode_enable())
      SetRenderFps(app_data->render_mode_fps());
    else
      SetRenderFps(1.f / CUDA_SPH_PARAMS.dt);

    // camera data
    auto camera_data = scene_data->camera();
    mCamera->SetYawPitchPos(camera_data->yaw(), camera_data->pitch(),
                            FbsToKiri(*camera_data->position()));

    // boundary sampling
    BoundaryData boundary_data;
    auto boundary_emitter = std::make_shared<CudaBoundaryEmitter>();

    boundary_emitter->buildWorldBoundary(
        boundary_data, CUDA_BOUNDARY_PARAMS.lowest_point,
        CUDA_BOUNDARY_PARAMS.highest_point, CUDA_SPH_PARAMS.particle_radius);

    auto boundary_particles = std::make_shared<CudaBoundaryParticles>(
        boundary_data.pos.size() * 4, boundary_data.pos, boundary_data.label);
    KIRI_LOG_INFO("Number of boundary particles = {0}",
                  boundary_particles->size());

    auto sph_solver_type = scene_config_data->sph_solver_type();
    bool adaptive_sub_timestep = false;

    // sampling SPH init volume particles
    auto volume_emitter = std::make_shared<CudaVolumeEmitter>(true);
    SphVolumeData volume_data;

    if (sph_solver_type == FlatBuffers::CudaSphType::CudaSphType_MSSPH)
    {
      auto [pos_data, mass_data] = KiriCudaUtils::ReadBgeoFileWithMassForGPU(
          "armadillo", "multi_armadillo");
      volume_emitter->buildSphShapeVolume(volume_data, pos_data, mass_data,
                                          init_volume_box_color);
    }
    else
    {

      if (mName.find("bunny") != std::string::npos)
      {
        auto [pos_data, mass_data] = KiriCudaUtils::ReadBgeoFileWithMassForGPU(
            "bunny", "bunny_sph");
        volume_emitter->buildSphShapeVolume(volume_data, pos_data, mass_data,
                                            init_volume_box_color);
      }
      else
        volume_emitter->buildSphVolume(
            volume_data, init_volume_box_lower, init_volume_box_size,
            CUDA_SPH_PARAMS.particle_radius, CUDA_SPH_PARAMS.rest_mass,
            init_volume_box_color);
    }
    KIRI_LOG_INFO("Number of fluid particles = {0}", volume_data.pos.size());

    CudaSphSolverPtr solver;
    CudaSphParticlesPtr fluid_particles;

    CUDA_SPH_PARAMS.solver_type = SPH_SOLVER;
    switch (sph_solver_type)
    {
    case FlatBuffers::CudaSphType::CudaSphType_SPH:
      fluid_particles = std::make_shared<CudaSphParticles>(
          CUDA_SPH_APP_PARAMS.max_num, volume_data.pos, volume_data.mass,
          volume_data.radius, volume_data.col);
      solver = std::make_shared<CudaSphSolver>(fluid_particles->maxSize());
      break;
    case FlatBuffers::CudaSphType::CudaSphType_MSSPH:
      fluid_particles = std::make_shared<CudaSphParticles>(
          CUDA_SPH_APP_PARAMS.max_num, volume_data.pos, volume_data.mass,
          volume_data.radius, volume_data.col);
      solver = std::make_shared<CudaMultiSphSolver>(fluid_particles->maxSize());
      CUDA_SPH_PARAMS.solver_type = MSSPH_SOLVER;
      break;
    case FlatBuffers::CudaSphType::CudaSphType_WCSPH:
      fluid_particles = std::make_shared<CudaSphParticles>(
          CUDA_SPH_APP_PARAMS.max_num, volume_data.pos, volume_data.mass,
          volume_data.radius, volume_data.col);
      adaptive_sub_timestep = true;
      CUDA_SPH_PARAMS.solver_type = WCSPH_SOLVER;
      solver = std::make_shared<CudaWCSphSolver>(fluid_particles->maxSize());
      break;
    case FlatBuffers::CudaSphType::CudaSphType_IISPH:
      CUDA_SPH_PARAMS.solver_type = IISPH_SOLVER;
      fluid_particles = std::make_shared<CudaIISphParticles>(
          CUDA_SPH_APP_PARAMS.max_num, volume_data.pos, volume_data.mass,
          volume_data.radius, volume_data.col);
      solver = std::make_shared<CudaIISphSolver>(fluid_particles->maxSize());
      break;
    case FlatBuffers::CudaSphType::CudaSphType_DFSPH:
      CUDA_SPH_PARAMS.solver_type = DFSPH_SOLVER;
      fluid_particles = std::make_shared<CudaDFSphParticles>(
          CUDA_SPH_APP_PARAMS.max_num, volume_data.pos, volume_data.mass,
          volume_data.radius, volume_data.col);
      // adaptive_sub_timestep = true;
      solver = std::make_shared<CudaDFSphSolver>(fluid_particles->maxSize());
      break;
    case FlatBuffers::CudaSphType_PBF:
      fluid_particles = std::make_shared<CudaPBFParticles>(
          CUDA_SPH_APP_PARAMS.max_num, volume_data.pos, volume_data.mass,
          volume_data.radius, volume_data.col);
      CUDA_SPH_PARAMS.solver_type = PBF_SOLVER;

      if (mName.find("incompressible") != std::string::npos || mName.find("bunny") != std::string::npos)
        solver =
            std::make_shared<CudaPBFSolver>(fluid_particles->maxSize(), true);
      else
        solver =
            std::make_shared<CudaPBFSolver>(fluid_particles->maxSize(), false);
      break;
    default:
      solver = std::make_shared<CudaSphSolver>(fluid_particles->maxSize());
      break;
    }

    CudaGNSearcherPtr searcher;

    if (CUDA_SPH_PARAMS.solver_type == SPH_SOLVER ||
        CUDA_SPH_PARAMS.solver_type == WCSPH_SOLVER)
    {
      searcher = std::make_shared<CudaGNSearcher>(
          CUDA_BOUNDARY_PARAMS.lowest_point, CUDA_BOUNDARY_PARAMS.highest_point,
          fluid_particles->maxSize(), CUDA_BOUNDARY_PARAMS.kernel_radius,
          SearcherParticleType::SPH);
    }
    else if (CUDA_SPH_PARAMS.solver_type == IISPH_SOLVER)
    {
      searcher = std::make_shared<CudaGNSearcher>(
          CUDA_BOUNDARY_PARAMS.lowest_point, CUDA_BOUNDARY_PARAMS.highest_point,
          fluid_particles->maxSize(), CUDA_BOUNDARY_PARAMS.kernel_radius,
          SearcherParticleType::IISPH);
    }
    else if (CUDA_SPH_PARAMS.solver_type == DFSPH_SOLVER)
    {
      searcher = std::make_shared<CudaGNSearcher>(
          CUDA_BOUNDARY_PARAMS.lowest_point, CUDA_BOUNDARY_PARAMS.highest_point,
          fluid_particles->maxSize(), CUDA_BOUNDARY_PARAMS.kernel_radius,
          SearcherParticleType::DFSPH);
    }
    else if (CUDA_SPH_PARAMS.solver_type == PBF_SOLVER)
    {
      searcher = std::make_shared<CudaGNSearcher>(
          CUDA_BOUNDARY_PARAMS.lowest_point, CUDA_BOUNDARY_PARAMS.highest_point,
          fluid_particles->maxSize(), CUDA_BOUNDARY_PARAMS.kernel_radius,
          SearcherParticleType::PBF);
    }
    else if (CUDA_SPH_PARAMS.solver_type == MSSPH_SOLVER)
    {

      CUDA_SPH_PARAMS.kernel_radius = volume_data.maxRadius * 4.f;
      CUDA_SPH_PARAMS.grid_size = make_int3((CUDA_BOUNDARY_PARAMS.highest_point -
                                             CUDA_BOUNDARY_PARAMS.lowest_point) /
                                            CUDA_SPH_PARAMS.kernel_radius);

      searcher = std::make_shared<CudaGNSearcher>(
          CUDA_BOUNDARY_PARAMS.lowest_point, CUDA_BOUNDARY_PARAMS.highest_point,
          fluid_particles->maxSize(), CUDA_SPH_PARAMS.kernel_radius,
          SearcherParticleType::SPH);
    }

    CudaGNBoundarySearcherPtr boundary_searcher =
        std::make_shared<CudaGNBoundarySearcher>(
            CUDA_BOUNDARY_PARAMS.lowest_point, CUDA_BOUNDARY_PARAMS.highest_point,
            boundary_particles->maxSize(), CUDA_BOUNDARY_PARAMS.kernel_radius);

    mSystem = std::make_shared<CudaSphSystem>(fluid_particles, boundary_particles,
                                              solver, searcher, boundary_searcher,
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

  void KiriSphApp::OnPBSUpdate(const KIRI::KiriTimeStep &DeltaTime)
  {

    if (CUDA_SPH_APP_PARAMS.move_boundary)
    {
      auto time = 0.05f * (CUDA_SPH_APP_PARAMS.current_frame -
                           CUDA_SPH_APP_PARAMS.move_boundary_frame);
      auto phi = sin(time);
      auto move_range = make_float3(0.5f, 0.f, 0.f);
      auto move_interval =
          move_range.x * phi > 0.f ? move_range * phi : make_float3(0.f);

      mSystem->moveBoundary(mInitLowestPoint, mInitHighestPoint + move_interval);

      mBoundaryModel->ResetBox(Vector3F(CUDA_BOUNDARY_PARAMS.world_center.x,
                                        CUDA_BOUNDARY_PARAMS.world_center.y,
                                        CUDA_BOUNDARY_PARAMS.world_center.z),
                               Vector3F(CUDA_BOUNDARY_PARAMS.world_size.x,
                                        CUDA_BOUNDARY_PARAMS.world_size.y,
                                        CUDA_BOUNDARY_PARAMS.world_size.z));
      mBoundaryEnity->SetModelMatrix(0, mBoundaryModel->GetModelMatrix());
    }

    if (CUDA_SPH_APP_PARAMS.run)
    {
      mSystem->updateSystemForVBO(mRenderInterval);
      SetParticleVBOWithRadius(mSystem->positionsVBO(), mSystem->colorsVBO(),
                               mSystem->numOfParticles());
    }
    else
    {
      CUDA_SPH_APP_PARAMS.move_boundary = false;
    }

    CUDA_SPH_APP_PARAMS.current_frame++;
  }

  void KiriSphApp::SetupPBSScene()
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
    KiriMaterialLampPtr boundary_material = std::make_shared<KiriMaterialLamp>();
    mBoundaryEnity =
        std::make_shared<KiriEntity>(id++, mBoundaryModel, boundary_material);

    mScene->Add(entity_roomtile);
    mScene->Add(mBoundaryEnity);
  }

  void KiriSphApp::OnImguiRender()
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
          const char *items[] = {
              "sph_muller",
              "sph_dambreak",
              "sph_bunny",
              "sph_surface_tension_akinci13",
              "sph_surface_tension_akinci13_bunny",
              "wcsph_dambreak",
              "wcsph_surface_tension_akinci13",
              "iisph_dambreak",
              "iisph_bunny",
              "dfsph_dambreak",
              "dfsph_bunny",
              "pbf_realtime",
              "pbf_incompressible",
              "pbf_bunny",
              "sph_emitter"};

          ImGui::Combo("Scene Config Data File",
                       &CUDA_SPH_APP_PARAMS.scene_data_idx, items,
                       IM_ARRAYSIZE(items));
          ChangeSceneConfigData(items[CUDA_SPH_APP_PARAMS.scene_data_idx]);
        }

        if (ImGui::CollapsingHeader("Simulation"))
        {

          if (ImGui::Button("Move Boundary"))
          {
            CUDA_SPH_APP_PARAMS.move_boundary =
                !CUDA_SPH_APP_PARAMS.move_boundary;
            CUDA_SPH_APP_PARAMS.move_boundary_frame =
                CUDA_SPH_APP_PARAMS.current_frame;
          }

          if (CUDA_SPH_EMITTER_PARAMS.enable)
            ImGui::Checkbox("Emit Particles", &CUDA_SPH_EMITTER_PARAMS.run);

          ImGui::Checkbox("Run", &CUDA_SPH_APP_PARAMS.run);
        }
        ImGui::End();
      }
    }
  }
} // namespace KIRI
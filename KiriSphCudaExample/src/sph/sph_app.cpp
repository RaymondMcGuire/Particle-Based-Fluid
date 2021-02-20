/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-27 00:49:33
 * @LastEditTime: 2021-02-20 01:26:23
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriExamples\src\sph\sph_app.cpp
 */

#include <sph/sph_app.h>
#include <imgui/include/imgui.h>

#include <kiri_pbs_cuda/sph/cuda_wcsph_solver.cuh>
#include <kiri_pbs_cuda/particle/particles_sampler_basic.h>

#include <fbs/generated/cuda_sph_app_generated.h>
#include <fbs/fbs_helper.h>

namespace KIRI
{

    void KiriSphApp::SetRenderFps(float Fps)
    {
        float sim_dt = CUDA_SPH_PARAMS.dt;
        float target_dt = 1.f / Fps;
        mSimRepeatNumer = (Int)(target_dt / sim_dt);
    }

    void KiriSphApp::SetupPBSParams()
    {
        auto scene_config_data = KIRI::FlatBuffers::GetCudaSphApp(mSceneConfigData.data());
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

        CUDA_SPH_PARAMS.dt = sph_data->fixed_dt();

        // scene data
        auto app_data = scene_config_data->app_data();
        auto scene_data = app_data->scene();
        CUDA_BOUNDARY_PARAMS.lowest_point = FbsToKiriCUDA(*scene_data->world_lower());
        CUDA_BOUNDARY_PARAMS.highest_point = FbsToKiriCUDA(*scene_data->world_upper());
        CUDA_BOUNDARY_PARAMS.world_size = FbsToKiriCUDA(*scene_data->world_size());
        CUDA_BOUNDARY_PARAMS.world_center = FbsToKiriCUDA(*scene_data->world_center());
        CUDA_BOUNDARY_PARAMS.kernel_radius = sph_data->kernel_radius();
        CUDA_BOUNDARY_PARAMS.grid_size = make_int3((CUDA_BOUNDARY_PARAMS.highest_point - CUDA_BOUNDARY_PARAMS.lowest_point) / CUDA_BOUNDARY_PARAMS.kernel_radius);

        // bgeo file export & render FPS
        CUDA_SPH_APP_PARAMS.bgeo_export = app_data->bgeo_export_mode_enable();
        if (app_data->render_mode_enable())
            SetRenderFps(app_data->render_mode_fps());
        else
            SetRenderFps(1.f / CUDA_SPH_PARAMS.dt);

        // camera data
        auto camera_data = scene_data->camera();
        mCamera->SetYawPitchPos(camera_data->yaw(), camera_data->pitch(), FbsToKiri(*camera_data->position()));

        // init volume data
        auto init_volume = scene_config_data->init_volume();

        // boundary sampling
        auto init_volume_box_size = FbsToKiriCUDA(*init_volume->box_size());
        auto init_volume_box_lower = FbsToKiriCUDA(*init_volume->box_lower());
        auto init_volume_box_color = FbsToKiriCUDA(*init_volume->box_color());

        auto diam = CUDA_SPH_PARAMS.particle_radius * 2.f;

        // sampling SPH init volume particles
        ParticlesSamplerBasicPtr mSampler = std::make_shared<ParticlesSamplerBasic>();
        auto bpos = mSampler->GetBoxSampling(CUDA_BOUNDARY_PARAMS.lowest_point, CUDA_BOUNDARY_PARAMS.highest_point, diam);

        Vec_Float3 pos;
        Vec_Float3 col;
        for (auto i = 0; i < init_volume_box_size.x; ++i)
        {
            for (auto j = 0; j < init_volume_box_size.y; ++j)
            {
                for (auto k = 0; k < init_volume_box_size.z; ++k)
                {
                    float3 p = make_float3(init_volume_box_lower.x + i * diam, init_volume_box_lower.y + j * diam, init_volume_box_lower.z + k * diam);

                    pos.emplace_back(p);
                    col.emplace_back(init_volume_box_color);
                }
            }
        }

        auto fluidParticles = std::make_shared<CudaSphParticles>(pos, col);
        auto boundaryParticles = std::make_shared<CudaBoundaryParticles>(bpos);
        KIRI_LOG_INFO("Number of Boundary Particles = {0}", boundaryParticles->Size());

        auto sph_solver_type = scene_config_data->sph_solver_type();
        CudaBaseSolverPtr pSolver;

        switch (sph_solver_type)
        {
        case FlatBuffers::CudaSphType::CudaSphType_SPH:
            pSolver = std::make_shared<CudaSphSolver>(
                fluidParticles->Size());
            break;
        case FlatBuffers::CudaSphType::CudaSphType_WCSPH:
            pSolver = std::make_shared<CudaWCSphSolver>(
                fluidParticles->Size());
            break;
        default:
            pSolver = std::make_shared<CudaSphSolver>(
                fluidParticles->Size());
            break;
        }

        CudaGNSearcherPtr searcher = std::make_shared<CudaGNSearcher>(
            CUDA_BOUNDARY_PARAMS.lowest_point,
            CUDA_BOUNDARY_PARAMS.highest_point,
            fluidParticles->Size(),
            CUDA_BOUNDARY_PARAMS.kernel_radius);

        CudaGNBoundarySearcherPtr boundarySearcher = std::make_shared<CudaGNBoundarySearcher>(
            CUDA_BOUNDARY_PARAMS.lowest_point,
            CUDA_BOUNDARY_PARAMS.highest_point,
            boundaryParticles->Size(),
            CUDA_BOUNDARY_PARAMS.kernel_radius);

        mSystem = std::make_shared<CudaSphSystem>(
            fluidParticles,
            boundaryParticles,
            pSolver,
            searcher,
            boundarySearcher);

        // ssf data
        auto ssf_data = scene_config_data->renderer_data();
        mFluidRenderSystem->EnableFluidTransparentMode(ssf_data->fluid_transparent_mode());
        mFluidRenderSystem->EnableSoildSsfMode(ssf_data->soild_particle_mode());

        // render particles
        SetParticleVBOWithRadius(mSystem->PositionsVBO(), mSystem->ColorsVBO(), mSystem->Size());
    }

    void KiriSphApp::OnPBSUpdate(const KIRI::KiriTimeStep &DeltaTime)
    {

        if (CUDA_SPH_APP_PARAMS.run)
        {
            for (int i = 0; i < mSimRepeatNumer; i++)
                mSystem->UpdateSystemForVBO();
            SetParticleVBOWithRadius(mSystem->PositionsVBO(), mSystem->ColorsVBO(), mSystem->Size());
        }
    }

    void KiriSphApp::SetupPBSScene()
    {
        float3 world_center = CUDA_BOUNDARY_PARAMS.world_center;
        float3 world_size = CUDA_BOUNDARY_PARAMS.world_size;

        // pre-computed map
        UInt irradianceCubeMap = mScene->getCubeSkybox()->getIrradianceCubeMap();
        UInt specCubeMap = mScene->getCubeSkybox()->getSpecularCubeMap();
        UInt brdfLUTTexure = mScene->getCubeSkybox()->getBrdfLutTexture();

        uint id = 0;
        // floor
        KiriPlanePtr plane = std::make_shared<KiriPlane>(100.0f, CUDA_BOUNDARY_PARAMS.lowest_point.y, Vector3F(0.0f, 1.0f, 0.0f));
        KiriPBRTexturePtr roomtile = std::make_shared<KiriPBRTexture>("tile");
        roomtile->Load();

        // floor material
        KiriMaterialPBRIBLTexPtr m_roomtile = std::make_shared<KiriMaterialPBRIBLTex>(
            irradianceCubeMap, specCubeMap, brdfLUTTexure, roomtile);
        m_roomtile->SetPointLights(mScene->getPointLights());
		plane->ResetModelMatrix();
        KiriEntityPtr entity_roomtile = std::make_shared<KiriEntity>(id++, plane, m_roomtile);

        // wall(no material)
        KiriBoxPtr boxModel = std::make_shared<KiriBox>(Vector3F(world_center.x, world_center.y, world_center.z), world_size.x, world_size.y, world_size.z);
        boxModel->SetWireFrame(true);
        KiriMaterialLampPtr m_debug = std::make_shared<KiriMaterialLamp>();
        KiriEntityPtr entity_debug = std::make_shared<KiriEntity>(id++, boxModel, m_debug);

        mScene->add(entity_roomtile);
        mScene->add(entity_debug);
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
                    ImGui::Combo("Render Mode", &SSF_DEMO_PARAMS.renderOpt, items, IM_ARRAYSIZE(items));
                }

                if (ImGui::CollapsingHeader("Example Scene"))
                {
                    const char *items[] = {"sph_standard_visc", "sph_atf_visc", "wcsph_standard_visc"};

                    ImGui::Combo("Scene Config Data File", &CUDA_SPH_APP_PARAMS.scene_data_idx, items, IM_ARRAYSIZE(items));
                    ChangeSceneConfigData(items[CUDA_SPH_APP_PARAMS.scene_data_idx]);
                }

                if (ImGui::CollapsingHeader("Simulation"))
                {
                    //ImGui::Checkbox("Emit Particles", &SPH_DEM_DEMO_PARAMS.EmitParticles);

                    if (ImGui::Button("Reset Simulation"))
                    {
                        SSF_DEMO_PARAMS.resetSSF = true;
                    }
                    ImGui::Checkbox("Run", &CUDA_SPH_APP_PARAMS.run);
                }
                ImGui::End();
            }
        }
    }
} // namespace KIRI
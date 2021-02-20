/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-20 19:14:59 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-05-15 00:06:01
 */
#include <kiri_core/kiri_scene.h>

KiriScene::KiriScene(UInt _w, UInt _h, KIRI::KiriCameraPtr camera)
{
    _particleRenderSys = std::make_shared<KIRI::KiriParticleRenderSystem>(camera);
    WINDOW_HEIGHT = _h;
    WINDOW_WIDTH = _w;
}

void KiriScene::Clear()
{
    entities.clear();
}

void KiriScene::add(KiriEntityPtr _entity)
{
    entities.append(_entity);
}

void KiriScene::add(KiriPointLightPtr _pl)
{
    pointLights.append(_pl);
}

void KiriScene::addDfs(KiriEntityPtr _entity)
{
    dfsEntities.append(_entity);
}

void KiriScene::addDfs(KiriPointLightPtr _pl)
{
    dfsPointLights.append(_pl);
}

void KiriScene::SetHDR(bool _hdr)
{
    hdr->SetHDR(_hdr);
}

void KiriScene::SetBloom(bool _bloom)
{
    hdr->SetBloom(_bloom);
}

void KiriScene::SetExposure(float _exposure)
{
    hdr->SetExposure(_exposure);
}

void KiriScene::enableHDR()
{
    enable_hdr = true;
    hdr = new KiriHDR(WINDOW_WIDTH, WINDOW_HEIGHT, true);
    hdr->enable();
}

void KiriScene::bindHDR()
{
    hdr->bindHDR();
}

void KiriScene::renderHDR()
{
    hdr->release();
    hdr->renderBloom();
    hdr->release();
    hdr->renderToScreen();
}

KiriHDR *KiriScene::getHDR()
{
    return hdr;
}

void KiriScene::SetUseNormalMapDF(bool _use_normal_map)
{
    deferredShading->SetUseNormalMap(_use_normal_map);
}

void KiriScene::SetUseSSAO(bool _use_ssao)
{
    deferredShading->SetUseSSAO(_use_ssao);
}

void KiriScene::enableDeferredShading(bool _ssao)
{
    enable_hdr = true;
    deferredShading = new KiriDeferredShading(WINDOW_WIDTH, WINDOW_HEIGHT);
    deferredShading->SetEntities(dfsEntities);
    deferredShading->SetPointLights(dfsPointLights);
    deferredShading->enable(_ssao);
}

void KiriScene::renderDF()
{
    deferredShading->render();
    glBindFramebuffer(GL_READ_FRAMEBUFFER, deferredShading->getGBuffer());
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0); // write to default framebuffer
    glBlitFramebuffer(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void KiriScene::enableCubeSkybox(bool _hdr, String _hdr_name)
{
    load_hdr = _hdr;
    cubeSkybox = std::make_shared<KiriCubeSkybox>(load_hdr, _hdr_name);
    enable_cubeSkybox = true;
}

KiriCubeSkyboxPtr KiriScene::getCubeSkybox()
{
    return cubeSkybox;
}

void KiriScene::renderCubeSkybox()
{
    if (enable_cubeSkybox)
    {
        if (load_hdr)
        {
            //KIRI_INFO << "Draw IBL CubeBox";
            cubeSkybox->drawIBL();
        }
        else
        {
            cubeSkybox->Draw();
        }
    }
}

void KiriScene::enableShadow(ShadowType _st)
{
    switch (_st)
    {
    case PointShadow:
        shadow = new KiriPointShadow();
        break;

    default:
        shadow = new KiriShadowMapping();
        break;
    }

    enable_shadow = true;
}

void KiriScene::renderShadow()
{
    if (enable_shadow)
    {
        //cout << "shadow enable" << endl;

        shadow->bind();
        entities.forEach([=](KiriEntityPtr _entity) {
            auto _model = _entity->getModel();
            _model->SetMaterial(shadow->getShadowDepthMaterial());
            _model->BindShader();
            _model->Draw();
        });
        shadow->release();
    }
}

// particle render
void KiriScene::enableParticleRenderSystem(bool enable_particle_render)
{
    _enable_particle_render = enable_particle_render;
    if (_enable_particle_render)
    {
        _particleRenderSys = std::make_shared<KIRI::KiriParticleRenderSystem>();
    }
}

void KiriScene::SetParticlesWithRadius(ArrayAccessor1<Vector4F> particles)
{
    if (_enable_particle_render)
    {
        _particleRenderSys->SetParticles(particles);
    }
    else
    {
        KIRI_LOG_ERROR("Please enable particle render system first!");
    }
}

void KiriScene::SetParticles(ArrayAccessor1<Vector3F> particles, float radius)
{
    if (_enable_particle_render)
    {
        _particleRenderSys->SetParticles(particles, radius);
    }
    else
    {
        KIRI_LOG_ERROR("Please enable particle render system first!");
    }
}

void KiriScene::SetParticlesVBO(UInt vbo, UInt num, float radius)
{
    if (_enable_particle_render)
    {
        _particleRenderSys->SetParticlesVBO(vbo, num, radius);
    }
    else
    {
        KIRI_LOG_ERROR("Please enable particle render system first!");
    }
}

void KiriScene::render()
{

    // render entity
    if (entities.size() != 0)
    {
        entities.forEach([](KiriEntityPtr _entity) {
            auto _model = _entity->getModel();
            _model->SetMaterial(_entity->GetMaterial());
            _model->BindShader();
            _model->Draw();
        });
    }
    else
    {
        //cout << "no entity" << endl;
    }

    if (pointLights.size() != 0)
    {
        pointLights.forEach([](KiriPointLightPtr _pl) {
            _pl->Draw();
        });
    }
    else
    {
        //cout << "no point light" << endl;
    }

    // render particles (point sprite render)
    if (_enable_particle_render)
    {
        if (_particleRenderSys->NumOfParticles() > 0)
        {
            _particleRenderSys->RenderParticles();
        }
    }
}
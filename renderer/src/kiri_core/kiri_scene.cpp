/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:26
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 13:54:31
 * @FilePath: \core\src\kiri_core\kiri_scene.cpp
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
/*
 * @Author: Xu.Wang
 * @Date: 2020-03-20 19:14:59
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-05-15 00:06:01
 */
#include <kiri_core/kiri_scene.h>

KiriScene::KiriScene(UInt _w, UInt _h)
{
    WINDOW_HEIGHT = _h;
    WINDOW_WIDTH = _w;
}

void KiriScene::Clear()
{
    mEntities.clear();
}

void KiriScene::Add(KiriEntityPtr _entity)
{
    mEntities.append(_entity);
}

void KiriScene::Add(KiriPointLightPtr _pl)
{
    mPointLights.append(_pl);
}

void KiriScene::AddDfs(KiriEntityPtr _entity)
{
    mDFSEntities.append(_entity);
}

void KiriScene::AddDfs(KiriPointLightPtr _pl)
{
    mDFSPointLights.append(_pl);
}

void KiriScene::SetHDR(bool _hdr)
{
    mHDR->SetHDR(_hdr);
}

void KiriScene::SetBloom(bool _bloom)
{
    mHDR->SetBloom(_bloom);
}

void KiriScene::SetExposure(float _exposure)
{
    mHDR->SetExposure(_exposure);
}

void KiriScene::EnableHDR()
{
    mEnableHDR = true;
    mHDR = new KiriHDR(WINDOW_WIDTH, WINDOW_HEIGHT, true);
    mHDR->Enable();
}

void KiriScene::BindHDR()
{
    mHDR->BindHDR();
}

void KiriScene::RenderHDR()
{
    mHDR->Release();
    mHDR->RenderBloom();
    mHDR->Release();
    mHDR->RenderToScreen();
}

KiriHDR *KiriScene::GetHDR()
{
    return mHDR;
}

void KiriScene::SetUseNormalMapDF(bool _use_normal_map)
{
    mDeferredShading->SetUseNormalMap(_use_normal_map);
}

void KiriScene::SetUseSSAO(bool _use_ssao)
{
    mDeferredShading->SetUseSSAO(_use_ssao);
}

void KiriScene::EnableDeferredShading(bool _ssao)
{
    mEnableHDR = true;
    mDeferredShading = new KiriDeferredShading(WINDOW_WIDTH, WINDOW_HEIGHT);
    mDeferredShading->SetEntities(mDFSEntities);
    mDeferredShading->SetPointLights(mDFSPointLights);
    mDeferredShading->Enable(_ssao);
}

void KiriScene::RenderDF()
{
    mDeferredShading->Render();
    glBindFramebuffer(GL_READ_FRAMEBUFFER, mDeferredShading->GetGBuffer());
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0); // write to default framebuffer
    glBlitFramebuffer(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void KiriScene::EnableCubeSkybox(bool _hdr, String _hdr_name)
{
    mLoadHDR = _hdr;
    mCubeSkybox = std::make_shared<KiriCubeSkybox>(mLoadHDR, _hdr_name);
    mEnableCubeSkybox = true;
}

KiriCubeSkyboxPtr KiriScene::GetCubeSkybox()
{
    return mCubeSkybox;
}

void KiriScene::RenderCubeSkybox()
{
    if (mEnableCubeSkybox)
    {
        if (mLoadHDR)
        {
            // KIRI_INFO << "Draw IBL CubeBox";
            mCubeSkybox->DrawIBL();
        }
        else
        {
            mCubeSkybox->Draw();
        }
    }
}

void KiriScene::EnableShadow(ShadowType _st)
{
    switch (_st)
    {
    case PointShadow:
        mShadow = new KiriPointShadow();
        break;

    default:
        mShadow = new KiriShadowMapping();
        break;
    }

    mEnableShadow = true;
}

void KiriScene::RenderShadow()
{
    if (mEnableShadow)
    {
        // cout << "mShadow Enable" << endl;

        mShadow->Bind();
        mEntities.forEach([=](KiriEntityPtr _entity)
                          {
            auto _model = _entity->GetModel();
            _model->SetMaterial(mShadow->GetShadowDepthMaterial());
            _model->BindShader();
            _model->Draw(); });
        mShadow->Release();
    }
}

void KiriScene::Render()
{

    // Render entity
    if (mEntities.size() != 0)
    {
        mEntities.forEach([](KiriEntityPtr _entity)
                          {
            auto _model = _entity->GetModel();
            _model->SetMaterial(_entity->GetMaterial());
            _model->BindShader();
            _model->Draw(); });
    }
    else
    {
        // cout << "no entity" << endl;
    }

    if (mPointLights.size() != 0)
    {
        mPointLights.forEach([](KiriPointLightPtr _pl)
                             { _pl->Draw(); });
    }
    else
    {
        // cout << "no point light" << endl;
    }
}
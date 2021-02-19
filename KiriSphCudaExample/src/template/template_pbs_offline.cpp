/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-26 17:26:05
 * @LastEditTime: 2021-02-19 11:14:10
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriExamples\src\template\template_pbs_offline.cpp
 */
#include <template/template_pbs_offline.h>
namespace KIRI
{

    void KiriTemplatePBSOffline::ChangeSceneConfigData(String Name)
    {
        if (mName == Name)
            return;

        mName = Name;
        mSceneConfigData = ImportBinaryFile(mName);
        auto &app = KiriApplication::Get();
        app.PopCurrentLayer();
        app.PushLayer(app.ExamplesList()[app.CurrentExampleName()]);
    }

    Vec_Char KiriTemplatePBSOffline::ImportBinaryFile(String const &Name)
    {
        String importPath = String(EXPORT_PATH) + "sceneconfig/" + Name + ".bin";

        if (RELEASE && PUBLISH)
        {
            importPath = "./resources/sceneconfig/" + Name + ".bin";
        }

        std::ifstream importer(importPath, std::ios::binary);
        KIRI_LOG_INFO("Import Scene Conifg File From:{0}", importPath);

        return Vec_Char(std::istreambuf_iterator<char>(importer),
                        std::istreambuf_iterator<char>());
    }

    void KiriTemplatePBSOffline::OnAttach()
    {
        SetupPBSParams();
        SetupPBSScene();
    }

    void KiriTemplatePBSOffline::OnUpdate(const KIRI::KiriTimeStep &DeltaTime)
    {
        //KIRI_LOG_TRACE("Delta Time {0}, ms: {1}, fps: {2}", DeltaTime.GetSeconds(), DeltaTime.GetMilliSeconds(), DeltaTime.GetFps());
        OnPBSUpdate(DeltaTime);

        KIRI::KiriRendererCommand::SetViewport(Vector4F(0.f, 0.f, (float)mWidth, (float)mHeight));
        KIRI::KiriRendererCommand::SetClearColor(Vector4F(0.1f, 0.1f, 0.1f, 1.f));
        KIRI::KiriRendererCommand::Clear();
    }

    void KiriTemplatePBSOffline::OnEvent(KIRI::KiriEvent &e)
    {
        //KIRI_LOG_TRACE("{0}", e);
    }
} // namespace KIRI
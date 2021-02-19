/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-25 14:02:18
 * @LastEditTime: 2021-02-18 20:32:22
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriExamples\src\kiri_app.cpp
 */

#include <kiri_app.h>
#include <kiri_entry_point.h>
#include <sph/sph_app.h>

namespace KIRI
{
    KiriApp::KiriApp()
    {
        auto &app = KiriApplication::Get();
        UInt height = app.GetWindow().GetWindowHeight();
        UInt width = app.GetWindow().GetWindowWidth();

        SetCurrentExampleName("sph_app");
        AddExample("sph_app", new KiriSphApp("sph_standard_visc", width, height));

        PushLayer(ExamplesList()[CurrentExampleName()]);
    }

    KiriApp::~KiriApp() {}

    KIRI::KiriApplicationPtr KIRI::CreateApplication()
    {
        return std::make_shared<KiriApp>();
    }
} // namespace KIRI
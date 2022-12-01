/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:27
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 11:50:08
 * @FilePath: \Kiri\app\src\kiri_app.cpp
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#include <kiri_app.h>
#include <kiri_entry_point.h>

#include <sph/multisph_ren14_app.h>
#include <sph/multisph_yang15_app.h>
#include <sph/sph_app.h>

namespace KIRI
{
  KiriApp::KiriApp()
  {
    auto &app = KiriApplication::Get();
    UInt height = app.GetWindow().GetWindowHeight();
    UInt width = app.GetWindow().GetWindowWidth();

    this->SetCurrentExampleName("sph_app");

    this->AddExample("sph_app",
                     std::make_shared<KiriSphApp>("sph_muller", width, height));
    this->AddExample("multisph_ren14_app", std::make_shared<KiriMultiSphRen14App>(
                                               "multisph_ren2014_non_miscible", width, height));
    this->AddExample("multisph_yang15_app", std::make_shared<KiriMultiSphYang15App>(
                                                "multisph_yang2015_two_phase", width, height));

    this->PushLayer(this->CurrentExampleName());
  }

  KiriApp::~KiriApp() {}

  KIRI::KiriApplicationPtr KIRI::CreateApplication()
  {
    return std::make_shared<KiriApp>();
  }
} // namespace KIRI
/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2021-02-26 11:37:06
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-06-24 15:51:20
 * @FilePath: \Kiri\KiriCore\include\kiri_core\gui\layer_imgui.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#ifndef _KIRI_LAYER_IMGUI_H_
#define _KIRI_LAYER_IMGUI_H_
#pragma once
#include <kiri_core/gui/layer.h>

namespace KIRI {
class KiriLayerImGui : public KiriLayer {
public:
  KiriLayerImGui();
  ~KiriLayerImGui();

  virtual void OnAttach() override;
  virtual void OnDetach() override;
  virtual void OnImguiRender() override;

  void begin();
  void end();

  void SwitchApp(String AppName);

protected:
  void ShowFPS(bool *Fps);
  void ShowSceneInfo(bool *SceneInfo);

private:
  bool mScreenShot = false;
  bool mFps = true;
  bool mSceneInfo = false;
};
typedef SharedPtr<KiriLayerImGui> KiriLayerImGuiPtr;
} // namespace KIRI
#endif
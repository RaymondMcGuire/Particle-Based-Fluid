/***
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2022-05-17 16:36:28
 * @LastEditors: Xu.WANG
 * @Description:
 */

#ifndef _KIRI_LAYER_STACK_H_
#define _KIRI_LAYER_STACK_H_
#pragma once
#include <kiri_core/gui/layer.h>

namespace KIRI {
class KiriLayerStack {
public:
  KiriLayerStack();
  ~KiriLayerStack();

  void PushLayer(KiriLayerPtr layer);
  void PushOverlay(KiriLayerPtr layer);
  void PopLayer(KiriLayerPtr layer);
  void PopOverlay(KiriLayerPtr layer);

  Vector<KiriLayerPtr>::iterator begin() { return mLayers.begin(); };
  Vector<KiriLayerPtr>::iterator end() { return mLayers.end(); };

  const auto size() const { return mLayers.size(); }
  constexpr auto empty() const { return this->size() == 0 ? true : false; }

private:
  UInt mLayerInsertIndex = 0;
  Vector<KiriLayerPtr> mLayers;
};
} // namespace KIRI
#endif
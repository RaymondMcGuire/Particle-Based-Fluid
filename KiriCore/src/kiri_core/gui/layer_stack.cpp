/*** 
 * @Author: Jayden Zhang
 * @Date: 2020-10-25 03:25:34
 * @LastEditTime: 2021-02-18 20:14:35
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\src\kiri_core\gui\layer_stack.cpp
 */
#include <kiri_core/gui/layer_stack.h>

namespace KIRI
{
    KiriLayerStack::KiriLayerStack() {}

    KiriLayerStack::~KiriLayerStack()
    {
        for (KiriLayer *layer : mLayerStack)
            delete layer;
    }

    void KiriLayerStack::PushLayer(KiriLayer *layer)
    {
        mLayerStack.emplace_back(layer);
    }

    void KiriLayerStack::PopLayer(KiriLayer *layer)
    {
        auto it = std::find(mLayerStack.begin(), mLayerStack.end(), layer);
        if (it != mLayerStack.end())
            mLayerStack.erase(it);
    }

} // namespace KIRI

/*** 
 * @Author: Jayden Zhang
 * @Date: 2020-10-25 03:25:34
 * @LastEditTime: 2021-02-24 21:30:47
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
    }

    void KiriLayerStack::PushLayer(KiriLayerPtr layer)
    {
        mLayers.emplace(mLayers.begin() + mLayerInsertIndex, layer);
        mLayerInsertIndex++;
    }

    void KiriLayerStack::PushOverlay(KiriLayerPtr overlay)
    {
        mLayers.emplace_back(overlay);
    }

    void KiriLayerStack::PopLayer(KiriLayerPtr layer)
    {
        auto it = std::find(mLayers.begin(), mLayers.end(), layer);
        if (it != mLayers.end())
        {
            mLayers.erase(it);
            mLayerInsertIndex--;
            layer->OnDetach();
        }
    }

    void KiriLayerStack::PopOverlay(KiriLayerPtr overlay)
    {
        auto it = std::find(mLayers.begin(), mLayers.end(), overlay);
        if (it != mLayers.end())
        {
            mLayers.erase(it);
            overlay->OnDetach();
        }
    }
} // namespace KIRI

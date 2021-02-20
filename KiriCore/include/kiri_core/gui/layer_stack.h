/*** 
 * @Author: Jayden Zhang
 * @Date: 2020-10-25 03:19:55
 * @LastEditTime: 2021-02-18 19:27:31
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\include\kiri_core\gui\layer_stack.h
 */
#ifndef _KIRI_LAYER_STACK_H_
#define _KIRI_LAYER_STACK_H_
#pragma once
#include <kiri_core/gui/layer.h>

namespace KIRI
{
    class KiriLayerStack
    {
    public:
        KiriLayerStack();
        ~KiriLayerStack();

        void PushLayer(KiriLayer *layer);
        void PopLayer(KiriLayer *layer);

        Vector<KiriLayer *>::iterator begin() { return mLayerStack.begin(); };
        Vector<KiriLayer *>::iterator end() { return mLayerStack.end(); };

        const auto Size() const { return mLayerStack.size(); }

    private:
        Vector<KiriLayer *> mLayerStack;
    };
} // namespace KIRI
#endif
/*** 
 * @Author: Jayden Zhang
 * @Date: 2020-10-25 03:19:55
 * @LastEditTime: 2021-02-24 20:01:52
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\include\kiri_core\gui\layer.h
 */

#ifndef _KIRI_LAYER_H_
#define _KIRI_LAYER_H_

#pragma once

#include <kiri_core/event/event.h>
#include <kiri_core/camera/camera.h>

namespace KIRI
{
    class KiriLayer
    {
    public:
        KiriLayer(const String &mName = "KiriLayer");
        virtual ~KiriLayer() noexcept {};

        virtual void OnAttach(){};
        virtual void OnDetach(){};
        virtual void OnUpdate(const KiriTimeStep &deltaTime){};
        virtual void OnEvent(KiriEvent &e){};
        virtual void OnImguiRender(){};

        inline String &GetName() { return mName; };

    protected:
        String mName;
    };
    typedef SharedPtr<KiriLayer> KiriLayerPtr;
} // namespace KIRI

#endif
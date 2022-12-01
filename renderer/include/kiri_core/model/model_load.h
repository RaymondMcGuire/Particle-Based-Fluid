/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-18 01:04:15
 * @LastEditTime: 2021-02-20 19:37:00
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\model\model_load.h
 */

#ifndef _KIRI_MODEL_LOAD_H_
#define _KIRI_MODEL_LOAD_H_
#pragma once
#include <kiri_core/model/model.h>

class KiriModelLoad : public KiriModel
{
public:
    virtual void Draw() = 0;
    virtual void WireFrameMode(bool WireFrame) { bWireFrame = WireFrame; }

protected:
    bool bWireFrame = false;
};

typedef SharedPtr<KiriModelLoad> KiriModelLoadPtr;
#endif
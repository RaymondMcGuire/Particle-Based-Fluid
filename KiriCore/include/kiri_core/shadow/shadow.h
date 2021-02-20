/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:39:57
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\shadow\shadow.h
 */

#ifndef _KIRI_SHADOW_H_
#define _KIRI_SHADOW_H_
#pragma once

#include <kiri_core/material/material.h>
class KiriShadow
{
public:
    virtual ~KiriShadow() {}
    virtual void enable(Vector3F) = 0;
    virtual void bind() = 0;
    virtual void release() = 0;

    virtual KiriMaterialPtr getShadowDepthMaterial() = 0;

protected:
    const UInt SHADOW_WIDTH = 1024, SHADOW_HEIGHT = 1024;
    UInt depthMapFBO;
};

#endif
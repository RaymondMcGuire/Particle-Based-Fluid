/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:26
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 14:24:46
 * @FilePath: \core\include\kiri_core\shadow\shadow.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#ifndef _KIRI_SHADOW_H_
#define _KIRI_SHADOW_H_
#pragma once

#include <kiri_core/material/material.h>
class KiriShadow
{
public:
    virtual ~KiriShadow() {}
    virtual void Enable(Vector3F) = 0;
    virtual void Bind() = 0;
    virtual void Release() = 0;

    virtual KiriMaterialPtr GetShadowDepthMaterial() = 0;

protected:
    const UInt SHADOW_WIDTH = 1024, SHADOW_HEIGHT = 1024;
    UInt depthMapFBO;
};

#endif
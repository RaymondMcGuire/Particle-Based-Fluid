/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:26
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 14:27:38
 * @FilePath: \core\include\kiri_core\model\model_custom.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#ifndef _KIRI_MODEL_CUSTOM_H_
#define _KIRI_MODEL_CUSTOM_H_
#pragma once
#include <kiri_core/model/model_internal.h>
#include <kiri_core/mesh/mesh_custom.h>

class KiriModelCustom : public KiriModelInternal
{
public:
    KiriModelCustom();

    void SetMesh(KiriMeshCustom *_mesh) { mMesh = _mesh; }

    void Draw() override;

    bool bWireFrame = true;
};
typedef SharedPtr<KiriModelCustom> KiriModelCustomPtr;
#endif
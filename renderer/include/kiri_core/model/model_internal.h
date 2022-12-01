/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:26
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 14:28:03
 * @FilePath: \core\include\kiri_core\model\model_internal.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#ifndef _KIRI_MODEL_INTERNAL_H_
#define _KIRI_MODEL_INTERNAL_H_
#pragma once

#include <kiri_core/model/model.h>
#include <kiri_core/mesh/mesh_internal.h>

class KiriModelInternal : public KiriModel
{
public:
    virtual void Draw() = 0;
    void UpdateInstance(Array1<Matrix4x4F>);
    void SetWireFrame(bool wf);

protected:
    KiriMeshInternal *mMesh;
    bool bWireFrame = false;
};
typedef SharedPtr<KiriModelInternal> KiriModelInternalPtr;
#endif
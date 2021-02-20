/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-18 01:04:15
 * @LastEditTime: 2021-02-20 19:36:25
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\model\model_internal.h
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
    KiriMeshInternal *mesh;
    bool bWireFrame = false;
};
typedef SharedPtr<KiriModelInternal> KiriModelInternalPtr;
#endif
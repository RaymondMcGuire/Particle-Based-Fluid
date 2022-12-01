/***
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:35:29
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath: \KiriCore\include\kiri_core\model\model_cube.h
 */

#ifndef _KIRI_MODEL_CUBE_H_
#define _KIRI_MODEL_CUBE_H_
#pragma once
#include <kiri_core/model/model_internal.h>
#include <kiri_core/mesh/mesh_cube.h>

class KiriCube : public KiriModelInternal
{
public:
    KiriCube();

    void SetDiffuseMap(bool);
    void LoadDiffuseMap(UInt);

    void SetRenderOutside(bool);
    void Draw() override;

protected:
    bool outside = true;

    bool mDiffuse = false;
    UInt diffuseTexure;
};
typedef SharedPtr<KiriCube> KiriCubePtr;
#endif
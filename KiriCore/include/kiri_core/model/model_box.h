/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-18 01:04:15
 * @LastEditTime: 2021-02-20 19:35:25
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\model\model_box.h
 */

#ifndef _KIRI_MODEL_BOX_H_
#define _KIRI_MODEL_BOX_H_
#pragma once
#include <kiri_core/model/model_cube.h>

class KiriBox : public KiriCube
{
public:
    KiriBox() { KiriBox(1.0f, 1.0f, 1.0f); }
    KiriBox(float xSide, float ySide, float zSide);
    KiriBox(Vector3F center, float xSide, float ySide, float zSide);

    void resetBox(float xSide, float ySide, float zSide);
    void resetBox(Vector3F center, float xSide, float ySide, float zSide);

private:
    float _xSide;
    float _ySide;
    float _zSide;
};
typedef SharedPtr<KiriBox> KiriBoxPtr;
#endif
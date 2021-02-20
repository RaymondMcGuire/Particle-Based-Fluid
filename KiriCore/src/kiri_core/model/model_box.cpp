/*
 * @Author: Xu.Wang 
 * @Date: 2020-04-29 18:19:44 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-04-29 18:55:58
 */

#include <kiri_core/model/model_box.h>

KiriBox::KiriBox(float xSide, float ySide, float zSide)
{
    resetBox(Vector3F(0.f), xSide, ySide, zSide);
}

KiriBox::KiriBox(Vector3F center, float xSide, float ySide, float zSide)
{
    resetBox(center, xSide, ySide, zSide);
}

void KiriBox::resetBox(float xSide, float ySide, float zSide)
{
    resetBox(Vector3F(0.f), xSide, ySide, zSide);
}

void KiriBox::resetBox(Vector3F center, float xSide, float ySide, float zSide)
{
    _xSide = xSide;
    _ySide = ySide;
    _zSide = zSide;

    ResetModelMatrix();
    Translate(center);
    Scale(Vector3F(_xSide / 2.0f, _ySide / 2.0f, _zSide / 2.0f));
}
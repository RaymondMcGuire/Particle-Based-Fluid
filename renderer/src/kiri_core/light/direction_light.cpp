/*
 * @Author: Xu.Wang
 * @Date: 2020-03-17 18:06:56
 * @Last Modified by:   Xu.Wang
 * @Last Modified time: 2020-03-17 18:06:56
 */
#include <kiri_core/light/direction_light.h>

KiriDirectionLight::KiriDirectionLight(Vector3F _direction, Vector3F _diffuse)
{
    direction = _direction;
    mDiffuse = _diffuse;
}

KiriDirectionLight::~KiriDirectionLight()
{
}
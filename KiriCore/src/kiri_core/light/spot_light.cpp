/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-17 18:07:14 
 * @Last Modified by:   Xu.Wang 
 * @Last Modified time: 2020-03-17 18:07:14 
 */
#include <kiri_core/light/spot_light.h>

KiriSpotLight::KiriSpotLight(Vector3F _position, Vector3F _direction, Vector3F _diffuse)
{
    position = _position;
    direction = _direction;
    diffuse = _diffuse;
}

KiriSpotLight::~KiriSpotLight()
{
    //cout << "delete KiriSpotLight" << endl;
}
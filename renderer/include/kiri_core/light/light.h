/***
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:46:02
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath: \KiriCore\include\kiri_core\light\light.h
 */

#ifndef _KIRI_LIGHT_H_
#define _KIRI_LIGHT_H_
#pragma once
#include <kiri_core/model/model.h>

class KiriLight
{
public:
    KiriModelPtr GetModel()
    {
        return model;
    }

    virtual void Draw()
    {
        model->Draw();
    }

    String mName;
    size_t id;

protected:
    KiriModelPtr model;
};
#endif
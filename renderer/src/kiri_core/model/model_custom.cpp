/***
 * @Author: Xu.WANG
 * @Date: 2021-12-23 16:50:56
 * @LastEditTime: 2021-12-23 16:51:30
 * @LastEditors: Xu.WANG
 * @Description:
 */

#include <kiri_core/model/model_custom.h>

KiriModelCustom::KiriModelCustom()
{
    mInstance = false;
}

void KiriModelCustom::Draw()
{

    KiriModel::Draw();

    if (bWireFrame)
    {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    }
    mMesh->Draw();
    if (bWireFrame)
    {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
}
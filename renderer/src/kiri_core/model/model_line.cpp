/*** 
 * @Author: Xu.WANG
 * @Date: 2021-04-07 14:20:49
 * @LastEditTime: 2021-04-07 15:23:21
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\src\kiri_core\model\model_line.cpp
 */

#include <kiri_core/model/model_line.h>

void KiriLine::Draw()
{
    KiriModel::Draw();
    mMesh->Draw();
}
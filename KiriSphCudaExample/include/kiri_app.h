/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-26 15:29:22
 * @LastEditTime: 2020-11-15 18:26:27
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriExamples\include\kiri_app.h
 */

#ifndef _KIRI_APP_H_
#define _KIRI_APP_H_

#include <kiri_application.h>
namespace KIRI
{
    class KiriApp : public KiriApplication
    {
    public:
        KiriApp();
        ~KiriApp();
    };

    typedef SharedPtr<KiriApp> KiriAppPtr;
} // namespace KIRI
#endif
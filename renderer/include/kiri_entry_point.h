
/*** 
 * @Author: Jayden Zhang
 * @Date: 2020-10-25 13:57:02
 * @LastEditTime: 2021-02-20 02:06:52
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_entry_point.h
 */
#ifndef _KIRI_ENTRY_POINT_H_
#define _KIRI_ENTRY_POINT_H_
#pragma once

#ifdef KIRI_WINDOWS
extern KIRI::KiriApplicationPtr KIRI::CreateApplication();

int main(int argc, char **argv)
{
    KIRI::KiriLog::Init();
    auto app = KIRI::CreateApplication();
    app->Run();
}
#else
#error KIRI only support Windows 64-bit for now
#endif

#endif
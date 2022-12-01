/*** 
 * @Author: Jayden Zhang
 * @Date: 2020-10-25 12:25:41
 * @LastEditTime: 2021-02-20 19:39:37
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\renderer\renderer_context.h
 */

#ifndef _KIRI_RENDERER_CONTEXT_H_
#define _KIRI_RENDERER_CONTEXT_H_
#pragma once
namespace KIRI
{
    class KiriRendererContext
    {
    public:
        virtual void Init() = 0;
        virtual void SwapBuffers() = 0;
    };
} // namespace KIRI

#endif
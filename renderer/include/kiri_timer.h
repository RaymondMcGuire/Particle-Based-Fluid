/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:26
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 13:41:33
 * @FilePath: \core\include\kiri_timer.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _KIRI_TIMER_H_
#define _KIRI_TIMER_H_
#pragma once
#include <kiri_pch.h>

namespace KIRI
{
    class KiriTimeStep
    {
    public:
        KiriTimeStep(float time = 0.0f)
            : mTime(time)
        {
        }

        operator float() const { return mTime; }

        inline const float Seconds() const { return mTime; }
        inline const float MilliSeconds() const { return mTime * 1000.0f; }
        inline const float Fps() const { return 1.f / mTime; }

    private:
        float mTime;
    };

    class KiriTimer
    {
    public:
        KiriTimer() : mName("Default")
        {
            Restart();
        }

        explicit KiriTimer(const String &name) : mName(name)
        {
            Restart();
        }

        inline void Restart()
        {
            mStartTime = std::chrono::steady_clock::now();
        }

        inline double Elapsed(bool Restart = false)
        {
            mEndTime = std::chrono::steady_clock::now();
            std::chrono::duration<double> diff = mEndTime - mStartTime;
            if (Restart)
                this->Restart();
            return diff.count();
        }

        /***
         * @description:
         * @param {reset:reset timer or not; unitMs: print ms / sec; tip: print extra info; kill: after print timer, kill thread or not}
         * @return {void}
         */
        void Log(bool reset = false, const String &tip = "",
                 bool unitMs = true, bool kill = false)
        {
            if (unitMs)
            {
                if (tip.length() > 0)
                    KIRI_LOG_INFO("KiriTimer({0}) Time Elapsed:{1} ms", tip, Elapsed() * 1000.f);
                else
                    KIRI_LOG_INFO("KiriTimer({0}) Time Elapsed:{1} ms", mName, Elapsed() * 1000.f);
            }
            else
            {
                if (tip.length() > 0)
                    KIRI_LOG_INFO("KiriTimer({0}) Time Elapsed:{1} s", tip, Elapsed());
                else
                    KIRI_LOG_INFO("KiriTimer({0}) Time Elapsed:{1} s", mName, Elapsed());
            }

            if (reset)
                this->Restart();

            if (kill)
                exit(5);
        }

    private:
        std::chrono::steady_clock::time_point mStartTime;
        std::chrono::steady_clock::time_point mEndTime;
        String mName;
    };
} // namespace KIRI

#endif //_KIRI_TIMER_H_
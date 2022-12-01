/*** 
 * @Author: Xu.WANG
 * @Date: 2020-06-16 01:32:28
 * @LastEditTime: 2021-02-20 18:27:22
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\include\kiri_params.h
 */

#ifndef _KIRI_PARAMS_H_
#define _KIRI_PARAMS_H_

#pragma once

#include <kiri_pch.h>

struct CameraParams
{
    bool debug;
    CameraParams(bool _debug)
    {
        debug = _debug;
    };
};

struct SSFDemoParams
{
    bool startSSF;
    bool resetSSF;

    bool moveBoundary;
    Int boundaryMode;
    Int moveBoundaryDTime;
    Int moveBoundaryUTime;
    Int moveBoundaryLTime;
    Int moveBoundaryRTime;
    Int moveBoundaryBTime;
    Int moveBoundaryFTime;

    Int currentFrame;

    // SSF Params
    bool particleView;
    Int renderOpt;

    // PBF Params
    float dt;
    Int maxIterNums;
    float coefXSPH;
    float sCorrK;
    float sCorrN;

    Int boxParticleType;

    SSFDemoParams(bool _startSSF, bool _resetSSF, bool _particleView)
    {
        startSSF = _startSSF;
        resetSSF = _resetSSF;

        moveBoundary = false;
        currentFrame = moveBoundaryRTime = moveBoundaryLTime = moveBoundaryDTime = moveBoundaryUTime = moveBoundaryBTime = moveBoundaryFTime = 0;

        particleView = _particleView;
        renderOpt = 4;
    };
};

extern CameraParams CAMERA_PARAMS;
extern SSFDemoParams SSF_DEMO_PARAMS;
#endif
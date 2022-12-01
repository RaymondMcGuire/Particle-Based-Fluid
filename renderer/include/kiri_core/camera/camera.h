/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-18 01:04:15
 * @LastEditTime: 2021-03-24 20:41:40
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\include\kiri_core\camera\camera.h
 */

#ifndef _KIRI_CAMERA_H_
#define _KIRI_CAMERA_H_

#pragma once
#include <glad/glad.h>
#include <kiri_params.h>

namespace KIRI
{
    enum CameraViewMode
    {
        PERSPECTIVE,
        ORTHOGRAPHIC
    };

    enum CameraMovementType
    {
        FORWARD,
        BACKWARD,
        LEFT,
        RIGHT
    };

    struct CameraProperty
    {
        Vector3F LookFrom;
        Vector3F LookAt;
        Vector3F VUp;

        // frustum camera
        float VFov;
        float Aspect;

        // ortho camera
        float VLeft;
        float VRight;
        float VTop;
        float VBottom;

        // common
        float ZNear;
        float ZFar;

        CameraViewMode ViewMode;

        CameraProperty(
            Vector3F lookFrom,
            Vector3F lookAt,
            Vector3F vUp,
            float vFov,
            float aspect,
            float left = -1.f,
            float right = 1.f,
            float top = 1.f,
            float bottom = -1.f,
            float zNear = 0.1f,
            float zFar = 100.0f,
            CameraViewMode viewMode = CameraViewMode::PERSPECTIVE)
            : LookFrom(lookFrom),
              LookAt(lookAt),
              VUp(vUp),
              VFov(vFov),
              Aspect(aspect),
              VLeft(left),
              VRight(right),
              VTop(top),
              VBottom(bottom),
              ZNear(zNear),
              ZFar(zFar),
              ViewMode(viewMode) {}
    };

    class KiriCamera
    {
    public:
        explicit KiriCamera(const CameraProperty &cameraProperty)
            : mCameraData(cameraProperty)
        {
            CAMERA_PARAMS.debug = false;
        }

        KiriCamera(const KiriCamera &) = delete;
        KiriCamera &operator=(const KiriCamera &) = delete;

        Matrix4x4F inverseViewMatrix() const { return mVMatrix.inverse(); }
        Matrix4x4F ViewMatrix() const { return mVMatrix; }
        Matrix4x4F ProjectionMatrix() const { return mPMatrix; }

        Vector3F Position() const { return mCameraData.LookFrom; }
        Vector3F Target() const { return mCameraData.LookAt; }
        Vector3F WorldUp() const { return mCameraData.VUp; }

        float GetAspect() const { return mCameraData.Aspect; }
        float GetFov() const { return mCameraData.VFov; }

        float GetOrthoLeft() const { return mCameraData.VLeft; }
        float GetOrthoRight() const { return mCameraData.VRight; }
        float GetOrthoTop() const { return mCameraData.VTop; }
        float GetOrthoButtom() const { return mCameraData.VBottom; }

        float GetNear() const { return mCameraData.ZNear; }
        float GetFar() const { return mCameraData.ZFar; }

        void SetPosition(Vector3F Position)
        {
            mCameraData.LookFrom = Position;
            Update();
        }

        void SetTarget(Vector3F Target)
        {
            mCameraData.LookAt = Target;
            Update();
        }

        virtual void OnUpdate(const KIRI::KiriTimeStep &DeltaTime)
        {
            OnUpdateKeyBoard(DeltaTime);
            OnUpdateMouseMovement(DeltaTime);
            OnUpdateMouseScroll(DeltaTime);
            Update();
        }

        virtual void ProcessKeyboard(CameraMovementType, float) = 0;
        virtual void ProcessMouseMovement(float, float, GLboolean = true) = 0;
        virtual void ProcessMouseScroll(float) = 0;

    protected:
        virtual void UpdateCameraMatrix() = 0;
        virtual void OnUpdateKeyBoard(const KIRI::KiriTimeStep &DeltaTime) = 0;
        virtual void OnUpdateMouseMovement(const KIRI::KiriTimeStep &DeltaTime) = 0;
        virtual void OnUpdateMouseScroll(const KIRI::KiriTimeStep &DeltaTime) = 0;

        CameraProperty mCameraData;

        void UpdateCamParams()
        {
            bDebugMode = CAMERA_PARAMS.debug;
        }

        virtual void Update()
        {
            UpdateCamParams();
            UpdateCameraMatrix();
        }

        Matrix4x4F mVMatrix;
        Matrix4x4F mPMatrix;

        // params
        bool bDebugMode = false;
    };
    typedef SharedPtr<KiriCamera> KiriCameraPtr;
}
#endif
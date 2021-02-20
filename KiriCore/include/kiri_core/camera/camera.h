/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-18 01:04:15
 * @LastEditTime: 2021-02-20 00:28:02
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
    enum Camera_Movement
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

        float VFov;
        float Aspect;
        float ZNear;
        float ZFar;

        // Constructor
        CameraProperty(
            Vector3F lookFrom,
            Vector3F lookAt,
            Vector3F vUp,
            float vFov,
            float aspect,
            float zNear = 0.1f,
            float zFar = 100.0f)
            : LookFrom(lookFrom),
              LookAt(lookAt),
              VUp(vUp),
              VFov(vFov),
              Aspect(aspect),
              ZNear(zNear),
              ZFar(zFar) {}
    };

    class KiriCamera
    {
    public:
        KiriCamera(const CameraProperty &cameraProperty)
        {
            CAMERA_PARAMS.debug = false;
            mCameraData.Position = cameraProperty.LookFrom;
            mCameraData.Target = cameraProperty.LookAt;
            mCameraData.VFov = cameraProperty.VFov;
            mCameraData.Aspect = cameraProperty.Aspect;
            mCameraData.ZNear = cameraProperty.ZNear;
            mCameraData.ZFar = cameraProperty.ZFar;
            mCameraData.WorldUp = cameraProperty.VUp;
        }

        Matrix4x4F inverseViewMatrix() const { return mVMatrix.inverse(); }
        Matrix4x4F ViewMatrix() const { return mVMatrix; }
        Matrix4x4F ProjectionMatrix() const { return mPMatrix; }

        Vector3F Position() const { return mCameraData.Position; }
        Vector3F Target() const { return mCameraData.Target; }
        Vector3F WorldUp() const { return mCameraData.WorldUp; }

        float GetAspect() const { return mCameraData.Aspect; }
        float GetFov() const { return mCameraData.VFov; }
        float GetNear() const { return mCameraData.ZNear; }
        float GetFar() const { return mCameraData.ZFar; }

        void SetPosition(Vector3F Position)
        {
            mCameraData.Position = Position;
            Update();
        }

        void SetTarget(Vector3F Target)
        {
            mCameraData.Target = Target;
            Update();
        }

        virtual void OnUpdate(const KIRI::KiriTimeStep &DeltaTime)
        {
            OnUpdateKeyBoard(DeltaTime);
            OnUpdateMouseMovement(DeltaTime);
            OnUpdateMouseScroll(DeltaTime);
            Update();
        }

        virtual void ProcessKeyboard(Camera_Movement, float) = 0;
        virtual void ProcessMouseMovement(float, float, GLboolean = true) = 0;
        virtual void ProcessMouseScroll(float) = 0;

    protected:
        virtual void UpdateCameraMatrix() = 0;
        virtual void OnUpdateKeyBoard(const KIRI::KiriTimeStep &DeltaTime) = 0;
        virtual void OnUpdateMouseMovement(const KIRI::KiriTimeStep &DeltaTime) = 0;
        virtual void OnUpdateMouseScroll(const KIRI::KiriTimeStep &DeltaTime) = 0;

        struct CameraData
        {
            Vector3F Position;
            Vector3F Target;
            Vector3F WorldUp;

            float VFov;
            float Aspect;
            float ZNear;
            float ZFar;
        };

        CameraData mCameraData;

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
/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-18 01:04:15
 * @LastEditTime: 2021-03-24 20:42:10
 * @LastEditors: Xu.WANG
 * @Description: First Person Camera
 * @FilePath: \Kiri\KiriCore\include\kiri_core\camera\camera_fpc.h
 */

#ifndef _KIRI_CAMERA_FPC_H_
#define _KIRI_CAMERA_FPC_H_

#pragma once

#include <kiri_core/camera/camera.h>

namespace KIRI
{

    class KiriCameraFPC : public KiriCamera
    {
    public:
        KiriCameraFPC(
            const CameraProperty &cameraProperty,
            float Yaw = -90.0f,
            float Pitch = 0.0f,
            float Speed = 2.5f,
            float Sensitivity = 0.1f)
            : KiriCamera(cameraProperty)
        {
            mYaw = Yaw;
            mPitch = Pitch;
            mSpeed = Speed;
            mSensitivity = Sensitivity;
            mZoom = cameraProperty.VFov;
            mConstrainPitch = true;
            Update();
        }

        void ProcessKeyboard(CameraMovementType, float) override;
        void ProcessMouseMovement(float, float, GLboolean = true) override;
        void ProcessMouseScroll(float) override;

        void SetYawPitchPos(float Yaw, float Pitch, Vector3F Position)
        {
            mYaw = Yaw;
            mPitch = Pitch;
            mCameraData.LookFrom = Position;
            Update();
        }

    protected:
        void UpdateCameraMatrix() override;
        virtual void OnUpdateKeyBoard(const KIRI::KiriTimeStep &DeltaTime) override;
        virtual void OnUpdateMouseMovement(const KIRI::KiriTimeStep &DeltaTime) override;
        virtual void OnUpdateMouseScroll(const KIRI::KiriTimeStep &DeltaTime) override;

    private:
        float mYaw;
        float mPitch;
        float mSpeed;
        float mSensitivity;
        float mZoom;
        bool mConstrainPitch;

        Vector2F mLastPos;
        Vector3F mFront;
        Vector3F mUp;
        Vector3F mRight;
    };

    typedef SharedPtr<KiriCameraFPC> KiriCameraFPCPtr;
}
#endif
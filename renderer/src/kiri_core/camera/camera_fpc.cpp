/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-18 01:04:15
 * @LastEditTime: 2021-03-24 20:44:05
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\src\kiri_core\camera\camera_fpc.cpp
 */

#include <kiri_core/camera/camera_fpc.h>

#include <kiri_keycodes.h>
#include <kiri_mbtncodes.h>
#include <kiri_core/gui/opengl/opengl_input.h>

namespace KIRI
{
    void KiriCameraFPC::OnUpdateKeyBoard(const KIRI::KiriTimeStep &DeltaTime)
    {
        float velocity = mSpeed * DeltaTime;

        if (KIRI::KiriInput::IsKeyDown(KIRI_KEY_UP) || KIRI::KiriInput::IsKeyDown(KIRI_KEY_W))
            mCameraData.LookFrom += mFront * velocity;
        if (KIRI::KiriInput::IsKeyDown(KIRI_KEY_DOWN) || KIRI::KiriInput::IsKeyDown(KIRI_KEY_S))
            mCameraData.LookFrom -= mFront * velocity;
        if (KIRI::KiriInput::IsKeyDown(KIRI_KEY_LEFT) || KIRI::KiriInput::IsKeyDown(KIRI_KEY_A))
            mCameraData.LookFrom -= mRight * velocity;
        if (KIRI::KiriInput::IsKeyDown(KIRI_KEY_RIGHT) || KIRI::KiriInput::IsKeyDown(KIRI_KEY_D))
            mCameraData.LookFrom += mRight * velocity;
    }
    void KiriCameraFPC::OnUpdateMouseMovement(const KIRI::KiriTimeStep &DeltaTime)
    {
        Vector2F pos = KIRI::KiriInput::GetMousePos();

        if (KIRI::KiriInput::IsMouseButtonDown(KIRI_MOUSE_BUTTON_RIGHT))
        {

            Vector2F offset = Vector2F(pos.x - mLastPos.x, mLastPos.y - pos.y) * mSensitivity;
            mYaw += offset.x;
            mPitch += offset.y;

            // Make sure that when mPitch is out of bounds, screen doesn't get flipped
            if (mConstrainPitch)
            {
                if (mPitch > 89.0f)
                    mPitch = 89.0f;
                if (mPitch < -89.0f)
                    mPitch = -89.0f;
            }
        }

        mLastPos = pos;
    }
    void KiriCameraFPC::OnUpdateMouseScroll(const KIRI::KiriTimeStep &DeltaTime)
    {
    }

    void KiriCameraFPC::UpdateCameraMatrix()
    {
        Vector3F _mFront;
        _mFront.x = cos(kiri_math::degreesToRadians<float>(mYaw)) * cos(kiri_math::degreesToRadians<float>(mPitch));
        _mFront.y = sin(kiri_math::degreesToRadians<float>(mPitch));
        _mFront.z = sin(kiri_math::degreesToRadians<float>(mYaw)) * cos(kiri_math::degreesToRadians<float>(mPitch));
        mFront = _mFront.normalized();

        mRight = mFront.cross(mCameraData.VUp).normalized();
        mUp = mRight.cross(mFront);

        mVMatrix = Matrix4x4F::viewMatrix(mCameraData.LookFrom, mCameraData.LookFrom + mFront, mUp);

        if (mCameraData.ViewMode == CameraViewMode::PERSPECTIVE)
            mPMatrix = Matrix4x4F::perspectiveMatrix(mCameraData.VFov, mCameraData.Aspect, mCameraData.ZNear, mCameraData.ZFar);
        else if (mCameraData.ViewMode == CameraViewMode::ORTHOGRAPHIC)
            mPMatrix = Matrix4x4F::orthoMatrix(mCameraData.VLeft, mCameraData.VRight, mCameraData.VTop, mCameraData.VBottom, mCameraData.ZNear, mCameraData.ZFar);

        if (bDebugMode)
        {
            KIRI_LOG_DEBUG("Camera LookFrom=({0:f},{1:f},{2:f}); mYaw={3:f}; mPitch={4:f}", mCameraData.LookFrom.x, mCameraData.LookFrom.y, mCameraData.LookFrom.z, mYaw, mPitch);
        }
    }

    void KiriCameraFPC::ProcessKeyboard(CameraMovementType direction, float deltaTime)
    {
        float velocity = mSpeed * deltaTime;
        if (direction == FORWARD)
            mCameraData.LookFrom += mFront * velocity;
        if (direction == BACKWARD)
            mCameraData.LookFrom -= mFront * velocity;
        if (direction == LEFT)
            mCameraData.LookFrom -= mRight * velocity;
        if (direction == RIGHT)
            mCameraData.LookFrom += mRight * velocity;

        Update();
    }

    // Processes input received from a mouse input system. Expects the offset value in both the x and y direction.
    void KiriCameraFPC::ProcessMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch)
    {
        xoffset *= mSensitivity;
        yoffset *= mSensitivity;

        mYaw += xoffset;
        mPitch += yoffset;

        // Make sure that when mPitch is out of bounds, screen doesn't get flipped
        if (constrainPitch)
        {
            if (mPitch > 89.0f)
                mPitch = 89.0f;
            if (mPitch < -89.0f)
                mPitch = -89.0f;
        }

        Update();
    }

    // Processes input received from a mouse scroll-wheel event. Only requires input on the vertical wheel-axis
    void KiriCameraFPC::ProcessMouseScroll(float yoffset)
    {
        if (mZoom >= 1.0f && mZoom <= 45.0f)
            mZoom -= yoffset;
        if (mZoom <= 1.0f)
            mZoom = 1.0f;
        if (mZoom >= 45.0f)
            mZoom = 45.0f;

        Update();
    }
}
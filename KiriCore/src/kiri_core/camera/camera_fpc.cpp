/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-18 01:04:15
 * @LastEditTime: 2021-02-20 00:28:51
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
            mCameraData.Position += mFront * velocity;
        if (KIRI::KiriInput::IsKeyDown(KIRI_KEY_DOWN) || KIRI::KiriInput::IsKeyDown(KIRI_KEY_S))
            mCameraData.Position -= mFront * velocity;
        if (KIRI::KiriInput::IsKeyDown(KIRI_KEY_LEFT) || KIRI::KiriInput::IsKeyDown(KIRI_KEY_A))
            mCameraData.Position -= mRight * velocity;
        if (KIRI::KiriInput::IsKeyDown(KIRI_KEY_RIGHT) || KIRI::KiriInput::IsKeyDown(KIRI_KEY_D))
            mCameraData.Position += mRight * velocity;
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

        mRight = mFront.cross(mCameraData.WorldUp).normalized();
        mUp = mRight.cross(mFront);

        mVMatrix = Matrix4x4F::viewMatrix(mCameraData.Position, mCameraData.Position + mFront, mUp);
        mPMatrix = Matrix4x4F::perspectiveMatrix(mCameraData.VFov, mCameraData.Aspect, mCameraData.ZNear, mCameraData.ZFar);

        if (bDebugMode)
        {
            KIRI_LOG_DEBUG("Camera Position=({0:f},{1:f},{2:f}); mYaw={3:f}; mPitch={4:f}", mCameraData.Position.x, mCameraData.Position.y, mCameraData.Position.z, mYaw, mPitch);
        }
    }

    void KiriCameraFPC::ProcessKeyboard(Camera_Movement direction, float deltaTime)
    {
        float velocity = mSpeed * deltaTime;
        if (direction == FORWARD)
            mCameraData.Position += mFront * velocity;
        if (direction == BACKWARD)
            mCameraData.Position -= mFront * velocity;
        if (direction == LEFT)
            mCameraData.Position -= mRight * velocity;
        if (direction == RIGHT)
            mCameraData.Position += mRight * velocity;

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
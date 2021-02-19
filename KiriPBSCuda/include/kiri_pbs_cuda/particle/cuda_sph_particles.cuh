/*
 * @Author: Xu.WANG
 * @Date: 2021-02-04 12:36:10
 * @LastEditTime: 2021-02-09 12:59:57
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\particle\cuda_sph_particles.cuh
 */

#ifndef _CUDA_SPH_PARTICLES_CUH_
#define _CUDA_SPH_PARTICLES_CUH_

#pragma once

#include <kiri_pbs_cuda/particle/cuda_particles.cuh>

namespace KIRI
{
	class CudaSphParticles final : public CudaParticles
	{
	public:
		explicit CudaSphParticles::CudaSphParticles(
			const Vec_Float3 &p,
			const Vec_Float3 &col)
			: CudaParticles(p),
			  mVel(p.size()),
			  mAcc(p.size()),
			  mCol(p.size()),
			  mPressure(p.size()),
			  mDensity(p.size()),
			  mMass(p.size())
		{
			KIRI_CUCALL(cudaMemcpy(mCol.Data(), &col[0], sizeof(float3) * col.size(), cudaMemcpyHostToDevice));
		}

		CudaSphParticles(const CudaSphParticles &) = delete;
		CudaSphParticles &operator=(const CudaSphParticles &) = delete;

		float3 *GetVelPtr() const { return mVel.Data(); }
		float3 *GetAccPtr() const { return mAcc.Data(); }
		float3 *GetColPtr() const { return mCol.Data(); }
		float *GetPressurePtr() const { return mPressure.Data(); }
		float *GetDensityPtr() const { return mDensity.Data(); }
		float *GetMassPtr() const { return mMass.Data(); }

		virtual ~CudaSphParticles() noexcept {}

		void Advect(const float dt);

	protected:
		CudaArray<float3> mVel;
		CudaArray<float3> mAcc;
		CudaArray<float3> mCol;
		CudaArray<float> mPressure;
		CudaArray<float> mDensity;
		CudaArray<float> mMass;
	};

	typedef SharedPtr<CudaSphParticles> CudaSphParticlesPtr;
} // namespace KIRI

#endif
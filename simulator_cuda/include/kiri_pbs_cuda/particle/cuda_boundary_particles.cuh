/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2021-02-08 21:45:16
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-22 17:14:33
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\particle\cuda_boundary_particles.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#ifndef _CUDA_BOUNDARY_PARTICLES_CUH_
#define _CUDA_BOUNDARY_PARTICLES_CUH_

#pragma once

#include <kiri_pbs_cuda/particle/cuda_particles.cuh>

namespace KIRI
{
	class CudaBoundaryParticles final : public CudaParticles
	{
	public:
		explicit CudaBoundaryParticles::CudaBoundaryParticles(
			const Vec_Float3 &p)
			: CudaParticles(p),
			  mVolume(p.size()),
			  mLabel(p.size()) {}

		explicit CudaBoundaryParticles::CudaBoundaryParticles(
			const Vec_Float3 &p,
			const Vec_SizeT &label)
			: CudaParticles(p),
			  mVolume(p.size()),
			  mLabel(p.size())
		{
			KIRI_CUCALL(cudaMemcpy(mLabel.data(), &label[0], sizeof(size_t) * label.size(), cudaMemcpyHostToDevice));
		}

		explicit CudaBoundaryParticles::CudaBoundaryParticles(
			const size_t numOfMaxParticles,
			const Vec_Float3 &p,
			const Vec_SizeT &label)
			: CudaParticles(numOfMaxParticles, p),
			  mVolume(numOfMaxParticles),
			  mLabel(numOfMaxParticles)
		{
			if (!p.empty())
				KIRI_CUCALL(cudaMemcpy(mLabel.data(), &label[0], sizeof(size_t) * label.size(), cudaMemcpyHostToDevice));
		}

		CudaBoundaryParticles(const CudaBoundaryParticles &) = delete;
		CudaBoundaryParticles &operator=(const CudaBoundaryParticles &) = delete;

		float *volumePtr() const { return mVolume.data(); }
		size_t *labelPtr() const { return mLabel.data(); }

		virtual ~CudaBoundaryParticles() noexcept {}

	protected:
		CudaArray<float> mVolume;
		CudaArray<size_t> mLabel;
	};

	typedef SharedPtr<CudaBoundaryParticles> CudaBoundaryParticlesPtr;
} // namespace KIRI

#endif
# Particle-Based Fluid

[![WindowsCUDA](https://github.com/RaymondMcGuire/Particle-Based-Fluid-Toolkit/actions/workflows/WindowsCUDA.yml/badge.svg?branch=master)](https://github.com/RaymondMcGuire/Particle-Based-Fluid-Toolkit/actions/workflows/WindowsCUDA.yml)

This project implemented several SPH-related papers using CUDA, including Weakly Compressible SPH (SCA2007), Position-based Fluid (SIGGRAPH2013), Implicit Incompressible SPH (TVCG2014), Divergence-free SPH (SCA2015), Volume Fraction-based Multiple-fluid (SIGGRAPH2014) and Helmholtz Free Energy based Multiple-fluid (SIGGRAPH2015).

## Environment

- C++ & CUDA11.6
- Install [CUDA](https://developer.nvidia.com/cuda-downloads) and [Cmake](https://cmake.org/download/) first

## Setup

### Command Line

```rb
cd /to/your/project/path
```

```rb
mkdir build
```

```rb
cd build
```

```rb
cmake .. -G"Visual Studio 17 2022" -A x64
```

### Scripts

#### For Windows

- cd to ./scripts folder
- choose your visual studio version(vs15/vs17/vs19/vs22)
- run the bat file

## Gallery
| Example | GIF |
| --- | --- |
| SPH | ![SPH](docs/gif/sph_surface_bunny.gif) | 
| WCSPH | ![WCSPH](docs/gif/wcsph_surface.gif) | 
| IISPH | ![WCSPH](docs/gif/iisph_bunny.gif) | 
| DFSPH | ![WCSPH](docs/gif/dfsph_bunny.gif) | 
| Volume-Fraction Based Multiple-Fluid : Non-Miscible | ![MFNM](docs/gif/ren14_non_miscible.gif) | 
| Volume-Fraction Based Multiple-Fluid : Miscible | ![MFM](docs/gif/ren14_miscible.gif) | 
| Helmholtz Free Energy Based Multiple-Fluid : Miscible  | ![MFM](docs/gif/yang15_miscible.gif) | 


## Papers implemented

 * Müller, Matthias, David Charypar, and Markus H. Gross. "Particle-based fluid simulation for interactive applications." Symposium on Computer animation. 2003.
 * Becker, Markus, and Matthias Teschner. "Weakly compressible SPH for free surface flows." Proceedings of the 2007 ACM SIGGRAPH/Eurographics symposium on Computer animation. 2007.
 * Akinci, Nadir, et al. "Versatile rigid-fluid coupling for incompressible SPH." ACM Transactions on Graphics (TOG) 31.4 (2012): 1-8.
 * Ihmsen, M., Cornelis, J., Solenthaler, B., Horvath, C., & Teschner, M. (2013). Implicit incompressible SPH. IEEE transactions on visualization and computer graphics, 20(3), 426-435.
 * Bender, J., & Koschier, D. (2015, August). Divergence-free smoothed particle hydrodynamics. In Proceedings of the 14th ACM SIGGRAPH/Eurographics symposium on computer animation (pp. 147-155).
 * Ren, B., Li, C., Yan, X., Lin, M. C., Bonet, J., & Hu, S. M. (2014). Multiple-fluid SPH simulation using a mixture model. ACM Transactions on Graphics (TOG), 33(5), 1-11.
 * Yang, T., Chang, J., Ren, B., Lin, M. C., Zhang, J. J., & Hu, S. M. (2015). Fast multiple-fluid simulation using Helmholtz free energy. ACM Transactions on Graphics (TOG), 34(6), 1-11.

cd ..
mkdir build_cuda
cd build_cuda
cmake .. -G "Visual Studio 17 2022" -A x64 -DENABLE_CUDA=TRUE

pause

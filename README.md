# pyflow
Python 3 bindings for CUDA optical flow algorithms in OpenCV 3

### Install

* CUDA (Tested with CUDA 9)
* OpenCV 3 (Tested with OpenCV 3.4.5) from source
* Python 3 (Tested with Python 3.6)

OpenCV compilation requires gcc-6 and --expt-relaxed-constexpr flag when building:
```
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_CUDA=ON -D WITH_CUBLAS=ON -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_OPENGL=ON -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF -D CUDA_NVCC_FLAGS="-D_FORCE_INLINES" -D CMAKE_C_COMPILER=gcc-6 -D CMAKE_CXX_COMPILER=g++-6 -D CUDA_NVCC_FLAGS=--expt-relaxed-constexpr ..
```

After `make` and `sudo make install` add the following equivalent to your `.bashrc` just in case so `setup.py` can find `libopencv_cudaoptflow.so.3.4.5`:

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{YOUR OPENCV DIR}/opencv-3.4.5/build/lib
```

Then clone this repo, build it and do a system wide install. 

```
python setup.py build_ext -i
python setup.py install --user
```

Refer to `demo.py` on how to use it.

### Methods

* Brox `pyflow.brox()`
* TVL1 `pyflow.tvl1()`
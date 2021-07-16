# FaceDetectors
Compare latency and accuracy of face detection algorithm with my own video data set.


# OpenCV compile
to use OpenCV's dnn module with Nvidia GPUs, CUDA, and cuDNN.  
The following descriptions were written by referring to [Ref.1](https://www.pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/) and [Ref.2](https://webnautes.tistory.com/1435).

1. install OpenCV and GPU dependencies
    * Package upgrade  
      ```
      $ sudo apt-get update
      $ sudo apt-get upgrade
      ```
    * install proper dependencies
      ```
      $ sudo apt-get install build-essential cmake unzip pkg-config
      $ sudo apt-get install libjpeg-dev libpng-dev libtiff-dev
      $ sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev
      $ sudo apt-get install libv4l-dev libxvidcore-dev libx264-dev
      $ sudo apt-get install libgtk-3-dev
      $ sudo apt-get install libatlas-base-dev gfortran
      $ sudo apt-get install python3-dev
      ```
2. Download OpenCV source
    ```
    $ cd ~
    $ wget -O opencv.zip https://github.com/opencv/opencv/archive/4.5.0.zip
    $ wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.5.0.zip
    $ unzip opencv.zip
    $ unzip opencv_contrib.zip
    $ mv opencv-4.5.0 opencv
    $ mv opencv_contrib-4.5.0 opencv_contrib
    ```
3. make virtual environment and install the numpy 
    ```
    $ mkvirtualenv opencv_cuda -p python3
    $ pip install numpy
    ```
    if you haven't inatalled the virtualenv package yet, you can use the following command:
      ```
      $ wget https://bootstrap.pypa.io/get-pip.py
      $ sudo python3 get-pip.py
      $ sudo pip install virtualenv virtualenvwrapper
      $ sudo rm -rf ~/get-pip.py ~/.cache/pip
      ```
      insert the following on the `~/.bashrc`:
      ```
      # virtualenv and virtualenvwrapper
      export WORKON_HOME=$HOME/.virtualenvs
      export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
      source /usr/local/bin/virtualenvwrapper.sh
      ```
4. Compile OpenCV
    ```
    $ cd ~/opencv
    $ mkdir build
    $ cd build
    $ cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D INSTALL_C_EXAMPLES=OFF \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D WITH_CUDA=ON \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D CUDA_ARCH_BIN=7.5 \
      -D CUDA_ARCH_PTX=7.5 \
      -D WITH_CUBLAS=1 \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
      -D HAVE_opencv_python3=ON \
      -D PYTHON_EXECUTABLE=~/.virtualenvs/opencv_cuda/bin/python \
      -D BUILD_EXAMPLES=ON ..
    $ time make -j$(nproc)
    $ sudo make install
    $ sudo ldconfig
    $ cd ~/.virtualenvs/opencv_cuda/lib/python3.8/site-packages/
    $ ln -s /usr/local/lib/python3.8/site-packages/cv2/python-3.8/cv2.cpython-38m-x86_64-linux-gnu.so cv2.so
    ```
    
5. Test on python
    ```
    $ python3
    >>> import cv2
    >>> print(cv2.cuda.getCudaEnabledDeviceCount())
    1
    >>>quit()
    ```

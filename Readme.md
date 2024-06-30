# Toy ML sample

Wanted to try something cool with basic ml deployed on webgpu using naive kernels with fp32.

# Windows Setup

* Install visual studio for native extensions
* Install CUDA SDK 11.8

```sh
# To install local python and setup venv and start powershell instance
$ setup_win_env.bat

# Might need and extra
$ pip install -v .
# or
$ python.exe setup.py build
# To build the native extensions

$ python.exe .\tests\test_torch.py

torch.cuda.get_device_name(0)=NVIDIA GeForce GTX 1650 Ti
All tests passed.

```
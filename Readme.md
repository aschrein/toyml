# Toy ML sample

Wanted to try something cool with basic ml deployed on webgpu using naive kernels with fp32.

# Windows Setup

```sh
# To install local python and setup venv and start powershell instance
$ setup_win_env.bat

$ python.exe .\tests\test_torch.py

torch.cuda.get_device_name(0)=NVIDIA GeForce GTX 1650 Ti
All tests passed.

```
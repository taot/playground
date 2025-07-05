

Install libtorch
```
paru -S libtorch-cuda
```


Run with python environment and libtorch shared library
```bash
time PYTHONPATH=/home/taot/programming/rust/Deep-Reinforcement-Learning-Hands-On-Third-Edition-Rust/venv313/lib/python3.13/site-packages/ LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH target/debug/chapter04_01_cartpole
```

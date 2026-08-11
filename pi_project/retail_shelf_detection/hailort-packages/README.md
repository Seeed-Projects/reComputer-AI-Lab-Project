# HailoRT Python wheel

Place exactly one HailoRT Python wheel in this directory before building the
Docker image. It must match all of the following:

- HailoRT major/minor version used by the host driver and firmware (4.23.x)
- Python 3.11 (`cp311`)
- Linux aarch64

Example filename:

```text
hailort-4.23.0-cp311-cp311-linux_aarch64.whl
```

Download HailoRT from the Hailo Developer Zone. The wheel is not committed to
this repository because access and redistribution are governed by Hailo.


# Third-party code

This repository includes code adapted from the following projects.

## mocca_envs (UBC MOCCA lab)

`vfplan/envs/` (environment, robot, pybullet utilities and the `walker3d.xml`
model) is adapted from the UBC MOCCA lab's `mocca_envs`
(https://github.com/UBCMOCCA/mocca_envs), which in turn builds on the pybullet
locomotion environments (`pybullet_envs`, zlib license).

## ALLSTEPS (Xie et al., 2020)

`vfplan/controller/` contains the low-level footstep controller trained following
"ALLSTEPS: Curriculum-driven Learning of Stepping Stone Skills"
(https://github.com/belinghy/SteppingStone).

## Diffuser (Janner et al., 2022)

`vfplan/diffusion/` is adapted from https://github.com/jannerm/diffuser:

    MIT License

    Copyright (c) 2022 Michael Janner

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

## robomimic

The `MLP` class in `vfplan/encoders.py` is adapted from robomimic
(https://github.com/ARISE-Initiative/robomimic, MIT License,
Copyright (c) 2021 Stanford Vision and Learning Lab).

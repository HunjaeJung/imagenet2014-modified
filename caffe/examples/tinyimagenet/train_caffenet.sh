#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/our_tinyimagenet_caffenet/solver.prototxt

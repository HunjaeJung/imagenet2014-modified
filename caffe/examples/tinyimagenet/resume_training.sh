#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/our_tinyimagenet_caffenet/solver.prototxt \
    --snapshot=models/our_tinyimagenet_caffenet/caffenet_train_10000.solverstate

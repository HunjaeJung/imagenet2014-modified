#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=examples/tinyimagenet
DATA=data/tiny-imagenet-200
TOOLS=build/tools

$TOOLS/compute_image_mean $EXAMPLE/tinyimagenet_train_lmdb \
  $DATA/tinyimagenet_mean.binaryproto

echo "Done."

#!/bin/sh

export _ADEVAL_OLD_PATH="$PATH"
export _ADEVAL_OLD_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
export PATH="$PATH:/usr/local/cuda/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/usr/local/cuda/lib64"
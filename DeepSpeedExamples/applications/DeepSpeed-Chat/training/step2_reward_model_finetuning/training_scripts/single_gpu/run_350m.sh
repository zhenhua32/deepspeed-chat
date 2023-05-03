#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT

deepspeed --num_gpus 1 main.py --model_name_or_path facebook/opt-350m \
   --num_padding_at_beginning 1 --weight_decay 0.1 --disable_dropout --gradient_accumulation_steps 16 --zero_stage $ZERO_STAGE \
   --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
   --data_path Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets \
   --deepspeed --output_dir $OUTPUT &> $OUTPUT/training.log

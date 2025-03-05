#!/bin/bash

LOGDIR="./logs"
rm -rf $LOGDIR && mkdir -p $LOGDIR 

for gen_model in "gpt-3.5-turbo" "gpt-4" "Meta-Llama-3.1-8B" "gpt2-xl" "gpt-neo-2.7b" "gpt-j-6B" "opt-2.7b"
do
    for dataset in "xsum" "writing" "squad"
    do
        for temp in 0.8
        do
            sbatch -J ${gen_model}_${dataset}_perterb \
                --gres=gpu:1 \
                --mem=80G \
                --output=$LOGDIR/slurm_%j.out \
                --qos=normal \
                --wrap="python3 ./scripts/generate_perterbs.py --gen-model=${gen_model} --dataset=${dataset} --temp=${temp}"
        done
    done
done

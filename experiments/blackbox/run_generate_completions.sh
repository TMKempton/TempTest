#!/bin/bash

LOGDIR="./logs"
rm -rf $LOGDIR && mkdir -p $LOGDIR 

for model in "Meta-Llama-3.1-8B" "gpt2-xl" "gpt-neo-2.7b" "gpt-j-6B" "opt-2.7b"
do
    for dataset in "xsum" "writing" "squad"
    do
        for temp in 0.8
        do
            echo "Running ./scripts/generate_completions.py ${model} on ${dataset} with temperature ${temp}"
            sbatch -J ${model}_${dataset} \
                --gres=gpu:1 \
                --output=$LOGDIR/slurm_%j.out \
                --qos=normal \
                --wrap="python3 ./scripts/generate_completions.py ${model} ${dataset} ${temp}"
        done
    done
done
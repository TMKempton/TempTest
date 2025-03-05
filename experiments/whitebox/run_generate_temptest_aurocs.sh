#!/bin/bash
set -e

LOGDIR="./logs"
mkdir -p $LOGDIR 

for model in "Meta-Llama-3.1-8B" "gpt2-xl" "gpt-neo-2.7b" "gpt-j-6B" "opt-2.7b"
do
    for dataset in "xsum" "writing" "squad"
    do
        for gen_temp in 0.8
        do
            for score_temp in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 # Used in ./experiments/graybox
            do
                sbatch --wrap="python3 ./experiments/whitebox/scripts/generate_temptest_aurocs.py ${dataset} ${model} ${gen_temp} ${score_temp}"
            done
        done
    done
done

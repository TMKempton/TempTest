#!/bin/bash
set -e

LOGDIR="./logs"
mkdir -p $LOGDIR 
for gen_model in gpt-3.5-turbo gpt-4 #"Meta-Llama-3.1-8B"
do
    for score_model in "Meta-Llama-3.1-8B" "gpt-neo-2.7b" # "gpt2-xl" "gpt-neo-2.7b" "gpt-j-6B" "opt-2.7b"
    do
        for dataset in "xsum" "writing" # "squad"
        do
            for gen_temp in 0.8
            do
                for score_temp in 0.5 0.6 0.7 0.8 0.9 1.0
                do
                    sbatch --wrap="python3 ./experiments/blackbox/scripts/generate_temptest_aurocs.py ${dataset} ${gen_model} ${score_model} ${gen_temp} ${score_temp}"
                done
            done
        done
    done
done

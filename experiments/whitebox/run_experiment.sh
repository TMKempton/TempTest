#!/bin/bash
set -e

LOGDIR="./logs"
mkdir -p $LOGDIR 

# # Run TempTest
for model in "Meta-Llama-3.1-8B" "gpt2-xl" "gpt-neo-2.7b" "gpt-j-6B" "opt-2.7b" 
do
    for dataset in "xsum" "writing" "squad"
    do
        for gen_temp in 0.8
        do
            for score_temp in 0.1 0.2 0.3 0.4
            do
                sbatch -J tt_${model}_${dataset}_${temp} \
                    --gres=gpu:1 \
                    --mem=80G \
                    --nice=10000 \
                    --output=$LOGDIR/slurm_%j.out \
                    --qos=normal \
                    --wrap="python3 ./experiments/whitebox/scripts/temptest.py ${dataset} ${model} ${gen_temp} ${score_temp}" 
            done
        done
    done
done


# Run FastDetect
for model in "Meta-Llama-3.1-8B" "gpt2-xl" "gpt-neo-2.7b" "gpt-j-6B" "opt-2.7b" 
do
    for dataset in "xsum" "writing" "squad"
    do
        for temp in 0.8
        do
            for num_tokens in 25 50 75 100
            do
                sbatch -J fd_${model}_${dataset}_${temp}_${num_tokens} \
                    --gres=gpu:1 \
                    --mem=80G \
                    --nice=10000 \
                    --output=$LOGDIR/slurm_%j.out \
                    --qos=normal \
                    --wrap="python3 ./experiments/whitebox/scripts/fastdetect_gpt.py ${dataset} ${model} ${temp} ${num_tokens}"  
            done
        done
    done
done

# Run DetectGPT
for model in "Meta-Llama-3.1-8B" "gpt2-xl" "gpt-neo-2.7b" "gpt-j-6B" "opt-2.7b" 
do
    for dataset in "xsum" "writing" "squad"
    do
        for temp in 0.8
        do
            for num_tokens in 25 50 75 100
            do
                sbatch -J dgpt_${model}_${dataset}_${temp}_${num_tokens} \
                    --gres=gpu:1 \
                    --mem=80G \
                    --nice=10000 \
                    --output=$LOGDIR/slurm_%j.out \
                    --qos=normal \
                    --wrap="python3 ./experiments/whitebox/scripts/detect_gpt.py ${dataset} ${model} ${temp} ${num_tokens}"  
            done
        done
    done
done

# Run NPR/DetectLLM
for model in "Meta-Llama-3.1-8B" "gpt2-xl" "gpt-neo-2.7b" "gpt-j-6B" "opt-2.7b"
do
    for dataset in "xsum" "writing" "squad"
    do
        for temp in 0.8
        do
            for num_tokens in 25 50 75 100
            do
                sbatch -J npr_${model}_${dataset}_${temp}_${num_tokens} \
                    --gres=gpu:1 \
                    --mem=80G \
                    --nice=10000 \
                    --output=$LOGDIR/slurm_%j.out \
                    --qos=normal \
                    --wrap="python3 ./experiments/whitebox/scripts/npr.py ${dataset} ${model} ${temp} ${num_tokens}"  
            done
        done
    done
done

# Run baselines
for model in "Meta-Llama-3.1-8B" "gpt2-xl" "gpt-neo-2.7b" "gpt-j-6B" "opt-2.7b"
do
    for dataset in "xsum" "writing" "squad"
    do
        for temp in 0.8
        do
            for num_tokens in 25 50 75 100
            do
                sbatch -J base_${model}_${dataset}_${temp}_${num_tokens} \
                    --gres=gpu:1 \
                    --mem=80G \
                    --nice=10000 \
                    --output=$LOGDIR/slurm_%j.out \
                    --qos=normal \
                    --wrap="python3 ./experiments/whitebox/scripts/baselines.py ${dataset} ${model} ${temp} ${num_tokens}"  
            done
        done
    done
done

# #!/bin/bash
set -e

LOGDIR="./logs"
mkdir -p $LOGDIR 

# Run TempTest
for gen_model in "Meta-Llama-3.1-8B"    
do
    for score_model in "gpt2-xl" "gpt-neo-2.7b" "gpt-j-6B" "opt-2.7b"
    do    
        for dataset in "xsum" "writing" "squad"
        do
            for gen_temp in 0.8
            do
                for score_temp in 0.5 0.6 0.7 0.8 0.9 1.0
                do
                    sbatch -J tt_${gen_model}_${score_model}_${dataset}_${temp} \
                        --gres=gpu:1 \
                        --mem=80G \
                        --nice=10000 \
                        --output=$LOGDIR/slurm_%j.out \
                        --qos=interactive \
                        --wrap="python3 ./experiments/blackbox/scripts/temptest.py ${dataset} ${gen_model} ${score_model} ${gen_temp} ${score_temp}" 
                done
            done
        done
    done
done


# Run FastDetect
for gen_model in "Meta-Llama-3.1-8B"
do
    for score_model in "gpt2-xl" "gpt-neo-2.7b" "gpt-j-6B" "opt-2.7b"
    do
        for dataset in "xsum" "writing" "squad"
        do
            for temp in 0.8
            do
                for num_tokens in 50 100 150 200 250 300
                do
                    sbatch -J fd_${model}_${dataset}_${temp}_${num_tokens} \
                            --gres=gpu:1 \
                            --mem=80G \
                            --nice=10000 \
                            --output=$LOGDIR/slurm_%j.out \
                            --qos=normal \
                            --wrap="python3 ./experiments/blackbox/scripts/fastdetect_gpt.py ${dataset} ${gen_model} ${score_model} ${temp} ${num_tokens}"  
                done
            done
        done
    done
done

# Run baselines
for gen_model in "Meta-Llama-3.1-8B"
do
    for score_model in "gpt2-xl" "gpt-neo-2.7b" "gpt-j-6B" "opt-2.7b"
    do
        for dataset in "xsum" "writing" "squad"
        do
            for temp in 0.8
            do
                for num_tokens in 50 100 150 200 250 300
                do
                    sbatch -J base_${gen_model}_${dataset}_${temp}_${num_tokens} \
                        --gres=gpu:1 \
                        --mem=80G \
                        --nice=10000 \
                        --output=$LOGDIR/slurm_%j.out \
                        --qos=normal \
                        --wrap="python3 ./experiments/blackbox/scripts/baselines.py ${dataset} ${gen_model} ${score_model} ${temp} ${num_tokens}"  
                done
            done
        done
    done
done

# Run PHD
for gen_model in "Meta-Llama-3.1-8B"
do
    for score_model in "roberta-base" 
    do
        for dataset in "squad" "xsum" "writing" 
        do
            for temp in 0.8
            do
                for num_tokens in 50 100 150
                do
                    sbatch -J phd_${gen_model}_${score_model}_${dataset}_${temp}_${num_tokens} \
                        --gres=gpu:1 \
                        --mem=80G \
                        --nice=10000 \
                        --output=$LOGDIR/slurm_%j.out \
                        --qos=interactive \
                        --wrap="python3 ./experiments/blackbox/scripts/phd.py ${dataset} ${gen_model} ${score_model} ${temp} ${num_tokens}"  
                done
            done
        done
    done
done

# Run DetectGPT
for gen_model in "Meta-Llama-3.1-8B"
do
    for score_model in "gpt2-xl" "gpt-neo-2.7b" "gpt-j-6B" "opt-2.7b"
    do
        for dataset in "xsum" "writing" "squad"
        do
            for temp in 0.8
            do
                for num_tokens in 50 100 150 200 250 300
                do
                    sbatch -J dgpt_${score_model}_${dataset}_${temp}_${num_tokens} \
                        --gres=gpu:1 \
                        --mem=80G \
                        --nice=10000 \
                        --output=$LOGDIR/slurm_%j.out \
                        --qos=normal \
                        --wrap="python3 ./experiments/blackbox/scripts/detect_gpt.py ${dataset} ${gen_model} ${score_model} ${temp} ${num_tokens}"  
                done
            done
        done
    done
done

# Run NPR
for gen_model in "Meta-Llama-3.1-8B"
do
    for score_model in "gpt2-xl" "gpt-neo-2.7b" "gpt-j-6B" "opt-2.7b"
    do
        for dataset in "xsum" "writing" "squad"
        do
            for temp in 0.8
            do
                for num_tokens in 50 100 150 200 250 300
                do
                    sbatch -J npr_${score_model}_${dataset}_${temp}_${num_tokens} \
                        --gres=gpu:1 \
                        --mem=80G \
                        --nice=10000 \
                        --output=$LOGDIR/slurm_%j.out \
                        --qos=normal \
                        --wrap="python3 ./experiments/blackbox/scripts/npr.py ${dataset} ${gen_model} ${score_model} ${temp} ${num_tokens}"  
                done
            done
        done
    done
done

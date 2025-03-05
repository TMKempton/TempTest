# TempTest: Local Normalization Distortion and the Detection of Machine-Generated Text

This repository accompanies the paper:

```bibtex
@inproceedings{kempton2025temptest,
  title={TempTest: Local Normalization Distortion and the Detection of Machine-Generated Text},
  author={Tom Kempton, Stuart Burrell, and Connor Cheverall},
  booktitle={Proceedings of the 28th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year={2025}
}
```

Experiments for the paper were run on a cluster using the Slurm job scheduler, though scripts may easily be adapted to run using your job scheduler of choice.
Using A100 GPUs with 40GB of VRAM most experiments take at most a day to run.
Models were obtained from [HuggingFace](https://huggingface.co/).

## Environment installation

Install the necessary libraries with

```bash
pip install -r requirements.txt
```

We recommend using a virtual environment, using e.g. `conda create -n temptest python=3.10` and `conda activate temptest`.

## Running experiments

Experimental scripts may be found in `/experiments` and shell scripts are provided for running
various configurations from the paper. These may be easily adapted to explore further.

Logic implementing our method may be found in `/src`. 

Code for baseline methods may be easily obtained from [Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt), with
the exception of PHD which was retrieved from [here](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwi6g_a8vI6JAxUrWEEAHfLAJIYQFnoECBMQAQ&url=https%3A%2F%2Fgithub.com%2FArGintum%2FGPTID&usg=AOvVaw0LNRNQC7uWVm-8tth4jkAn&opi=89978449).

## Contributions
Any questions or suggestions please let us know and we'd be happy to help!

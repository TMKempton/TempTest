import json
import sys
from pathlib import Path
from threading import Thread

import numpy as np
import torch
from loguru import logger
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
from transformers import RobertaModel, RobertaTokenizer

from src.hardware import DEVICE

MINIMAL_CLOUD = 47


def prim_tree(adj_matrix, alpha=1.0):
    """
    From: https://github.com/ArGintum/GPTID
    """
    infty = np.max(adj_matrix) + 10

    dst = np.ones(adj_matrix.shape[0]) * infty
    visited = np.zeros(adj_matrix.shape[0], dtype=bool)
    ancestor = -np.ones(adj_matrix.shape[0], dtype=int)

    v, s = 0, 0.0
    for i in range(adj_matrix.shape[0] - 1):
        visited[v] = 1
        ancestor[dst > adj_matrix[v]] = v
        dst = np.minimum(dst, adj_matrix[v])
        dst[visited] = infty

        v = np.argmin(dst)
        s += adj_matrix[v][ancestor[v]] ** alpha

    return s.item()


def process_string(sss):
    """
    From: https://github.com/ArGintum/GPTID
    """
    return sss.replace("\n", " ").replace("  ", " ")


class PHD:
    """
    From: https://github.com/ArGintum/GPTID
    """

    def __init__(
        self, alpha=1.0, metric="euclidean", n_reruns=3, n_points=7, n_points_min=3
    ):
        """
        Initializes the instance of PH-dim computer
        Parameters:
            1) alpha --- real-valued parameter Alpha for computing PH-dim (see the reference paper). Alpha should be chosen lower than
        the ground-truth Intrinsic Dimensionality; however, Alpha=1.0 works just fine for our kind of data.
            2) metric --- String or Callable, distance function for the metric space (see documentation for Scipy.cdist)
            3) n_reruns --- Number of restarts of whole calculations (each restart is made in a separate thread)
            4) n_points --- Number of subsamples to be drawn at each subsample
            5) n_points_min --- Number of subsamples to be drawn at larger subsamples (more than half of the point cloud)
        """
        self.alpha = alpha
        self.n_reruns = n_reruns
        self.n_points = n_points
        self.n_points_min = n_points_min
        self.metric = metric
        self.is_fitted_ = False

    def _sample_W(self, W, nSamples):
        n = W.shape[0]
        random_indices = np.random.choice(n, size=nSamples, replace=False)
        return W[random_indices]

    def _calc_ph_dim_single(self, W, test_n, outp, thread_id):
        lengths = []
        for n in test_n:
            if W.shape[0] <= 2 * n:
                restarts = self.n_points_min
            else:
                restarts = self.n_points

            reruns = np.ones(restarts)
            for i in range(restarts):
                tmp = self._sample_W(W, n)
                reruns[i] = prim_tree(cdist(tmp, tmp, metric=self.metric), self.alpha)

            lengths.append(np.median(reruns))
        lengths = np.array(lengths)

        x = np.log(np.array(list(test_n)))
        y = np.log(lengths)
        N = len(x)
        outp[thread_id] = (N * (x * y).sum() - x.sum() * y.sum()) / (
            N * (x**2).sum() - x.sum() ** 2
        )

    def fit_transform(self, X, y=None, min_points=50, max_points=512, point_jump=40):
        """
        Computing the PH-dim
        Parameters:
        1) X --- point cloud of shape (n_points, n_features),
        2) y --- fictional parameter to fit with Sklearn interface
        3) min_points --- size of minimal subsample to be drawn
        4) max_points --- size of maximal subsample to be drawn
        5) point_jump --- step between subsamples
        """
        ms = np.zeros(self.n_reruns)
        test_n = range(min_points, max_points, point_jump)
        threads = []

        for i in range(self.n_reruns):
            threads.append(
                Thread(target=self._calc_ph_dim_single, args=[X, test_n, ms, i])
            )
            threads[-1].start()

        for i in range(self.n_reruns):
            threads[i].join()

        m = np.mean(ms)
        return 1 / (1 - m)


MIN_SUBSAMPLE = 40
INTERMEDIATE_POINTS = 7


def preprocess_text(text):
    """
    From: https://github.com/ArGintum/GPTID
    """
    return text.replace("\n", " ").replace("  ", " ")


def get_phd_single(text, solver, model, tokenizer, num_tokens):
    """
    Adapted from: https://github.com/ArGintum/GPTID
    """
    inputs = tokenizer(preprocess_text(text), return_tensors="pt").input_ids[
        :, :num_tokens
    ]

    with torch.no_grad():
        outp = model(inputs.to(DEVICE))

    # We omit the first and last tokens (<CLS> and <SEP> because they do not directly correspond to any part of the)
    mx_points = inputs.shape[1] - 2

    mn_points = MIN_SUBSAMPLE
    step = (mx_points - mn_points) // INTERMEDIATE_POINTS

    return solver.fit_transform(
        outp[0][0].cpu().numpy()[1:-1],
        min_points=mn_points,
        max_points=mx_points - step,
        point_jump=step,
    )


if __name__ == "__main__":
    # model_path = "/data/llms/roberta-base"
    # tokenizer_path = model_path

    ### Loading the model

    result_dir = Path(__file__).parent.parent / Path("results")
    result_dir.mkdir(parents=True, exist_ok=True)

    dataset = sys.argv[1]
    gen_model = sys.argv[2]
    score_model = sys.argv[3]
    gen_temp = float(sys.argv[4])
    num_tokens = int(sys.argv[5])

    with open(
        f"./data/generated/{dataset}_{gen_model}_{gen_temp}.completions.json", "r"
    ) as f:
        data = json.load(f)["completions"]

    results = {"real_phd_metric_lists": [], "synthetic_phd_metric_lists": []}

    tokenizer = RobertaTokenizer.from_pretrained("/data/llms/roberta-base")
    model = RobertaModel.from_pretrained("/data/llms/roberta-base").to(DEVICE)
    for text in data:
        real_completion = text["real_completion"]
        synthetic_completion = text["synthetic_completion"]
        logger.info(f"Context: {text['context']}")
        logger.info(f"Real completion: {real_completion}")
        logger.info(f"Synthetic completion: {synthetic_completion}")
        solver = PHD()
        real_phd_metrics = get_phd_single(
            real_completion,
            solver=solver,
            model=model,
            tokenizer=tokenizer,
            num_tokens=num_tokens,
        )
        synthetic_phd_metrics = get_phd_single(
            synthetic_completion,
            solver=solver,
            model=model,
            tokenizer=tokenizer,
            num_tokens=num_tokens,
        )

        results["real_phd_metric_lists"].append(real_phd_metrics)
        results["synthetic_phd_metric_lists"].append(synthetic_phd_metrics)

        auroc = roc_auc_score(
            [1] * len(results["real_phd_metric_lists"])
            + [0] * len(results["synthetic_phd_metric_lists"]),
            results["real_phd_metric_lists"] + results["synthetic_phd_metric_lists"],
        )
        results["auroc"] = auroc

    with open(
        result_dir
        / f"{dataset}_gen{gen_model}_score{score_model}_gen{gen_temp}_scoreNULL_{num_tokens}_phd_results.json",
        "w",
    ) as f:
        f.write(json.dumps(results))

    logger.info("Completed experiment.")

import json
import os
from tqdm import tqdm
import re
import torch
import argparse

from metrics.auroc import AUROC


from detectors.fast_detect_gpt import Fast_Detect_GPT
from detectors.detect_gpt import Detect_GPT
from detectors.logprob import LogProbDetector
from detectors.logrank import LogRankDetector
from detectors.ghostbuster import Ghostbuster


def load_json_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return json.load(file)
    else:
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")


def remove_punctuation(text):
    return re.sub(r"[^\w\s]", "", text)


def load_dataset(args):
    if args.dataset not in ["xsum", "squad", "writing", "abstract"]:
        raise ValueError(
            "Selected Dataset is invalid. Valid choices: 'xsum','squad','writing','abstract'"
        )
    if args.data_generator_llm not in ["gpt-3.5-turbo", "davinci"]:
        raise ValueError(
            "Selected Data Generator LLM is invalid. Valid choices: 'gpt-3.5-turbo', 'davinci'"
        )

    if args.dipper:
        file_name = os.path.join(
            args.dataset_dir, args.dataset, f"{args.dataset}_evasion_dipper.json"
        )
        return load_json_file(file_name)

    else:
        file_name = os.path.join(
            args.dataset_dir,
            args.dataset,
            f"{args.dataset}_{args.data_generator_llm}.raw_data.json",
        )
        if os.path.exists(file_name):
            print(
                f"Dataset {args.dataset} generated with {args.data_generator_llm} loaded successfully!"
            )
            return load_json_file(file_name)
        else:
            raise ValueError(f"Data filepath {file_name} does not exist")

        pass


def get_args():
    parser = argparse.ArgumentParser(
        description="A simple command-line argument example."
    )
    parser.add_argument("--dataset", choices=["squad", "xsum", "abstract"])
    parser.add_argument("--dipper", action="store_true")
    parser.add_argument(
        "--data_generator_llm",
        choices=["gpt-3.5-turbo", "davinci"],
        default="gpt-3.5-turbo",
    )
    parser.add_argument(
        "--detector", choices=["fdgpt", "logrank", "logprob", "dgpt", "ghostbuster"]
    )
    parser.add_argument("--dataset_dir", default="./datasets/")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    data = load_dataset(args)
    print(args)
    print()
    if args.detector == "fdgpt":
        print("Detector Used: FD-GPT")
        detector = Fast_Detect_GPT(
            "gpt2-xl",
            "gpt2-xl",
            "xsum",
            "./results/*sampling_discrepancy.json",
            args.device,
        )
    if args.detector == "dgpt":
        print("Detector Used: D-GPT")
        detector = Detect_GPT(
            "./results/*sampling_discrepancy.json", 0.3, 1.0, 2, 10, "gpt2-xl", "t5-3b"
        )
    if args.detector == "logprob":
        print("Detector Used: LogProb")
        detector = LogProbDetector(device=args.device)
    if args.detector == "logrank":
        print("Detector Used: LogRank")
        detector = LogRankDetector(device=args.device)
    if args.detector == "ghostbuster":
        print("Detector Used: Ghostbuster")
        detector = Ghostbuster()

    results, labels = [], []
    for data_value in tqdm(data["original"]):
        if args.detector in ["fdgpt", "dgpt"]:
            results.append(detector.crit(data_value))
        else:
            results.append(detector.llm_likelihood(data_value))
        torch.cuda.empty_cache()
        labels.append(0)

    if args.dipper:
        for data_value in tqdm(data["evade"]):
            print("Running attack...")
            if args.detector in ["fdgpt", "dgpt"]:
                results.append(detector.crit(data_value))
            else:
                results.append(detector.llm_likelihood(data_value))
            torch.cuda.empty_cache()
            labels.append(1)
    else:
        for data_value in tqdm(data["sampled"]):
            if args.detector in ["fdgpt", "dgpt"]:
                results.append(detector.crit(data_value))
            else:
                results.append(detector.llm_likelihood(data_value))
            torch.cuda.empty_cache()
            labels.append(1)
    auroc = AUROC(results, labels)
    print(f"Evasion AUROC (Y/N): {args.dipper}")
    print(f"Dataset: {args.dataset}")
    print(f"Detector: {args.detector}")
    print(f"AUROC: {auroc}")

    res = {
        "evasion": args.dipper,
        "dataset": args.dataset,
        "detector": args.detector,
        "auroc": auroc,
        "results": results,
        "labels": labels,
    }

    evasion_status = ""
    if args.dipper:
        evasion_status = "_dipper"

    with open(
        os.path.join(
            "./experiments", f"{args.dataset}_{args.detector}{evasion_status}.json"
        ),
        "w",
    ) as output_file:
        json.dump(res, output_file)

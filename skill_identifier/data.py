import json
import os
import sys
import copy
import math
import random
import numpy as np
import yaml

from collections import defaultdict
from datasets import load_dataset, load_from_disk
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

import re
from utils import calculate_metrics, parse_output, parse_rankings, calculate_retrieval_metrics

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def post_processing(output, example):
    prediction = output["output"]
    skills = []
    match = re.search(r"<skill>(.*?)</skill>", prediction, flags=re.DOTALL)
    if match:
        skill_content = match.group(1)
        # Split by comma, strip whitespace
        skills = [skill.strip() for skill in skill_content.split(",") if skill.strip()]
    
    judge_pred = True
    match = re.search(r"<judge>(.*?)</judge>", prediction, flags=re.DOTALL)
    if match:
        judge_content = match.group(1)
        if "incorrect" in judge_content.lower() or "false" in judge_content.lower():
            judge_pred = False


    parsed_pred = {"skills": skills, "judge_pred": judge_pred}
    return {"accuracy": 1.0, "partial_accuracy":  1.0, "extraction_rate": 1.0}, {"parsed_output": parsed_pred, "error_report": None}

def load_classified_data(dataset: str, data_name="gsm8k", subject="geometry", path=None, max_test_samples=None, seed=42,):
    '''
    Load initial inference data to label skills.
    '''
    print("Loading initial inference data to label skills")

    data_file = os.path.join(path, dataset)
    subject_name = " ".join(subject.split("_")).strip()
    if subject_name.lower() == "counting and probability":
        subject_name = "Counting & Probability".lower()

    # subject_file = "_".join(subject.split(" "))
    if data_name.lower() == "math":
        prompt_file = os.path.join(f"skill_label_prompts/skill_label_prompts_{subject}.yaml")
    elif data_name.lower() == "gsm8k":
        prompt_file = os.path.join(f"skill_label_prompts/skill_label_prompts_gsm8k.yaml")
    data = load_dataset("json", data_files=data_file, split="train")

    with open(prompt_file, "r") as f:
        prompt = yaml.safe_load(f)
        user_prompt = prompt['USER_PROMPT']

    data_purged = []

    if "train" in dataset:
        for d in data:
            if False in d["score"] and (data_name.lower() != "math" or d["subject"].lower() == subject_name):
                d["model_sol"] = d["code"][0]
                data_purged.append(d)
    else:
        print(f"DEBUG: labeling test set w/ size {len(data)}")
        for d in data:
            if d["PRM_pred"] == False and (data_name.lower() != "math" or d["subject"].lower() == subject_name):
                d["model_sol"] = d["code"][0]
                data_purged.append(d)

    data_purged = HFDataset.from_list(data_purged)
    if max_test_samples != -1:
        data_purged = data_purged.select(range(min(max_test_samples, len(data_purged))))

    return {
        "data": data_purged,
        "prompt_template": user_prompt,
        "user_template": user_prompt,
        "post_process": post_processing,
    }

def load_data(args, dataset, path=None, demo_path=None):
    if args.label_skill:
        return load_classified_data(dataset, data_name=args.data_name, subject=args.subject, path=path, max_test_samples=args.max_test_samples, seed=args.seed)
    return None


class TestItemDataset(Dataset):
    def __init__(self, data, llm, tokenizer):
        self.data = data
        self.llm = llm
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data["data"])

    def __getitem__(self, idx):
        inputs = self.llm.prepare_inputs(self.data["data"][idx], self.data)
        original_text = None
        if "input_ids" in inputs:
            original_text = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
        return inputs, original_text

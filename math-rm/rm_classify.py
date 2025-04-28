from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from datasets import load_dataset
from accelerate import Accelerator
from collections import Counter
import numpy as np
import torch
from tqdm import tqdm
import argparse
import json
import time
import os
import sys
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_name_or_path", type=str, default='pwork7/llama31_it_prm_2e6_bz32_1epoch_conversation')  # model path
    parser.add_argument("--dataset", type=str, default='RLHFlow/Deepseek-GSM8K-Test')  # data path
    parser.add_argument("--output_dir", type=str, default="math_best_of_n")  # output dir
    parser.add_argument("--pred_thres", type=float, default=0.8)
    parser.add_argument("--pred_thres2", type=float, default=0.7)
    parser.add_argument("--num_n", type=int, default=1024)  # number of N for each question
    parser.add_argument("--num_test_sample", type=int, default=8)  # number of selected test samples
    parser.add_argument("--model_type",type=str,choices=["Mistral","Deepseek"],default='Mistral')
    return parser.parse_args()

def batch_data(data_list, batch_size=8):
    n = batch_size
    batch_data = []
    for i in range(n-1):
        start = i * (len(data_list) // batch_size)
        end = (i+1)* (len(data_list) // batch_size)
        batch_data.append(data_list[start:end])

    last_start = (n-1) * (len(data_list) // batch_size)
    batch_data.append(data_list[last_start:len(data_list)])
    return batch_data

def select_sample(args,sample,model,tokenizer,candidate_tokens,local_rank,pred_thres=0.8,pred_thres2=0.7):
    prompt = sample['question']
    scores_list = []
    #text_list = []
    answers = sample['code'][:args.num_n]
    step_scores = []
    for i, ans in enumerate(answers):
        single_step_score = []
        conversation = []
        forward_conv = []
        if args.model_type == "Mistral":
            ans_list = ans.split("ки\n")
        else:
            ans_list = ans.split("\n\n")
        pred = sample["pred"][i]
        ans_list.append(f"The final answer is: \\boxed{{{pred}}}.")
        ans_list = [j.strip() for j in ans_list]
        for k in range(len(ans_list)):
            if k == 0:
                text = prompt + " " + ans_list[0]
            else:
                text = ans_list[k]
            conversation.append({"content":text,"role":"user"})
            conversation.append({"content":"+","role":"assistant"})

            input_ids = tokenizer.apply_chat_template(conversation,return_tensors="pt").to(local_rank)  
            with torch.no_grad():
                logits = model(input_ids).logits[:,-3,candidate_tokens] #simple version, the +/- is predicted by the '-3' position
                scores = logits.softmax(dim=-1)[:,0] # 0 means the prob of + (1 mean -)
                #print(scores)
                single_step_score.append(scores[0].detach().to('cpu', dtype=torch.float32).item())

        step_scores.append(single_step_score)
        # TODO: incorporate best answer algorithm
        scores_list.append(sum(single_step_score)/len(single_step_score))

    idx = scores_list.index(max(scores_list))

    # Predict if the answer is correct
    best_scores = step_scores[idx]

    # Ours: Judge whether the sample is correct
    sample['PRM_pred'] = True
    if best_scores[-1] <= pred_thres or np.mean(best_scores) <= pred_thres or any(x <= pred_thres2 for x in best_scores[1:]):
        sample['PRM_pred'] = False

    sample['step_scores'] = step_scores
    sample['best_ans_idx'] = idx
    sample['best_ans'] = answers[idx]
    # Find the worse step
    if len(best_scores) <= 1:
        worst_step_idx = 0
    else:
        worst_step_idx = best_scores[:-1].index(min(best_scores[:-1])) # Don't consider the last step
    worst_step = answers[idx].split("\n\n")[worst_step_idx]
    sample['worst_step_idx'] = worst_step_idx
    sample['worst_step'] = worst_step

    # Determine accuracy: F1
    pred_cls = ""
    if sample['score'][idx] == True: # TP or FN
        pred_cls = "TP" if sample['PRM_pred'] else "FN"
    else:
        pred_cls = "FP" if sample['PRM_pred'] else "TN"

    return pred_cls,sample


def worker(args, model, tokenizer, data, local_rank):

    temp_instances = []
    plus_tag_id = tokenizer.encode('+')[-1]
    minus_tag_id = tokenizer.encode('-')[-1]
    candidate_tokens = [plus_tag_id,minus_tag_id]
    for i,sample in enumerate(tqdm(data)):
        sign,new_sample = select_sample(args,sample,model,tokenizer,candidate_tokens,local_rank, args.pred_thres, args.pred_thres2)
        data[i] = new_sample
        temp_instances.append(sign)

    return temp_instances,data

if __name__ == "__main__":
    args = parse_args()

    accelerator = Accelerator()
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    #print(world_size)

    ds = load_dataset("json", data_files={"test": args.dataset},split="test")
    if args.num_test_sample == -1:
        num_sample = len(ds)
    else:
        num_sample = min(args.num_test_sample, len(ds))
    ds = ds.select(range(num_sample))

    local_rank = Accelerator().local_process_index
    print("---------------")
    print(f"begin to load reward model {args.reward_name_or_path}.")
    print("---------------")
    downloaded = False
    while not downloaded:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.reward_name_or_path)
            model = AutoModelForCausalLM.from_pretrained(args.reward_name_or_path, torch_dtype=torch.bfloat16).to(local_rank).eval()
            downloaded = True
        except Exception as error:
            print("An error occurred:", error)
            print("Failed to load the reward model. Retrying....")
            time.sleep(2)

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    data = []
    data_size = len(ds["question"])

    share = int(data_size / world_size) + 1
    ds = ds.select(np.arange(local_rank * share, min((local_rank + 1) * share, len(ds))))
    print(ds)
    for sample in ds:
        data.append(sample)

    # selected_data are prediction classes ['TP', 'TN', 'FP', 'FN']
    # new_data are the enriched samples
    selected_data, new_data = worker(args,model,tokenizer,data,local_rank)

    # Send the data to other GPUs
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    all_process_list = [{}] * world_size

    data_to_send = {
        "data": [[selected_data[i]] for i in range(len(selected_data))],
        "new_data": [[new_data[i]] for i in range(len(new_data))]}
    # with open(f"{args.output_dir}_{args.num_test_sample}_thres{args.pred_thres}_save_data_{local_rank}.jsonl",'w') as f: # We also save a copy of the step score for each local rank
    #     for entry in new_data:
    #         f.write(json.dumps(entry) + "\n")

    import torch.distributed as dist

    # If running on multiple GPUs, initialize the process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    if not dist.is_initialized():
        dist.init_process_group(
            "gloo",
            rank=local_rank,
            world_size=world_size
        )
    dist.all_gather_object(all_process_list, data_to_send)
    gathered_data = []
    gathered_save_data = []

    for i in range(world_size):
        tmp_data = [tmp[0] for tmp in all_process_list[i]["data"]]
        gathered_data.extend(tmp_data)

        tmp_save_data = [tmp[0] for tmp in all_process_list[i]["new_data"]]
        gathered_save_data.extend(tmp_save_data)

    if local_rank == 0:
    #print(len(gathered_data))
        # calculate F1
        counter = Counter(gathered_data)
        num_TP = counter["TP"]
        num_FN = counter["FN"]
        num_FP = counter["FP"]
        num_TN = counter["TN"]

        recall_reverse = None
        if num_TN + num_FP > 0:
            recall_reverse = num_TN / (num_TN + num_FP)
        # precision = num_TP / (num_TP + num_FP)
        # f1 = 2 * precision * recall / (precision + recall)
        f1=0
        false_rec=0
        precision=0
        # false_rec = num_TN / (num_FP + num_TN)
        accuracy = None
        if num_TN + num_TP + num_FN + num_FP > 0:
            accuracy = (num_TN + num_TP) / (num_TN + num_TP + num_FN + num_FP)
        # print(f"F1: {f1}, false_rec: {false_rec}, precision: {precision}")
        # print(f"TP: {num_TP}, FN: {num_FN}, FP: {num_FP}, TN: {num_TN}")
        metrics = {"TP": num_TP, "FN": num_FN, "FP": num_FP, "TN": num_TN}
        # print(f"acc: {sum(gathered_data)/len(gathered_data)}")
        # acc = {"accuracy":sum(gathered_data)/len(gathered_data)}
        os.makedirs(args.output_dir, exist_ok=True)

        with open(f"{args.output_dir}/size{args.num_test_sample}_thres1={args.pred_thres}_thres2={args.pred_thres2}.json",'w') as f:
            json.dump(metrics,f,indent=4,ensure_ascii=False)
        
        with open(f"{args.output_dir}/size{args.num_test_sample}_thres1={args.pred_thres}_thres2={args.pred_thres2}_save_data.jsonl",'w') as f: # We also save a copy of the step score.
            for entry in gathered_save_data:
                f.write(json.dumps(entry) + "\n")

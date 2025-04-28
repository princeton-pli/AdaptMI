import random
import os
import argparse
import time
from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm
import transformers
import sys
from transformers import AutoConfig

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoModel

from evaluate import evaluate
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from parser import *
from trajectory import *
from data_loader import load_data
from python_executor import PythonExecutor
from model_utils import load_hf_lm_and_tokenizer, generate_completions

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--data_path", default=None, type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")

    # Ours
    parser.add_argument("--LLM_judge", action="store_true")
    parser.add_argument("--PRM_judge", action="store_true")
    parser.add_argument("--random_shots", action="store_true")
    parser.add_argument("--llm_sol", action="store_true")
    
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=5)
    parser.add_argument("--num_skill_shots", type=int, default=0)
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )
    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    return args


def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_path, args.data_dir)

    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)
        
    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        # examples = random.sample(examples, min(args.num_test_sample, len(examples)))
        examples = examples[: args.num_test_sample]


    # select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # get out_file name
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-1:])
    out_file_prefix = f"{args.split}_{args.num_test_sample}_{args.num_skill_shots}+{args.num_shots-args.num_skill_shots}shots"

    output_dir = f"{args.output_dir}"
    
    out_file = f"{output_dir}/{out_file_prefix}.jsonl"
    os.makedirs(f"{output_dir}", exist_ok=True)

    # load all processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f
            for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(
                list(load_jsonl(f"{output_dir}/{data_name}/{f}"))
            )

    # dedepulicate
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples, out_file

def load_model(model_name_or_path, args):
    """Load the model from the given path."""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    cls = AutoModelForCausalLM
    if "Yarn" in model_name_or_path:
        # this is a hack... for some reason trust_remote_code does not work with local models
        sys.path.append(model_name_or_path)
        from modeling_llama_together_yarn import LlamaForCausalLM
        cls = LlamaForCausalLM
    

    kwargs = {}
    from pkg_resources import parse_version
    if parse_version(transformers.__version__) <= parse_version("4.34.1"):
        kwargs["use_flash_attention_2"] = True
    else:
        kwargs["attn_implementation"] = "flash_attention_2"
    if "recurrentgemma" in model_name_or_path:
        kwargs = {}

    model = cls.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        **kwargs
    ).eval()
    logger.info(f"loaded model with {sum([p.numel() for p in model.parameters()])} parameters")
    model = torch.compile(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    if args.max_tokens_per_call < tokenizer.model_max_length:
        logger.info(f"setting tokenizer.model_max_length to {args.max_tokens_per_call}")
        tokenizer.model_max_length = args.max_tokens_per_call

    # stop_token_ids = None
    # if args.stop_newline:
    #     stop = list(set(["\n", "Ċ", "ĊĊ", "<0x0A>"]))
    #     stop_token_ids = list(set([tokenizer.convert_tokens_to_ids(stop_token) for stop_token in stop] + [tokenizer.eos_token_id]))
    #     if "llama" in model_name_or_path.lower():
    #         stop_token_ids.remove(tokenizer.unk_token_id)
    #     stop_token_ids = [x for x in stop_token_ids if x is not None]

    # gen_config = GenerationConfig(
    #     max_new_tokens=args.generation_max_length,
    #     min_new_tokens=args.generation_min_length,
    #     do_sample=args.do_sample,
    #     temperature=args.temperature,
    #     top_p=args.top_p,
    #     eos_token_id=stop_token_ids,
    #     pad_token_id=tokenizer.pad_token_id,
    # )
    gen_config = None

    return tokenizer, model, gen_config

def setup(args):
    # load model
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if args.use_vllm:
    #     config = AutoConfig.from_pretrained(args.model_name_or_path)
    #     if hasattr(confisg, "rope_scaling"):
    #         # Extract relevant values; adjust based on your model's config
    #         config.rope_scaling = {
    #             "type": config.rope_scaling.get("rope_type", "linear"),  # Use 'rope_type' or default
    #             "factor": config.rope_scaling["factor"]
    #         }
        llm = LLM(
            model=args.model_name_or_path,
            tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            dtype="bfloat16",
            trust_remote_code=True,
        )
        tokenizer = None
        if args.apply_chat_template:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path, trust_remote_code=True
            )
    else:
        # llm, tokenizer = load_hf_lm_and_tokenizer(
        #     model_name_or_path=args.model_name_or_path,
        #     load_in_half=True,
        #     use_fast_tokenizer=True,
        #     use_safetensors=args.use_safetensors,
        # )
        tokenizer, llm, _ = load_model(args.model_name_or_path, args)

    # infer & eval
    data_list = args.data_names.split(",")
    results = []
    for data_name in data_list:
        results.append(main(llm, tokenizer, data_name, args))

    # add "avg" result to data_list and results
    data_list.append("avg")
    results.append(
        {
            "acc": sum([result["acc"] for result in results]) / len(results),
        }
    )

    # print all results
    pad = max([len(data_name) for data_name in data_list])
    print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))


def is_multi_choice(answer):
    if not answer: 
        return False
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def main(llm, tokenizer, data_name, args):
    examples, processed_samples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    correct_samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]

        # parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        full_prompt = construct_prompt(example, data_name, args)

        if idx == args.start:
            print(full_prompt)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
        }

        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "field",
            "theorem",
            "answer",
            "subject",
            "unique_id",
            "prm_pred",
            "correct_flagged"
        ]:
            if key in example:
                sample[key] = example[key]
        if "skill" in data_name and "step_scores" in example:
            sample["initial_sol"] = example["code"]
            sample["initial_pred"] = example["pred"]
            sample["initial_score"] = example["score"]
            sample["initial_step_scores"] = example["step_scores"]
        
        if args.LLM_judge and "skill" in data_name and example["parsed_output"]["judge_pred"] == True:
            correct_samples.append(sample)
            sample["correct_flagged"] = True
        # elif "missing_skills" in example and args.LLM_judge and "skill" in data_name and example["skill_output"]["judge_pred"] == True:
        #     correct_samples.append(sample)
        elif args.PRM_judge and ("prm_pred" in example and example["prm_pred"] or "PRM_pred" in example and example["PRM_pred"] or "correct_flagged" in sample and sample["correct_flagged"]):
            sample["correct_flagged"] = True
            correct_samples.append(sample)
        else:
            sample["correct_flagged"] = False
            samples.append(sample)
        
    # repeat n times
    input_prompts = [
        sample["prompt"] for sample in samples for _ in range(args.n_sampling)
    ]
    if args.apply_chat_template:
        input_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in input_prompts
        ]
    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []

    max_func_call = 1 if args.prompt_type in ["cot", "pal"] else 4

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

    if args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")
    if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
        stop_words.extend(["Instruction", "Response"])
    elif "jiuzhang" in args.prompt_type:
        stop_words.append("\n\n## Question")
    elif "numina" in args.prompt_type:
        stop_words.append("\n### Problem")
    elif "pure" in args.prompt_type:
        stop_words.append("\n\n\n")

    # start inference
    # measure time use
    start_time = time.time()
    for epoch in range(max_func_call):
        print("-" * 20, "Epoch", epoch)
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        # get all outputs
        prompts = [item[1] for item in current_prompts]
        if args.use_vllm:
            outputs = llm.generate(
                prompts,
                SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens_per_call,
                    n=1,
                    stop=stop_words,
                    stop_token_ids=(
                        [151645, 151643]
                        if "qwen2" in args.model_name_or_path.lower()
                        else None
                    ),
                ),
            )

            outputs = sorted(
                outputs, key=lambda x: int(x.request_id)
            )  # sort outputs by request_id
            outputs = [output.outputs[0].text for output in outputs]
        else:
            outputs = generate_completions(
                model=llm,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_tokens_per_call,
                batch_size=16,
                stop_id_sequences=stop_words,
            )

        assert len(outputs) == len(current_prompts)

        # process all outputs
        remain_prompts = []
        remain_codes = []
        for (i, query), output in zip(current_prompts, outputs):
            output = output.rstrip()
            query += output
            if args.prompt_type == "pal":
                remain_prompts.append((i, query))
                if "```python" in output:
                    output = extract_program(query)
                remain_codes.append(output)
            elif args.prompt_type == "cot":
                end_prompts.append((i, query))
            elif "boxed" not in output and output.endswith("```"):
                program = extract_program(query)
                remain_prompts.append((i, query))
                remain_codes.append(program)
            else:
                end_prompts.append((i, query))

        # execute the remain prompts
        remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            i, query = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            if "pal" in args.prompt_type:
                exec_result = "\\boxed{" + exec_result + "}"
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            # not end
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query)

    # unsolved samples
    print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])

    # remove input_prompt from end_prompt
    codes = []
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        for stop_word in stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)

    # extract preds
    results = [
        run_execute(executor, code, args.prompt_type, data_name) for code in codes
    ]
    time_use = time.time() - start_time

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
        result = results[i * args.n_sampling : (i + 1) * args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        # for j in range(len(preds)):
        #     if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
        #         "A",
        #         "B",
        #         "C",
        #         "D",
        #         "E",
        #     ]:
        #         preds[j] = choice_answer_clean(code[j])
        #     elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
        #         # remove any non-choice char
        #         preds[j] = "".join(
        #             [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
        #         )

        # sample.pop("prompt")
        sample.update({"code": code, "pred": preds, "report": reports})
        all_samples.append(sample)

    # add processed samples
    for sample in correct_samples:
        sample.update({"code": sample["initial_sol"], "pred": sample["initial_pred"], "report": None})
    all_samples.extend(correct_samples)
    all_samples.extend(processed_samples)
    all_samples, result_json = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=args.prompt_type,
        execute=True,
    )

    # save outputs
    if len(processed_samples) < len(all_samples) and args.save_outputs:
        save_jsonl(all_samples, out_file)

    result_json["time_use_in_second"] = time_use
    result_json["time_use_in_minite"] = (
        f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    )

    with open(
        out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w"
    ) as f:
        json.dump(result_json, f, indent=4)
    return result_json


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)

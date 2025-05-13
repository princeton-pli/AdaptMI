#!/bin/bash

# Stage 1-1 inference parameters
MODEL_PATH="../models/Qwen2.5-3B-Instruct"
MODEL_NAME=$(basename "$MODEL_PATH")
NUM_TEST_SAMPLE="1"

SPLIT="test"

DATASET="math"
INFERENCE_OUTPUT_DIR="../outputs/adaptmi+/$DATASET/$MODEL_NAME/stage1_inference"

# Stage 1-2 classification parameters
REWARD_MODEL="RLHFlow/Llama3.1-8B-PRM-Deepseek-Data"

CLASSIFICATION_OUTPUT_DIR="../outputs/adaptmi+/$DATASET/$MODEL_NAME/stage1_classified"
PRED_THRES1="0.9"
PRED_THRES2="0.7"

# Stage 2-1 parameters
LABELER_LLM="gpt-4o-mini"
STAGE21_TEST_FILE="size${NUM_TEST_SAMPLE}_thres1=${PRED_THRES1}_thres2=${PRED_THRES2}_save_data.jsonl"
STAGE21_OUTPUT_DIR="../outputs/adaptmi+/$DATASET/$MODEL_NAME/stage2_labeled"


# Stage 2-2 parameters
STAGE22_OUTPUT_DIR="../outputs/adaptmi+/$DATASET/$MODEL_NAME/stage2_inference"


echo "Stage1-1: Running evaluation on $MODEL_NAME on $DATASET"
conda activate matheval
cd evaluation

TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --model_name_or_path ${MODEL_PATH} \
    --data_name ${DATASET} \
    --output_dir ${INFERENCE_OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type "qwen25-math-cot" \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --num_shots 5 \
    --seed 0 \
    --temperature 0.7 \
    --n_sampling 1 \
    --top_p 1 \
    --use_vllm \
    --start 0 \
    --end -1 \
    --save_outputs \
    --overwrite


echo "Stage1-2: Running classification on $MODEL_NAME on $DATASET"
conda activate stage1
cd ../math-rm

accelerate launch rm_classify.py \
    --reward_name_or_path ${REWARD_MODEL} \
    --dataset ${INFERENCE_OUTPUT_DIR}/${SPLIT}_${NUM_TEST_SAMPLE}_0+5shots.jsonl \
    --output_dir ${CLASSIFICATION_OUTPUT_DIR} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --pred_thres1 ${PRED_THRES1} \
    --pred_thres2 ${PRED_THRES2} \


echo "Stage2-1: Running skill labeler on $MODEL_NAME on $DATASET"
conda activate stage2+
cd ../skill_identifier

python eval.py \
		--datasets ${STAGE21_TEST_FILE} \
		--test_files ${CLASSIFICATION_OUTPUT_DIR} \
        --model_name_or_path ${LABELER_LLM} \
        --max_test_samples -1 \
		--overwrite \
		--data_name ${DATASET} \
		--output_dir ${STAGE21_OUTPUT_DIR} \
		--label_skill


echo "Stage2-2: Running skill-based evaluation on $MODEL_NAME on $DATASET"
conda activate matheval
cd ../evaluation

TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --model_name_or_path ${MODEL_PATH} \
    --data_path ../outputs/adaptmi+/${DATASET}/${MODEL_NAME}/stage2_labeled/${STAGE21_TEST_FILE}_labeler=${LABELER_LLM}.jsonl \
    --data_name ${DATASET}-skill \
    --output_dir ${STAGE22_OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type "qwen25-math-cot" \
    --num_test_sample -1 \
    --num_shots 5 \
    --num_skill_shots 5 \
    --seed 0 \
    --temperature 0.7 \
    --n_sampling 1 \
    --top_p 1 \
    --use_vllm \
    --start 0 \
    --end -1 \
    --save_outputs \
    --overwrite \
    --PRM_judge

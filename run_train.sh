#!/bin/bash

# Gemma3-LISA モデルの学習スクリプト
# 使用方法:
#   ./run_train.sh [GPUS] [MODEL_SIZE] [BATCH_SIZE] [EPOCHS]
#   例: ./run_train.sh 2 4b 2 10

# デフォルト値
NUM_GPUS=${1:-1}      # GPUの数
MODEL_SIZE=${2:-4b}   # モデルサイズ (4b, 7b, 12b, 27b)
BATCH_SIZE=${3:-2}    # バッチサイズ
EPOCHS=${4:-10}       # エポック数
GRAD_ACCUM=${5:-10}   # 勾配集積ステップ
PRECISION=${6:-bf16}  # 精度 (fp32, bf16, fp16)

# モデルサイズごとのバージョン設定
case $MODEL_SIZE in
  4b)
    MODEL_VERSION="google/gemma-3-4b-it"
    ;;
  7b)
    MODEL_VERSION="google/gemma-3-7b-it"
    ;;
  12b)
    MODEL_VERSION="google/gemma-3-12b-it"
    ;;
  27b)
    MODEL_VERSION="google/gemma-3-27b-it"
    ;;
  *)
    echo "不明なモデルサイズ: $MODEL_SIZE"
    echo "4b, 7b, 12b, 27b のいずれかを指定してください"
    exit 1
    ;;
esac

# 実験名の設定
EXP_NAME="lisa-gemma3-${MODEL_SIZE}-bs${BATCH_SIZE}-e${EPOCHS}-ga${GRAD_ACCUM}"

# データセットパスの確認
DATASET_DIR="H:/download/LISA-dataset/dataset"
if [ ! -d "$DATASET_DIR" ]; then
  echo "警告: データセットディレクトリが存在しません: $DATASET_DIR"
  echo "データセットパスを確認してください"
  exit 1
fi

# SAMモデルパスの確認
SAM_PATH="C:/Users/oda/foodlmm-llama/weights/sam_vit_h_4b8939.pth"
if [ ! -f "$SAM_PATH" ]; then
  echo "警告: SAMモデルが見つかりません: $SAM_PATH"
  echo "SAMモデルのパスを確認してください"
  exit 1
fi

# CUDA_VISIBLE_DEVICESの設定
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))

# ログディレクトリの作成
LOG_DIR="./runs/${EXP_NAME}"
mkdir -p $LOG_DIR

# DeepSpeedの起動
python -m torch.distributed.launch \
  --nproc_per_node=$NUM_GPUS \
  train_ds.py \
  --version $MODEL_VERSION \
  --dataset "sem_seg||refer_seg||vqa||reason_seg" \
  --sample_rates "9,3,3,1" \
  --sem_seg_data "ade20k||cocostuff" \
  --refer_seg_data "refclef||refcoco||refcoco+||refcocog" \
  --vqa_data "llava_instruct_150k" \
  --reason_seg_data "ReasonSeg|train" \
  --val_dataset "ReasonSeg|val" \
  --dataset_dir $DATASET_DIR \
  --vision_pretrained $SAM_PATH \
  --log_base_dir "./runs" \
  --exp_name $EXP_NAME \
  --epochs $EPOCHS \
  --steps_per_epoch 500 \
  --batch_size $BATCH_SIZE \
  --grad_accumulation_steps $GRAD_ACCUM \
  --val_batch_size 1 \
  --workers 4 \
  --lr 0.0003 \
  --ce_loss_weight 1.0 \
  --dice_loss_weight 0.5 \
  --bce_loss_weight 2.0 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules "q_proj,k_proj,v_proj" \
  --precision $PRECISION \
  --model_max_length 512 \
  --train_mask_decoder \
  --gradient_checkpointing \
  --use_mm_start_end \
  --auto_resume \
  --conv_type "gemma_v1" \
  2>&1 | tee $LOG_DIR/train.log 
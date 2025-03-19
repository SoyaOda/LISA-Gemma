# import argparse
import os
import sys
import time
from functools import partial

import numpy as np
import torch
import tqdm
import transformers
# from peft import LoraConfig, get_peft_model  # 一時的にコメントアウト
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor

from model.gemma3.model.gemma3_model import GemmaLISAModel  # GemmaLISA.pyから直接インポート
from utils.dataset import HybridDataset, ValDataset, collate_fn
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)


# テスト用の簡略化されたスクリプト
def main():
    print("LISA-Gemma3テストスクリプトを開始します")
    
    # デバイスの確認
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}")
    
    # トークナイザーの初期化
    model_name = "google/gemma-3-4b-it"  # 4Bモデルに変更
    print(f"モデル名: {model_name}")
    
    try:
        # トークナイザーのロード
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, use_fast=False, padding_side="right"
        )
        print("トークナイザーを正常にロードしました")
        
        # 特殊トークンの追加
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.add_tokens("[SEG]")
        seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        print(f"[SEG]トークンのインデックス: {seg_token_idx}")
        
        # 画像トークンの追加
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        print("画像トークンを追加しました")
        
        # SAMの準備
        sam_checkpoint = "C:/Users/oda/foodlmm-llama/weights/sam_vit_h_4b8939.pth"
        
        # モデルの初期化（簡略化）
        print("モデルを初期化しています...")
        model = GemmaLISAModel.from_pretrained(
            model_name,
            sam_checkpoint=sam_checkpoint,
            seg_token_idx=seg_token_idx,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        print("モデルを正常に初期化しました")
        
        # モデルのサイズを確認
        model_size = sum(p.numel() for p in model.parameters())
        print(f"モデルのパラメータ数: {model_size:,}")
        
        print("テストが正常に完了しました")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

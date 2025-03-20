#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gemma3とSAMを統合したLISAモデルのテストスクリプト
"""

import os
import argparse
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoProcessor
from PIL import Image

from model import LISAForCausalLM
from model.segment_anything.utils.transforms import ResizeLongestSide
from model.gemma3.mm_utils import GemmaImageProcessor, get_gemma_processor
from model.segment_anything import sam_model_registry
from utils.utils import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description="LISA-Gemma3モデルテスト")
    parser.add_argument(
        "--version",
        type=str,
        default="google/gemma-3-4b-it",
        help="モデル名",
    )
    parser.add_argument(
        "--model_max_length",
        type=int,
        default=512,
        help="モデルの最大シーケンス長",
    )
    parser.add_argument(
        "--vision_pretrained",
        type=str,
        default="C:/Users/oda/foodlmm-llama/weights/sam_vit_h_4b8939.pth",
        help="SAMの事前学習重み",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="test_images/example.jpg",
        help="テスト画像へのパス",
    )
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='使用するデバイス (cuda/cpu)'
    )
    parser.add_argument(
        "--save_mask",
        action="store_true",
        help="セグメンテーションマスクを保存するかどうか"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="出力ディレクトリ"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="生成する最大トークン数"
    )
    return parser.parse_args()


def visualize_and_save_mask(image, mask, output_path=None):
    """セグメンテーションマスクの可視化と保存"""
    plt.figure(figsize=(10, 10))
    
    # 画像を表示
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")
    
    # マスクを表示
    plt.subplot(1, 3, 2)
    plt.imshow(mask.cpu().numpy(), cmap="gray")
    plt.title("Generated Mask")
    plt.axis("off")
    
    # マスクを重ねた画像を表示
    plt.subplot(1, 3, 3)
    # マスクをRGBA形式に変換
    mask_rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    mask_rgba[..., 0] = 1.0  # R
    mask_rgba[..., 3] = mask.cpu().numpy() * 0.6  # Alpha
    
    # 画像を表示
    plt.imshow(image)
    # マスクを重ねる
    plt.imshow(mask_rgba)
    plt.title("Image with Mask Overlay")
    plt.axis("off")
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"マスク画像を保存しました: {output_path}")
    
    plt.show()


def main():
    args = parse_args()
    
    print(f"Gemma-LISA テストスクリプト")
    print(f"モデル: {args.version}")
    print(f"デバイス: {args.device}")
    
    # デバイスを設定
    device = torch.device(args.device)
    
    # 精度を設定
    if args.precision == "bf16" and torch.cuda.is_available():
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16" and torch.cuda.is_available():
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    try:
        print("トークナイザーとプロセッサを初期化しています...")
        # トークナイザーとプロセッサを初期化
        tokenizer = AutoTokenizer.from_pretrained(
            args.version,
            model_max_length=args.model_max_length,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True,
        )
        tokenizer.pad_token = tokenizer.unk_token
        
        # AutoProcessorを初期化
        processor = AutoProcessor.from_pretrained(
            args.version,
            trust_remote_code=True,
        )
        
        # 特殊トークンの追加
        tokenizer.add_tokens("[SEG]")
        seg_token_idx = tokenizer.convert_tokens_to_ids("[SEG]")
        # [SEG]トークンがない場合、追加する
        if seg_token_idx == tokenizer.unk_token_id:
            tokenizer.add_tokens("[SEG]")
            seg_token_idx = tokenizer.convert_tokens_to_ids("[SEG]")
        print(f"[SEG]トークンのID: {seg_token_idx}")
        
        # 画像トークンの追加
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        
        # モデルの設定
        model_args = {
            "seg_token_idx": seg_token_idx,
            "vision_pretrained": args.vision_pretrained,
            "train_mask_decoder": False,  # 推論時はマスクデコーダは学習しない
            "freeze_maskdecoder": True,
            "freeze_vision_encoder": True,
            "freeze_lm": True,
            "out_dim": 256,  # SAMのプロンプトエンコーダ出力次元
        }
        
        print("モデルを初期化しています...")
        model = LISAForCausalLM.from_pretrained(
            args.version,
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            **model_args
        )
        
        model.to(device)
        print("モデル初期化完了")
        
        # 画像をロード
        if os.path.exists(args.image):
            print(f"画像 {args.image} をロードしています...")
            image = Image.open(args.image).convert('RGB')
            image_np = np.array(image)
            
            # SAM用の画像前処理
            sam_transform = ResizeLongestSide(1024)  # SAMのデフォルトサイズ
            sam_image = sam_transform.apply_image(image_np)
            sam_tensor = torch.from_numpy(sam_image).permute(2, 0, 1).float().to(device)
            
            # 質問の構築
            prompt = "この画像に写っているものを教えてください。また、主要な物体を[SEG]で分割してください。"
            
            # Gemma3のチャットテンプレートを使用して入力を準備
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]}
            ]
            
            print(f"入力プロンプト: {prompt}")
            
            # プロセッサを使用して入力をエンコード
            inputs = processor.apply_chat_template(
                messages, 
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
            
            # 入力をデバイスに転送
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # SAM用の画像を追加
            inputs["images_sam"] = sam_tensor.unsqueeze(0)
            
            print("モデルで推論を実行しています...")
            with torch.no_grad():
                # 生成設定
                generation_kwargs = {
                    "max_new_tokens": args.max_new_tokens,
                    "do_sample": False,
                    "temperature": 0.7,
                    "use_cache": True,
                }
                
                # 1. まず生成して[SEG]トークンを含む出力を取得
                outputs = model.generate(
                    **inputs,
                    **generation_kwargs,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                )
                
                # 生成されたIDをデコード
                output_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)
                print(f"\n生成されたテキスト:\n{output_text}\n")
                
                # [SEG]トークンの位置を確認
                generated_ids = outputs.sequences[0]
                seg_positions = (generated_ids == seg_token_idx).nonzero(as_tuple=True)[0]
                
                if len(seg_positions) > 0:
                    print(f"[SEG]トークンが見つかりました。位置: {seg_positions.tolist()}")
                    
                    # 2. forward関数でセグメンテーションマスクを取得
                    forward_inputs = {
                        "input_ids": generated_ids.unsqueeze(0),
                        "attention_mask": torch.ones_like(generated_ids).unsqueeze(0),
                        "pixel_values": inputs["pixel_values"],
                        "images_sam": inputs["images_sam"],
                        "inference": True,  # 推論モード
                    }
                    
                    # forwardを呼び出してマスクを取得
                    forward_outputs = model(**forward_inputs)
                    
                    # マスク予測を取得
                    mask_predictions = forward_outputs["mask_predictions"]
                    
                    # マスクの可視化
                    if mask_predictions and mask_predictions[0] is not None:
                        print("セグメンテーションマスクを生成しました。")
                        
                        # マスクのロジットをシグモイドして0-1に変換
                        mask = torch.sigmoid(mask_predictions[0][0, 0])
                        
                        # 0.5より大きい値を1、それ以外を0とするバイナリマスクに変換
                        binary_mask = (mask > 0.5).float()
                        
                        # マスクを可視化・保存
                        output_path = None
                        if args.save_mask:
                            output_path = os.path.join(
                                args.output_dir,
                                f"mask_{os.path.basename(args.image)}"
                            )
                        
                        visualize_and_save_mask(image_np, binary_mask, output_path)
                    else:
                        print("セグメンテーションマスクの生成に失敗しました。")
                else:
                    print("[SEG]トークンが生成されませんでした。プロンプトを変更して再試行してください。")
        else:
            print(f"エラー: 画像ファイル {args.image} が見つかりません。")
    
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
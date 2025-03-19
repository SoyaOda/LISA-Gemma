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

from model.GemmaLISA import LISAForCausalLM
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
            # Transformers 4.50.0以降の仕様に合わせてパラメータを追加
            inputs = processor.apply_chat_template(
                messages, 
                add_generation_prompt=True,
                tokenize=True,      # トークン化も一緒に行う
                return_dict=True,   # 辞書形式で返す
                return_tensors="pt",
            )
            
            # 入力をデバイスに転送
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 注: 上記の方法でもエラーが発生する場合、以下の代替手法を試してください
            # (1) 文字列出力を直接トークナイズする方法:
            """
            # まずテンプレートを適用（文字列を取得）
            prompt_text = processor.apply_chat_template(
                messages, 
                add_generation_prompt=True,
                tokenize=False
            )
            
            # 手動でトークナイズして辞書形式の入力を作成
            inputs = tokenizer(prompt_text, return_tensors="pt")
            
            # 画像があれば、pixel_valuesを追加
            if "image" in messages[0]["content"][0]["type"]:
                # 画像を処理
                image = messages[0]["content"][0]["image"]
                pixel_values = processor.image_processor(images=image, return_tensors="pt").pixel_values
                inputs["pixel_values"] = pixel_values
            
            # デバイスに転送
            inputs = {k: v.to(device) for k, v in inputs.items()}
            """
            
            # (2) 最新のプロセッサAPIが利用できない場合、分離して処理する:
            """
            # テキスト部分をトークナイズ
            text = prompt
            text_inputs = tokenizer(text, return_tensors="pt").to(device)
            
            # 画像を処理
            image_processor = processor.image_processor
            image_inputs = image_processor(images=image, return_tensors="pt").to(device)
            
            # モデルに渡す入力を作成
            inputs = {
                "input_ids": text_inputs.input_ids,
                "attention_mask": text_inputs.attention_mask,
                "pixel_values": image_inputs.pixel_values
            }
            """
            
            # SAM用の高解像度画像を準備
            sam_transform = ResizeLongestSide(1024)
            sam_image = sam_transform.apply_image(image_np)
            sam_image_tensor = torch.from_numpy(sam_image).permute(2, 0, 1).float().unsqueeze(0).to(device)
            
            # 画像サイズ情報を保存
            original_size = image_np.shape[:2]
            input_size = sam_transform.get_preprocess_shape(original_size[0], original_size[1], 1024)
            
            # 推論
            print("推論を実行しています...")
            with torch.no_grad():
                # Gemma3 + SAMモデルでの生成
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                )
                
            response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            print("\n--- 応答 ---")
            print(response)
            print("--- 応答終了 ---\n")
            
            # [SEG]トークンの有無を確認し、あればマスクを生成
            seq = outputs.sequences[0]
            seg_positions = (seq == seg_token_idx).nonzero(as_tuple=True)[0]
            
            if len(seg_positions) > 0:
                print("[SEG]トークンが見つかりました。マスクを生成します。")
                # 最後の層の隠れ状態を取得（各ステップごとのタプル）
                # 各ステップは (batch_size, seq_len, hidden_dim)
                # 最後のステップを取得
                hidden_states = outputs.hidden_states[-1]  
                
                # [SEG]トークン位置の埋め込みを取得
                seg_embedding = hidden_states[0, seg_positions[0]]
                
                # SAMの視覚エンコーダで画像埋め込みを取得
                # モデルの視覚エンコーダを使用
                image_embeddings = model.get_image_embeddings(sam_image_tensor)
                
                # [SEG]トークンの埋め込みをSAMの入力次元に射影
                seg_embedding_projected = model.text_hidden_fcs[0](seg_embedding).unsqueeze(0).unsqueeze(1)  # [1, 1, 256]
                
                # プロンプトエンコーダを使用して埋め込みをSAMの入力形式に変換
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=seg_embedding_projected,
                )
                
                # SAMのマスクデコーダを使用してマスクを生成
                low_res_masks, _ = model.mask_decoder(
                    image_embeddings=image_embeddings[0].unsqueeze(0),
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,  # 単一マスクを出力
                )
                
                # マスクを後処理して元の画像サイズに戻す
                masks = model.postprocess_masks(
                    low_res_masks,
                    input_size=input_size,
                    original_size=original_size,
                )
                
                # シグモイド関数を適用してマスクを[0,1]の範囲に変換
                mask = torch.sigmoid(masks[0, 0])
                # 閾値を適用してバイナリマスクに変換
                binary_mask = (mask > 0.5).float()
                
                # マスクを可視化して保存（オプション）
                if args.save_mask:
                    output_path = os.path.join(args.output_dir, f"{os.path.basename(args.image).split('.')[0]}_mask.png")
                    visualize_and_save_mask(image_np, binary_mask, output_path)
                else:
                    visualize_and_save_mask(image_np, binary_mask)
                
            else:
                print("[SEG]トークンが見つかりませんでした。マスクは生成されません。")
            
        else:
            print(f"エラー: 画像 {args.image} が見つかりません")
        
        print("テストが正常に完了しました")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 
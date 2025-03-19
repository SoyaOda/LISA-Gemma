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
from transformers import AutoTokenizer

from model.GemmaLISA import LISAForCausalLM
from model.segment_anything.utils.transforms import ResizeLongestSide
from model.gemma3.mm_utils import GemmaImageProcessor, get_gemma_processor
from model.segment_anything import sam_model_registry


def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description='Gemma LISA モデルのテスト')
    
    parser.add_argument('--model_name', type=str, default='google/gemma-3-4b-it',
                        help='使用するGemma3モデル名（例: google/gemma-3-4b-it）')
    
    parser.add_argument('--sam_checkpoint', type=str, default='C:/Users/oda/foodlmm-llama/weights/sam_vit_h_4b8939.pth',
                        help='SAMモデルのチェックポイントパス')
    
    parser.add_argument('--image_path', type=str, default='test_images/cat.jpg',
                        help='テスト用画像パス')
    
    parser.add_argument('--query', type=str, default='この画像の中の猫をセグメンテーションしてください。',
                        help='画像に対する質問')
    
    parser.add_argument('--output_dir', type=str, default='output',
                        help='出力ディレクトリ')
    
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='使用するデバイス (cuda/cpu)')
    
    parser.add_argument('--precision', type=str, default='fp16',
                        choices=['fp32', 'fp16', 'bf16'],
                        help='精度 (fp32/fp16/bf16)')
    
    parser.add_argument('--load_in_8bit', action='store_true',
                        help='モデルを8bit精度でロード')
    
    parser.add_argument('--load_in_4bit', action='store_true',
                        help='モデルを4bit精度でロード')
    
    parser.add_argument('--sam_only', action='store_true',
                        help='SAMモデルのみをテスト（Gemma3モデルはロードしない）')
    
    return parser.parse_args()


def preprocess_image(image_path, sam_processor, gemma_processor):
    """画像の前処理を行う関数"""
    # 画像を読み込み
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # SAM用の画像前処理
    sam_transform = ResizeLongestSide(1024)
    image_sam = sam_transform.apply_image(image)
    image_size = image_sam.shape[:2]
    
    # ピクセル値を正規化
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    
    image_sam_tensor = torch.from_numpy(image_sam).permute(2, 0, 1).contiguous()
    image_sam_tensor = (image_sam_tensor - pixel_mean) / pixel_std
    
    # パディング
    h, w = image_sam_tensor.shape[-2:]
    padh = 1024 - h
    padw = 1024 - w
    image_sam_tensor = torch.nn.functional.pad(image_sam_tensor, (0, padw, 0, padh))
    
    # Gemma3用の画像前処理（必要な場合）
    image_gemma = gemma_processor(image) if gemma_processor else None
    
    return image_sam_tensor, image_gemma, image_size, image


def test_sam_only(args):
    """SAMモデルのみのテスト"""
    print("=" * 50)
    print("SAMモデルのみのテスト")
    print("=" * 50)
    
    device = torch.device(args.device)
    
    # SAMモデルのロード
    print(f"SAMモデルをロード中: {args.sam_checkpoint}")
    try:
        sam = sam_model_registry["vit_h"](checkpoint=args.sam_checkpoint)
        sam.to(device)
        print("SAMモデルのロードに成功しました")
    except Exception as e:
        print(f"SAMモデルのロード中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 画像の前処理
    try:
        print(f"画像を処理中: {args.image_path}")
        image_tensor, _, image_size, original_image = preprocess_image(
            args.image_path, None, None
        )
        image_tensor = image_tensor.to(device)
    except Exception as e:
        print(f"画像処理中にエラーが発生しました: {e}")
        return
    
    # テキスト埋め込みを生成（実際のモデルを模倣）
    text_embedding = torch.randn(1, 256, device=device)
    
    # SAMでマスク生成
    print("SAMによるマスク生成を実行中...")
    with torch.no_grad():
        # 画像エンコーダで特徴を抽出
        image_embedding = sam.image_encoder(image_tensor.unsqueeze(0))
        
        # プロンプトエンコーダでテキスト埋め込みを処理
        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
            text_embeds=text_embedding.unsqueeze(1),
        )
        
        # マスクデコーダでマスクを生成
        low_res_masks, iou_predictions = sam.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,  # 複数マスクを生成
        )
        
        # マスクの後処理
        masks = sam.postprocess_masks(
            low_res_masks,
            input_size=image_size,
            original_size=original_image.shape[:2],
        )
        
    print("SAMによるマスク生成が成功しました!")
    
    # 最良のマスクを選択
    best_mask_idx = iou_predictions.argmax(dim=1)
    best_mask = masks[0, best_mask_idx[0]].cpu().numpy()
    
    # 結果の可視化
    visualize_results(original_image, best_mask, args.output_dir, prefix="sam_only")


def visualize_results(original_image, mask, output_dir, prefix="result"):
    """結果の可視化"""
    plt.figure(figsize=(15, 5))
    
    # 元画像
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis('off')
    
    # マスク
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis('off')
    
    # マスクを重ねた画像
    plt.subplot(1, 3, 3)
    masked_img = original_image.copy()
    mask_3channel = np.stack([mask > 0.5] * 3, axis=2)
    masked_img = masked_img * 0.7 + np.ones_like(masked_img) * np.array([0, 255, 0]) * 0.3 * mask_3channel
    plt.imshow(masked_img.astype(np.uint8))
    plt.title("Image with Mask")
    plt.axis('off')
    
    # 保存
    plt.savefig(os.path.join(output_dir, f"{prefix}_visualization.png"))
    plt.close()
    
    # マスクをバイナリ画像として保存
    cv2.imwrite(
        os.path.join(output_dir, f"{prefix}_mask.png"),
        (mask > 0.5).astype(np.uint8) * 255
    )


def main():
    """メイン関数"""
    args = parse_args()
    
    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # SAMのみのテストモード
    if args.sam_only:
        test_sam_only(args)
        return
    
    # デバイスとデータ型の設定
    device = torch.device(args.device)
    if args.precision == 'fp32':
        torch_dtype = torch.float32
    elif args.precision == 'fp16':
        torch_dtype = torch.float16
    elif args.precision == 'bf16':
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    
    # トークナイザーの初期化
    print(f"トークナイザーをロード中: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # [SEG]トークンをトークナイザーに追加
    print("[SEG]トークンを追加中...")
    tokenizer.add_tokens("[SEG]", special_tokens=True)
    seg_token_idx = tokenizer.convert_tokens_to_ids("[SEG]")
    
    print(f"[SEG]トークンのID: {seg_token_idx}")
    
    # 画像開始・終了トークンが存在するか確認し、必要に応じて追加
    tokens_to_add = []
    if "<im_start>" not in tokenizer.get_vocab():
        tokens_to_add.append("<im_start>")
    if "<im_end>" not in tokenizer.get_vocab():
        tokens_to_add.append("<im_end>")
    if tokens_to_add:
        tokenizer.add_tokens(tokens_to_add, special_tokens=True)
    
    # Gemma LISA モデルのロード
    print(f"モデルをロード中: {args.model_name}")
    
    model = LISAForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        vision_pretrained=args.sam_checkpoint,
        seg_token_idx=seg_token_idx,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=True,
    )
    
    # 必要に応じて埋め込み層のサイズを調整
    if model.get_input_embeddings().weight.shape[0] < len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    
    # モデルをデバイスに移動
    model.to(device)
    
    # 画像プロセッサの初期化
    print("画像プロセッサを初期化中...")
    gemma_processor = GemmaImageProcessor(get_gemma_processor(args.model_name))
    
    # 画像の前処理
    print(f"画像を処理中: {args.image_path}")
    image_sam, image_gemma, image_size, original_image = preprocess_image(
        args.image_path, None, gemma_processor
    )
    
    # プロンプトの構築
    system_prompt = "You are a helpful visual assistant that can segment objects in images."
    prompt = f"System: {system_prompt}\n\nUser: <im_start><image><im_end>{args.query}\n\nAssistant:"
    
    # トークン化
    print("入力をトークン化中...")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    # 評価モードに設定
    model.eval()
    
    # 推論実行
    print("推論実行中...")
    with torch.no_grad():
        output_ids, texts, pred_masks = model.evaluate(
            pixel_values=image_gemma.unsqueeze(0).to(device),
            images=image_sam.unsqueeze(0).to(device),
            input_ids=input_ids,
            resize_list=[image_size],
            original_size_list=[original_image.shape[:2]],
            max_new_tokens=256,
            tokenizer=tokenizer,
        )
    
    # 結果の表示
    print("出力テキスト:", texts[0])
    
    # マスクの可視化と保存
    if len(pred_masks) > 0:
        print("マスクを可視化中...")
        mask = pred_masks[0].cpu().numpy()
        visualize_results(original_image, mask[0], args.output_dir)
    else:
        print("マスクが生成されませんでした")
    
    print(f"結果は {args.output_dir} に保存されました")


if __name__ == "__main__":
    main() 
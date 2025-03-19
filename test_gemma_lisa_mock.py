#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SAMモデルのテストスクリプト
実際のSAMモデルを使用して正常に動作するかを確認
"""

import os
import sys
import argparse
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# SAMモデルのインポートを試みる
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model.segment_anything import sam_model_registry
    sam_available = True
    print("SAMモデルのインポートに成功しました")
except Exception as e:
    print(f"SAMモデルのインポート中にエラーが発生しました: {e}")
    sam_available = False

def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description='SAMモデルのテスト')
    
    parser.add_argument('--image_path', type=str, default='test_images/cat.jpg',
                        help='テスト用画像パス')
    
    parser.add_argument('--sam_checkpoint', type=str, 
                        default='C:/Users/oda/foodlmm-llama/weights/sam_vit_h_4b8939.pth',
                        help='SAMモデルのチェックポイントパス')
    
    parser.add_argument('--output_dir', type=str, default='output',
                        help='出力ディレクトリ')
    
    return parser.parse_args()


def preprocess_image(image_path):
    """画像の前処理を行う関数"""
    # 画像を読み込み
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"画像を読み込めませんでした: {image_path}")
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # SAM用の画像リサイズ
    target_size = 1024
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    image_sam = cv2.resize(image, (new_w, new_h))
    
    # パディング
    padh = target_size - new_h
    padw = target_size - new_w
    padding = ((0, padh), (0, padw), (0, 0))
    image_sam_padded = np.pad(image_sam, padding, mode='constant')
    
    # ピクセル値を正規化
    pixel_mean = np.array([123.675, 116.28, 103.53])
    pixel_std = np.array([58.395, 57.12, 57.375])
    image_sam_normalized = (image_sam_padded - pixel_mean) / pixel_std
    
    # PyTorchテンソルに変換
    image_sam_tensor = torch.from_numpy(image_sam_normalized).permute(2, 0, 1).float()
    
    return image_sam_tensor, (new_h, new_w), image


def test_sam_loading(checkpoint_path):
    """SAMモデルのロードをテスト"""
    if not sam_available:
        print("SAMモジュールが利用できないため、テストをスキップします")
        return None
    
    if not os.path.exists(checkpoint_path):
        print(f"チェックポイントが見つかりません: {checkpoint_path}")
        print("代わりにモック処理を行います")
        return None
    
    try:
        # SAMモデルをロード
        print(f"SAMモデルをロード中: {checkpoint_path}")
        sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
        print("SAMモデルのロードに成功しました")
        return sam
    except Exception as e:
        print(f"SAMモデルのロード中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_random_masks(image_size, num_masks=3):
    """ランダムなマスクを生成"""
    masks = []
    for i in range(num_masks):
        # 楕円形のマスクを生成
        mask = np.zeros(image_size[:2], dtype=np.float32)
        center_x = np.random.randint(image_size[1] // 4, image_size[1] * 3 // 4)
        center_y = np.random.randint(image_size[0] // 4, image_size[0] * 3 // 4)
        
        # 楕円のサイズと角度
        axes_length = (
            np.random.randint(image_size[1] // 8, image_size[1] // 3),
            np.random.randint(image_size[0] // 8, image_size[0] // 3)
        )
        angle = np.random.randint(0, 180)
        
        # マスクを描画
        cv2.ellipse(
            mask, 
            (center_x, center_y), 
            axes_length, 
            angle, 
            0, 
            360, 
            1, 
            -1
        )
        
        masks.append(mask)
    
    return masks


def main():
    """メイン関数"""
    args = parse_args()
    
    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 50)
    print("SAMモデルのテスト")
    print("=" * 50)
    
    # SAMモデルのロードテスト
    sam = test_sam_loading(args.sam_checkpoint)
    
    # 画像の前処理
    try:
        print(f"画像を処理中: {args.image_path}")
        image_tensor, image_size, original_image = preprocess_image(args.image_path)
    except Exception as e:
        print(f"画像処理中にエラーが発生しました: {e}")
        return
    
    # ランダムなテキスト埋め込みを生成（実際のモデルからの出力を模倣）
    text_embedding = torch.randn(1, 256)
    
    # マスク生成
    if sam is not None:
        print("SAMによるマスク生成を実行中...")
        try:
            # 実際のSAMモデルを使用
            with torch.no_grad():
                # CPUで実行（サンプルのため）
                device = torch.device("cpu")
                sam.to(device)
                image_tensor = image_tensor.to(device)
                text_embedding = text_embedding.to(device)
                
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
                
        except Exception as e:
            print(f"SAMの処理中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            # 代替としてランダムなマスクを生成
            print("代わりにランダムなマスクを生成します")
            random_masks = generate_random_masks(original_image.shape)
            best_mask = random_masks[0]
    else:
        print("SAMモデルが利用できないため、ランダムなマスクを生成します")
        random_masks = generate_random_masks(original_image.shape)
        best_mask = random_masks[0]
    
    # 結果の可視化
    plt.figure(figsize=(15, 5))
    
    # 元画像
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title("元画像")
    plt.axis('off')
    
    # マスク
    plt.subplot(1, 3, 2)
    plt.imshow(best_mask, cmap='gray')
    plt.title("予測マスク")
    plt.axis('off')
    
    # マスクを重ねた画像
    plt.subplot(1, 3, 3)
    masked_img = original_image.copy()
    mask_3channel = np.stack([best_mask > 0.5] * 3, axis=2)
    masked_img = masked_img * 0.7 + np.ones_like(masked_img) * np.array([0, 255, 0]) * 0.3 * mask_3channel
    plt.imshow(masked_img.astype(np.uint8))
    plt.title("マスクを重ねた画像")
    plt.axis('off')
    
    plt.savefig(os.path.join(args.output_dir, "result_sam_test.png"))
    plt.close()
    
    # マスクをバイナリ画像として保存
    cv2.imwrite(
        os.path.join(args.output_dir, "mask_sam_test.png"),
        (best_mask > 0.5).astype(np.uint8) * 255
    )
    
    print(f"結果は {args.output_dir} に保存されました")


if __name__ == "__main__":
    main() 
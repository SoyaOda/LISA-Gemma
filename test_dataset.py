#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gemma3+SAM用データセットのテストスクリプト
データのロード、前処理、バッチ処理までの流れを確認します
"""

import os
import sys
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

from utils.dataset import HybridDataset, collate_fn
from utils.vqa_dataset import VQADataset
from utils.sem_seg_dataset import SemSegDataset
from utils.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description='データセットのテスト')
    
    parser.add_argument('--model_name', type=str, default='google/gemma-3-4b-it',
                        help='使用するGemma3モデル名（例: google/gemma-3-4b-it）')
    
    parser.add_argument('--image_size', type=int, default=224, help='画像サイズ')
    
    parser.add_argument('--precision', type=str, default='fp32', help='精度 (fp32, fp16, bf16)')
    
    parser.add_argument('--dataset_dir', type=str, default='H:/download/LISA-dataset/dataset',
                        help='データセットディレクトリ')
    
    parser.add_argument('--dataset_type', type=str, default='vqa',
                        choices=['vqa', 'sem_seg', 'hybrid'],
                        help='テストするデータセットタイプ')
    
    parser.add_argument('--batch_size', type=int, default=2,
                        help='バッチサイズ')
    
    parser.add_argument('--num_samples', type=int, default=5,
                        help='テストするサンプル数')
    
    parser.add_argument('--output_dir', type=str, default='output/dataset_test',
                        help='出力ディレクトリ')
    
    return parser.parse_args()


def visualize_sample(sample, index, output_dir):
    """サンプルの可視化"""
    os.makedirs(output_dir, exist_ok=True)
    
    # サンプルからデータを取得
    image_path, image_sam, image_gemma, conversation, mask, label, resize, questions, sampled_classes, inference = sample
    
    # 会話の選択（複数ある場合は最初のものを使用）
    if isinstance(conversation, list):
        conversation = conversation[0] if conversation else "会話なし"
    
    # フィギュアの作成
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # SAM用画像の可視化
    try:
        if image_sam is not None:
            pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
            pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
            img_sam = image_sam * pixel_std + pixel_mean
            img_sam = img_sam.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            axes[0].imshow(img_sam)
            axes[0].set_title('SAM用画像')
        else:
            axes[0].text(0.5, 0.5, 'SAM画像なし', horizontalalignment='center', verticalalignment='center')
            axes[0].set_title('SAM画像なし')
    except Exception as e:
        print(f"SAM画像の表示エラー: {e}")
        axes[0].text(0.5, 0.5, f'画像エラー: {type(e).__name__}', horizontalalignment='center', verticalalignment='center')
    axes[0].axis('off')
    
    # Gemma3用画像の可視化
    try:
        # image_gemmaがNoneでないことを確認
        if image_gemma is not None and isinstance(image_gemma, torch.Tensor) and image_gemma.dim() >= 3:
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1)
            img_gemma = image_gemma * std + mean
            img_gemma = img_gemma.permute(1, 2, 0).cpu().numpy()
            img_gemma = np.clip(img_gemma, 0, 1) * 255
            img_gemma = img_gemma.astype(np.uint8)
            axes[1].imshow(img_gemma)
            axes[1].set_title('Gemma3用画像')
        else:
            axes[1].text(0.5, 0.5, 'Gemma画像なし', horizontalalignment='center', verticalalignment='center')
            axes[1].set_title('Gemma画像なし')
    except Exception as e:
        print(f"Gemma画像の表示エラー: {e}")
        axes[1].text(0.5, 0.5, f'画像エラー: {type(e).__name__}', horizontalalignment='center', verticalalignment='center')
    axes[1].axis('off')
    
    # マスクの可視化
    try:
        if mask is not None and isinstance(mask, torch.Tensor) and mask.numel() > 0 and mask.sum() > 0:
            if len(mask.shape) > 2:  # バッチ次元がある場合
                axes[2].imshow(mask[0].cpu().numpy(), cmap='gray')
            else:
                axes[2].imshow(mask.cpu().numpy(), cmap='gray')
            axes[2].set_title('マスク')
        else:
            axes[2].text(0.5, 0.5, 'マスクなし', horizontalalignment='center', verticalalignment='center')
            axes[2].set_title('マスクなし')
    except Exception as e:
        print(f"マスクの表示エラー: {e}")
        axes[2].text(0.5, 0.5, f'マスクエラー: {type(e).__name__}', horizontalalignment='center', verticalalignment='center')
    axes[2].axis('off')
    
    # 画像を保存
    try:
        plt.savefig(os.path.join(output_dir, f'sample_{index}_images.png'))
    except Exception as e:
        print(f"画像保存エラー: {e}")
    plt.close()
    
    # 会話文を保存
    try:
        with open(os.path.join(output_dir, f'sample_{index}_conversation.txt'), 'w', encoding='utf-8') as f:
            f.write(f'画像パス: {image_path}\n\n')
            f.write(f'会話:\n{conversation}')
    except Exception as e:
        print(f"会話保存エラー: {e}")


def visualize_batch(batch, tokenizer, output_dir):
    """バッチの可視化"""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # バッチからデータを取得
        image_paths = batch.get("image_paths", ["パスなし"])
        images = batch.get("images", None)  # SAM用高解像度画像
        pixel_values = batch.get("pixel_values", None)  # Gemma3用画像
        input_ids = batch.get("input_ids", None)
        labels = batch.get("labels", None)
        masks_list = batch.get("masks_list", [])
        conversation_list = batch.get("conversation_list", [])
        
        # バッチ情報をテキストファイルに保存
        with open(os.path.join(output_dir, 'batch_info.txt'), 'w', encoding='utf-8') as f:
            f.write(f'バッチサイズ: {len(image_paths)}\n')
            
            if images is not None:
                f.write(f'SAM用画像テンソル形状: {images.shape}\n')
            else:
                f.write('SAM用画像テンソルなし\n')
                
            if pixel_values is not None:
                f.write(f'Gemma3用画像テンソル形状: {pixel_values.shape}\n')
            else:
                f.write('Gemma3用画像テンソルなし\n')
                
            if input_ids is not None:
                f.write(f'入力ID形状: {input_ids.shape}\n')
            else:
                f.write('入力IDなし\n')
                
            if labels is not None:
                f.write(f'ラベル形状: {labels.shape}\n')
            else:
                f.write('ラベルなし\n')
                
            f.write(f'マスクリスト長: {len(masks_list)}\n\n')
            
            # 各サンプルの情報
            for i in range(min(len(image_paths), 2)):  # 最初の2サンプルのみ情報表示
                f.write(f'サンプル {i+1}:\n')
                f.write(f'  画像パス: {image_paths[i]}\n')
                
                if i < len(masks_list) and masks_list[i] is not None:
                    f.write(f'  マスク形状: {masks_list[i].shape}\n')
                else:
                    f.write('  マスクなし\n')
                
                # トークンIDをデコード
                if input_ids is not None and i < len(input_ids):
                    try:
                        input_text = tokenizer.decode(input_ids[i], skip_special_tokens=False)
                        f.write(f'  入力テキスト: {input_text}\n\n')
                    except Exception as e:
                        f.write(f'  入力テキストのデコードエラー: {e}\n\n')
                
                # 元の会話も保存
                if i < len(conversation_list):
                    f.write(f'  元の会話:\n{conversation_list[i]}\n\n')
        
        # サンプル画像の可視化
        for i in range(min(len(image_paths), 2)):  # 最初の2サンプルのみ可視化
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # SAM用画像の可視化
            if images is not None and i < len(images):
                try:
                    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
                    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
                    img_sam = images[i] * pixel_std + pixel_mean
                    img_sam = img_sam.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    axes[0].imshow(img_sam)
                except Exception as e:
                    print(f"サンプル{i}のSAM画像表示エラー: {e}")
                    axes[0].text(0.5, 0.5, '画像エラー', horizontalalignment='center', verticalalignment='center')
            else:
                axes[0].text(0.5, 0.5, 'SAM画像なし', horizontalalignment='center', verticalalignment='center')
            axes[0].set_title('SAM用画像')
            axes[0].axis('off')
            
            # Gemma3用画像の可視化
            if pixel_values is not None and i < len(pixel_values):
                try:
                    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1)
                    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1)
                    img_gemma = pixel_values[i] * std + mean
                    img_gemma = img_gemma.permute(1, 2, 0).cpu().numpy()
                    img_gemma = np.clip(img_gemma, 0, 1) * 255
                    img_gemma = img_gemma.astype(np.uint8)
                    axes[1].imshow(img_gemma)
                except Exception as e:
                    print(f"サンプル{i}のGemma画像表示エラー: {e}")
                    axes[1].text(0.5, 0.5, '画像エラー', horizontalalignment='center', verticalalignment='center')
            else:
                axes[1].text(0.5, 0.5, 'Gemma画像なし', horizontalalignment='center', verticalalignment='center')
            axes[1].set_title('Gemma3用画像')
            axes[1].axis('off')
            
            # マスクの可視化
            if i < len(masks_list) and masks_list[i] is not None:
                try:
                    if masks_list[i].sum() > 0:
                        if len(masks_list[i].shape) > 2:  # バッチ次元がある場合
                            axes[2].imshow(masks_list[i][0].cpu().numpy(), cmap='gray')
                        else:
                            axes[2].imshow(masks_list[i].cpu().numpy(), cmap='gray')
                        axes[2].set_title('マスク')
                    else:
                        axes[2].text(0.5, 0.5, 'マスクなし', horizontalalignment='center', verticalalignment='center')
                        axes[2].set_title('マスクなし')
                except Exception as e:
                    print(f"サンプル{i}のマスク表示エラー: {e}")
                    axes[2].text(0.5, 0.5, 'マスクエラー', horizontalalignment='center', verticalalignment='center')
            else:
                axes[2].text(0.5, 0.5, 'マスクなし', horizontalalignment='center', verticalalignment='center')
                axes[2].set_title('マスクなし')
            axes[2].axis('off')
            
            plt.savefig(os.path.join(output_dir, f'batch_sample_{i}.png'))
            plt.close()
    
    except Exception as e:
        print(f"バッチ可視化中にエラーが発生: {e}")
        # エラー情報を記録
        with open(os.path.join(output_dir, 'visualization_error.txt'), 'w', encoding='utf-8') as f:
            f.write(f'バッチ可視化エラー: {e}\n')
            f.write(f'バッチの内容: {str(batch.keys()) if hasattr(batch, "keys") else str(type(batch))}\n')


def main():
    """メイン関数"""
    args = parse_args()
    
    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"データセットテスト: {args.dataset_type}")
    print(f"データセットディレクトリ: {args.dataset_dir}")
    print(f"モデル名: {args.model_name}")
    
    # トークナイザーの初期化
    print("トークナイザーの初期化...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # 特殊トークンの追加
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[SEG]")
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    print(f"[SEG]トークンのID: {seg_token_idx}")
    
    # 画像トークンの追加
    tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    
    # データセットの初期化
    print("データセットの初期化...")
    
    try:
        if args.dataset_type == "vqa":
            dataset = VQADataset(
                base_image_dir=args.dataset_dir,
                tokenizer=tokenizer,
                model_name=args.model_name,
                precision=args.precision,
                image_size=args.image_size,
                samples_per_epoch=args.num_samples,
            )
        elif args.dataset_type == "sem_seg":
            dataset = SemSegDataset(
                base_image_dir=args.dataset_dir,
                tokenizer=tokenizer,
                model_name=args.model_name,
                precision=args.precision,
                image_size=args.image_size,
                samples_per_epoch=args.num_samples,
                sem_seg_data="ade20k||cocostuff",  # Mapillaryを除外
            )
        elif args.dataset_type == "hybrid":
            dataset = HybridDataset(
                base_image_dir=args.dataset_dir,
                tokenizer=tokenizer,
                model_name=args.model_name,
                precision=args.precision,
                image_size=args.image_size,
                samples_per_epoch=args.num_samples,
            )
        else:
            raise ValueError(f"不明なデータセットタイプ: {args.dataset_type}")
        
        print(f"データセット長: {len(dataset)}")
        
        # 個別サンプルのテスト
        print("個別サンプルのテスト...")
        for i in range(min(args.num_samples, 3)):  # 最初の3サンプルのみ可視化
            try:
                sample = dataset[i]
                visualize_sample(sample, i, os.path.join(args.output_dir, "samples"))
                print(f"サンプル {i} の処理に成功しました")
            except Exception as e:
                print(f"サンプル {i} の処理中にエラーが発生: {e}")
                # エラー情報をファイルに保存
                with open(os.path.join(args.output_dir, f"sample_{i}_error.txt"), "w", encoding="utf-8") as f:
                    f.write(f"サンプル {i} 処理エラー: {e}\n")
                    import traceback
                    f.write(traceback.format_exc())
        
        # データローダーの作成
        print("データローダーの作成...")
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, conv_type="gemma_v1", use_mm_start_end=True),
        )
        
        # バッチ処理のテスト
        print("バッチ処理のテスト...")
        try:
            batch = next(iter(dataloader))
            visualize_batch(batch, tokenizer, os.path.join(args.output_dir, "batch"))
            print("バッチ処理に成功しました")
        except Exception as e:
            print(f"バッチ処理中にエラーが発生: {e}")
            # エラー情報をファイルに保存
            with open(os.path.join(args.output_dir, "batch_error.txt"), "w", encoding="utf-8") as f:
                f.write(f"バッチ処理エラー: {e}\n")
                import traceback
                f.write(traceback.format_exc())
        
        print("データセットテスト完了")
    
    except Exception as e:
        print(f"データセット初期化/テスト中にエラーが発生: {e}")
        # エラー情報をファイルに保存
        with open(os.path.join(args.output_dir, "dataset_error.txt"), "w", encoding="utf-8") as f:
            f.write(f"データセットエラー: {e}\n")
            import traceback
            f.write(traceback.format_exc())


if __name__ == "__main__":
    main() 
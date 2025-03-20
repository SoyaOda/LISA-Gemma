"""
GemmaLISA モデルのユーティリティ関数
"""

import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from typing import List, Union, Optional, Tuple, Dict

from model.segment_anything.utils.transforms import ResizeLongestSide


def prepare_images_for_sam(
    images: Union[torch.Tensor, List[torch.Tensor], List[np.ndarray]],
    sam_img_size: int = 1024,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    画像をSAMモデル用に処理する関数
    
    Args:
        images: 処理する画像（テンソル、テンソルのリスト、またはnumpy配列のリスト）
        sam_img_size: SAMのモデル入力サイズ
        device: 結果を配置するデバイス
        
    Returns:
        torch.Tensor: SAM用に処理された画像バッチ
    """
    sam_transform = ResizeLongestSide(sam_img_size)
    images_sam = []

    # バッチ処理：imagesの次元を確認
    if torch.is_tensor(images):
        # テンソルの場合、次元数を確認
        if images.dim() == 4:  # [B, C, H, W]または[B, H, W, C]
            # torchvisionの期待するフォーマットに変換
            if images.shape[1] == 3 or images.shape[1] == 1:  # [B, C, H, W]形式
                # チャネル次元を最後に移動: [B, C, H, W] -> [B, H, W, C]
                images_np = images.permute(0, 2, 3, 1).cpu().numpy()
            else:  # [B, H, W, C]形式と仮定
                images_np = images.cpu().numpy()
            
            # 各バッチ要素を個別に処理
            for i in range(images_np.shape[0]):
                image_np = images_np[i]  # [H, W, C]
                transformed_image = sam_transform.apply_image(image_np)
                images_sam.append(transformed_image)
        elif images.dim() == 3:  # 単一画像 [C, H, W]または[H, W, C]
            if images.shape[0] == 3 or images.shape[0] == 1:  # [C, H, W]形式
                image_np = images.permute(1, 2, 0).cpu().numpy()
            else:  # [H, W, C]形式と仮定
                image_np = images.cpu().numpy()
            transformed_image = sam_transform.apply_image(image_np)
            images_sam.append(transformed_image)
        else:
            raise ValueError(f"サポートされていない画像次元です: {images.dim()}。"
                           "3次元または4次元のテンソルを期待しています。")
    elif isinstance(images, list):
        # リスト形式の場合、各要素を処理
        for img in images:
            if torch.is_tensor(img):
                if img.dim() == 3:  # [C, H, W]または[H, W, C]
                    if img.shape[0] == 3 or img.shape[0] == 1:  # [C, H, W]形式
                        img_np = img.permute(1, 2, 0).cpu().numpy()
                    else:  # [H, W, C]形式と仮定
                        img_np = img.cpu().numpy()
                elif img.dim() == 2:  # [H, W]
                    # グレースケール画像を3チャンネルに拡張
                    img_np = img.cpu().numpy()[..., None].repeat(3, axis=2)
                else:
                    raise ValueError(f"サポートされていない画像次元です: {img.dim()}。"
                                   "2次元または3次元のテンソルを期待しています。")
            else:
                # すでにnumpy配列と仮定
                img_np = img
                # 2次元配列（グレースケール）の場合は3チャンネルに拡張
                if img_np.ndim == 2:
                    img_np = img_np[..., None].repeat(3, axis=2)
            
            transformed_image = sam_transform.apply_image(img_np)
            images_sam.append(transformed_image)
    else:
        raise ValueError("imagesはテンソルまたはテンソルのリストである必要があります。")
    
    # 処理された画像をスタックしてテンソルに変換
    images_sam = torch.stack([torch.from_numpy(image).permute(2, 0, 1).float() for image in images_sam])
    
    # デバイスに移動（指定があれば）
    if device is not None:
        images_sam = images_sam.to(device)
    
    return images_sam


def prepare_images_for_gemma(
    images: Union[torch.Tensor, List[torch.Tensor], List[np.ndarray], List[Image.Image]],
    processor=None,
    gemma_img_size: int = 896,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    画像をGemma3モデル用に処理する関数
    
    Args:
        images: 処理する画像（テンソル、テンソルのリスト、numpy配列のリスト、またはPIL画像のリスト）
        processor: Transformersのプロセッサ（指定された場合、そちらを優先使用）
        gemma_img_size: Gemma3の入力画像サイズ
        device: 結果を配置するデバイス
        
    Returns:
        torch.Tensor: Gemma3用に処理された画像バッチ
    """
    # プロセッサが指定されている場合、それを使用
    if processor is not None:
        try:
            # リスト形式に正規化
            image_list = images if isinstance(images, list) else [images]
            
            # PILへの変換が必要なら変換
            pil_images = []
            for img in image_list:
                if isinstance(img, Image.Image):
                    pil_images.append(img)
                elif torch.is_tensor(img):
                    # テンソルをPIL画像に変換
                    if img.dim() == 3 and (img.shape[0] == 3 or img.shape[0] == 1):
                        # [C,H,W] -> [H,W,C]
                        img_np = img.permute(1, 2, 0).cpu().numpy()
                    elif img.dim() == 3:
                        # [H,W,C]
                        img_np = img.cpu().numpy()
                    else:
                        raise ValueError(f"サポートされていない画像次元です: {img.dim()}")
                    
                    # 値範囲を0-255に正規化
                    if img_np.max() <= 1.0:
                        img_np = (img_np * 255).astype(np.uint8)
                    else:
                        img_np = img_np.astype(np.uint8)
                    
                    pil_images.append(Image.fromarray(img_np))
                elif isinstance(img, np.ndarray):
                    # 値範囲を0-255に正規化
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                    
                    pil_images.append(Image.fromarray(img))
            
            # プロセッサでバッチ処理
            processed = processor(images=pil_images, return_tensors="pt")
            pixel_values = processed.pixel_values
            
            # デバイスに移動（指定があれば）
            if device is not None:
                pixel_values = pixel_values.to(device)
            
            return pixel_values
            
        except Exception as e:
            print(f"プロセッサでの処理に失敗しました: {e}")
            print("代替の画像処理方法を使用します")
    
    # プロセッサが使用できない場合の代替手段
    pixel_values_list = []
    
    # テンソルの場合
    if torch.is_tensor(images):
        if images.dim() == 4:  # [B, C, H, W]
            # バッチ内の各画像を処理
            for i in range(images.shape[0]):
                img = images[i]
                # リサイズして正規化
                img = F.interpolate(
                    img.unsqueeze(0), 
                    size=(gemma_img_size, gemma_img_size), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                
                # 標準的な正規化値を適用
                mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1)
                std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1)
                img = (img - mean) / std
                
                pixel_values_list.append(img)
        elif images.dim() == 3:  # 単一画像 [C, H, W]
            # リサイズして正規化
            img = F.interpolate(
                images.unsqueeze(0), 
                size=(gemma_img_size, gemma_img_size), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            
            # 標準的な正規化値を適用
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1)
            img = (img - mean) / std
            
            pixel_values_list.append(img)
    # リストの場合
    elif isinstance(images, list):
        for img in images:
            if torch.is_tensor(img):
                # [C, H, W]形式の場合
                if img.dim() == 3 and (img.shape[0] == 3 or img.shape[0] == 1):
                    # リサイズして正規化
                    img = F.interpolate(
                        img.unsqueeze(0), 
                        size=(gemma_img_size, gemma_img_size), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(0)
                    
                    # 標準的な正規化値を適用
                    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1)
                    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1)
                    img = (img - mean) / std
                    
                    pixel_values_list.append(img)
                else:
                    raise ValueError(f"サポートされていない画像形式です: {img.shape}")
            else:
                # numpyまたはPIL画像の場合
                if isinstance(img, Image.Image):
                    # PILからnumpyに変換
                    img_np = np.array(img)
                else:
                    img_np = img
                
                # PIL画像またはnumpy配列をテンソルに変換し、チャネル次元を最初に移動
                if img_np.ndim == 2:  # グレースケール
                    img_tensor = torch.from_numpy(img_np).unsqueeze(0).float()
                    img_tensor = img_tensor.repeat(3, 1, 1)  # 3チャンネルに複製
                else:  # カラー
                    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
                
                # 値の正規化（0-1の範囲に）
                if img_tensor.max() > 1.0:
                    img_tensor = img_tensor / 255.0
                
                # リサイズ
                img_tensor = F.interpolate(
                    img_tensor.unsqueeze(0), 
                    size=(gemma_img_size, gemma_img_size), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                
                # 標準的な正規化値を適用
                mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1)
                std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1)
                img_tensor = (img_tensor - mean) / std
                
                pixel_values_list.append(img_tensor)
    
    # 処理した画像をバッチ化
    if pixel_values_list:
        pixel_values = torch.stack(pixel_values_list)
        
        # デバイスに移動（指定があれば）
        if device is not None:
            pixel_values = pixel_values.to(device)
        
        return pixel_values
    else:
        raise ValueError("有効な画像が見つかりませんでした") 
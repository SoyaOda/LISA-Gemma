import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from transformers import AutoProcessor
import requests
from io import BytesIO

from .constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def get_gemma_processor(model_name):
    """Gemma3モデル用のProcessorを取得"""
    try:
        from transformers import AutoProcessor
        
        # trust_remote_code=Trueを指定してプロセッサを取得
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        print(f"Gemma3プロセッサを正常に読み込みました: {model_name}")
        return processor
    except Exception as e:
        print(f"Gemma3プロセッサの読み込みに失敗: {e}")
        print("ダミープロセッサを返します")
        return DummyProcessor()


class DummyProcessor:
    """Gemma3プロセッサのフォールバック用ダミークラス"""
    
    def __init__(self):
        """初期化"""
        pass
        
    def __call__(self, images=None, text=None, return_tensors=None):
        """画像処理用の呼び出しメソッド"""
        print("ダミープロセッサが呼ばれました - 処理なし")
        return None
        
    def decode(self, token_ids, skip_special_tokens=False):
        """トークンIDをテキストにデコード"""
        print("ダミープロセッサのデコードが呼ばれました - 処理なし") 
        return ""


class GemmaImageProcessor:
    """Gemma3モデル用の画像プロセッサ - シンプルで明確な実装"""
    
    def __init__(self, processor=None):
        """初期化"""
        # プロセッサは基本的に使用しない（念のため保持）
        self.processor = processor
        
        # 画像サイズとパラメータ - CLIP/Gemma3標準値
        self.size = 224
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073])
        self.std = np.array([0.26862954, 0.26130258, 0.27577711])
    
    def __call__(self, image):
        """
        画像を処理してGemma3モデルに入力可能な形式に変換します
        
        Args:
            image: PIL.Image.Image または numpy.ndarray または ファイルパス
            
        Returns:
            pixel_values: モデルに入力可能な形式の画像テンソル、常に(3, 224, 224)の形状
        """
        # 画像のロードと変換
        try:
            # ファイルパスの場合はPIL画像に読み込む
            if isinstance(image, str):
                try:
                    from PIL import Image
                    image = Image.open(image).convert('RGB')
                except Exception as e:
                    print(f"ファイルからの画像読み込み失敗: {e}")
                    return self._create_zero_tensor()
            
            # numpy配列の場合はPIL画像に変換
            if isinstance(image, np.ndarray):
                try:
                    from PIL import Image
                    image = Image.fromarray(image.astype('uint8')).convert('RGB')
                except Exception as e:
                    print(f"numpy配列からのPIL変換失敗: {e}")
                    return self._create_zero_tensor()
            
            # PIL画像でない場合は処理不可
            if not isinstance(image, Image.Image):
                print(f"未対応の画像タイプ: {type(image)}")
                return self._create_zero_tensor()
            
            # リサイズ (224x224)
            image = image.resize((self.size, self.size), Image.BICUBIC)
            
            # numpy配列に変換
            image_array = np.array(image)
            
            # 値の範囲を[0, 1]に変換
            image_array = image_array.astype(np.float32) / 255.0
            
            # 正規化 (CLIP/Gemma3標準値)
            image_array = (image_array - self.mean.reshape(1, 1, 3)) / self.std.reshape(1, 1, 3)
            
            # PyTorchテンソルに変換 (C, H, W)形式
            tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
            
            return tensor
            
        except Exception as e:
            print(f"画像処理中にエラーが発生: {e}")
            return self._create_zero_tensor()
    
    def _create_zero_tensor(self):
        """ゼロテンソルを作成（エラー時のフォールバック用）"""
        return torch.zeros(3, self.size, self.size) 
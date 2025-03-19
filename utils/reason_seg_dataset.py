import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor
from pycocotools import mask

# 会話テンプレートのインポート
try:
    from model.gemma3 import conversation as gemma_conversation_lib
    GEMMA_AVAILABLE = True
except ImportError:
    GEMMA_AVAILABLE = False

try:
    from model.llava import conversation as conversation_lib
    LLAVA_AVAILABLE = True
except ImportError:
    LLAVA_AVAILABLE = False

from model.segment_anything.utils.transforms import ResizeLongestSide

from .data_processing import get_mask_from_json
from .constants import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, 
                       DEFAULT_IM_END_TOKEN, SHORT_QUESTION_LIST, SYSTEM_PROMPT)
from .conversation import get_default_conv_template

# 説明的な質問リスト
EXPLANATORY_QUESTION_LIST = [
    "Can you explain why this is {cls}?",
    "How do you recognize this as {cls}?",
    "What are the characteristics of {cls} in this image?",
    "What visual features identify this as {cls}?",
    "Why do you think this is {cls}?",
]

# 長い質問リスト
LONG_QUESTION_LIST = [
    "Can you provide a detailed explanation of why this region contains {cls}?",
    "Please explain in detail how you determined that this area shows {cls}.",
    "I'd like to understand the visual cues that led you to identify {cls} here. Can you elaborate?",
    "What specific features in this region indicate that this is {cls}? Please provide details.",
    "Could you give a comprehensive explanation of why you've identified this as {cls}?",
]

class ReasonSegDataset(torch.utils.data.Dataset):
    """理由付きセグメンテーションデータセット"""
    
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255
    
    def __init__(
        self,
        base_image_dir,
        tokenizer,
        model_name,  # Gemma3モデル名
        samples_per_epoch=500 * 8 * 2 * 10,
        precision="fp32",
        image_size=224,
        num_classes_per_sample=3,
        exclude_val=False,
        reason_seg_data="ReasonSeg|train",
        explanatory=0.1,
    ):
        """初期化
        
        Args:
            base_image_dir: ベースとなる画像ディレクトリ
            tokenizer: トークナイザ
            model_name: Gemma3モデル名
            samples_per_epoch: エポックあたりのサンプル数
            precision: 精度
            image_size: 画像サイズ
            num_classes_per_sample: サンプルあたりのクラス数
            exclude_val: 検証データを除外するか
            reason_seg_data: 理由付きセグメンテーションデータ
            explanatory: 説明付きデータの割合
        """
        self.base_image_dir = base_image_dir
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.precision = precision
        self.image_size = image_size
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample
        self.explanatory = explanatory
        
        # データセットの初期化
        self.all_datasets = reason_seg_data.split("||")
        
        # 各データセットの設定
        self.images = []
        self.image_masks = {}
        
        for ds_info in self.all_datasets:
            ds_parts = ds_info.split("|")
            if len(ds_parts) == 2:
                ds, split = ds_parts
                self.init_reasonseg_dataset(ds, split)
        
        print(f"理由付きセグメンテーションデータセット合計: {len(self.images)}画像")
        
        # 説明付き応答用のサンプル
        self.explanatory_images = []
        if os.path.exists(os.path.join(self.base_image_dir, "reason_seg", "ReasonSeg", "explanatory")):
            self.explanatory_images = list(
                glob.glob(
                    os.path.join(self.base_image_dir, "reason_seg", "ReasonSeg", "explanatory", "*.jpg")
                )
            )
            print(f"説明付きデータセット: {len(self.explanatory_images)}画像")
        
        # Gemma3用の画像処理を初期化
        from model.gemma3.mm_utils import get_gemma_processor, GemmaImageProcessor
        
        try:
            # モデル名からプロセッサを取得
            self.processor = get_gemma_processor(model_name)
            self.image_processor = GemmaImageProcessor(self.processor)
            print(f"Gemma3プロセッサを初期化: {model_name}")
        except Exception as e:
            print(f"プロセッサの初期化エラー: {e}")
            # フォールバック: 標準的な前処理を使用
            self.processor = None
            self.image_processor = None
            print("警告: 標準的な画像前処理を使用します")
        
        # SAM用の変換処理
        self.transform = ResizeLongestSide(self.img_size)
    
    def init_reasonseg_dataset(self, ds, split):
        """ReasonSegデータセットを初期化"""
        # データディレクトリパス
        data_dir = os.path.join(self.base_image_dir, "reason_seg", ds, split)
        
        # 画像ファイルパスを取得
        image_files = list(glob.glob(os.path.join(data_dir, "*.jpg")))
        
        for image_path in image_files:
            # 対応するJSONファイルをチェック
            json_path = image_path.replace(".jpg", ".json")
            if os.path.exists(json_path):
                self.images.append(image_path)
        
        print(f"{ds} {split}: {len(self.images)}画像")
    
    def __len__(self):
        return self.samples_per_epoch
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """SAM用に正規化して前処理する"""
        # 正規化
        x = (x - self.pixel_mean) / self.pixel_std
        
        # パディング
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def __getitem__(self, idx):
        """データセットからアイテムを取得"""
        # 説明付きデータを使用するかを決定
        use_explanatory = random.random() < self.explanatory and len(self.explanatory_images) > 0
        
        # 画像パスの選択
        if use_explanatory:
            image_path = random.choice(self.explanatory_images)
        else:
            image_path = random.choice(self.images)
        
        # 画像を読み込み
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # JSONデータを読み込み
        json_path = image_path.replace(".jpg", ".json")
        mask_json, reason_text, is_sentence = get_mask_from_json(json_path, image)
        
        # 会話を生成
        conversations = []
        
        # Gemma3用のテンプレート取得
        template = get_default_conv_template("gemma_v1")
        template.system = SYSTEM_PROMPT
        
        # メッセージを追加
        template.messages = []
        
        if is_sentence:
            # 文章ベースの理由付きセグメンテーション
            query = f"{DEFAULT_IMAGE_TOKEN}\n{reason_text} Please output segmentation mask."
            template.append_message(template.roles[0], query)
            
            if use_explanatory:
                # 説明を含む回答
                response = "[SEG] is the mask for the region described. This region is highlighted because it fulfills the condition mentioned in the instruction."
                template.append_message(template.roles[1], response)
            else:
                # 通常の回答
                template.append_message(template.roles[1], "[SEG].")
        else:
            # 物体ベースの理由付きセグメンテーション
            query = f"{DEFAULT_IMAGE_TOKEN}\nWhat is {reason_text} in this image? Please output segmentation mask."
            template.append_message(template.roles[0], query)
            
            if use_explanatory:
                # 説明を含む回答
                response = "The [SEG] highlights the region of the image containing the requested object. This is the specific area that matches the description."
                template.append_message(template.roles[1], response)
            else:
                # 通常の回答
                template.append_message(template.roles[1], "[SEG].")
        
        conversations.append(template.get_prompt())
        
        # 画像を処理
        # Gemma3用の画像処理
        if self.image_processor is not None:
            images_gemma = self.image_processor(image)
        else:
            # フォールバック: 標準的なリサイズと正規化
            h, w = image.shape[:2]
            size = self.image_size
            image_gemma = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
            image_gemma = torch.from_numpy(image_gemma).permute(2, 0, 1).float() / 255.0
            # 標準的な正規化値を適用
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1)
            images_gemma = (image_gemma - mean) / std
            
        # SAM用の高解像度画像処理
        image_sam = self.transform.apply_image(image)
        resize = image_sam.shape[:2]
        image_sam = self.preprocess(torch.from_numpy(image_sam).permute(2, 0, 1).contiguous())
        
        # マスクの処理
        mask = torch.from_numpy(mask_json).unsqueeze(0)  # (1, H, W)
        
        # ダミーラベルを作成
        label = torch.ones(image.shape[0], image.shape[1]) * self.ignore_label
        
        # 推論フラグと質問のプレースホルダー
        inference = False
        questions = reason_text
        sampled_classes = [reason_text]
        
        return (
            image_path,
            image_sam,
            images_gemma,
            conversations,
            mask,
            label,
            resize,
            questions,
            sampled_classes,
            inference,
        )

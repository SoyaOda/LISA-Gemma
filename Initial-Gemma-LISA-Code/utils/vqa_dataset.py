import json
import os
import random

import cv2
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

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

from .utils import DEFAULT_IMAGE_TOKEN
from .conversation import get_default_conv_template


def preprocess_multimodal(source, mm_use_im_start_end, conv_version="llava_v1"):
    """マルチモーダル入力の前処理"""
    for sentence in source:
        if DEFAULT_IMAGE_TOKEN in sentence["value"]:
            sentence["value"] = (
                sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
            )
            sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
            sentence["value"] = sentence["value"].strip()
            
            # Gemma3とLLaVAのテンプレート形式の違いを処理
            if "gemma" in conv_version and GEMMA_AVAILABLE:
                # Gemma3のフォーマット
                pass  # Gemma3では特別な処理は不要
            elif LLAVA_AVAILABLE and "mmtag" in conversation_lib.default_conversation.version:
                # LLaVAのタグ形式
                sentence["value"] = sentence["value"].replace(
                    DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>"
                )
    return source


class VQADataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        args,
        tokenizer,
        sam_transform,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision="fp32",
        image_size=224,
        num_classes_per_sample=3,
        is_train=True,
    ):
        self.base_image_dir = args.dataset_dir
        self.exclude_val = args.exclude_val if hasattr(args, 'exclude_val') else False
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = args.num_classes_per_sample if hasattr(args, 'num_classes_per_sample') else num_classes_per_sample
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.is_train = is_train
        
        # SAM用のリサイザーを設定
        self.transform = sam_transform
        
        # 会話テンプレートタイプを設定
        self.conv_type = getattr(args, "conv_type", "llava_v1")
        
        # Gemma3を使用しているかどうかを確認
        self.is_gemma3 = "gemma" in self.conv_type and GEMMA_AVAILABLE
        
        # Gemma3用の画像プロセッサ設定
        if hasattr(args, 'version') and args.version:
            if "gemma-3" in args.version:
                try:
                    from transformers import AutoProcessor
                    self.processor = AutoProcessor.from_pretrained(args.version)
                except Exception as e:
                    print(f"Gemma3プロセッサの読み込みに失敗: {e}")
                    self.processor = None
            else:
                # LLaVA用CLIPプロセッサを設定
                self.clip_image_processor = CLIPImageProcessor.from_pretrained(
                    "openai/clip-vit-large-patch14"
                )
        else:
            # デフォルトのCLIPプロセッサを設定
            self.clip_image_processor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-large-patch14"
            )

        # データセットの初期化
        vqa_data_type = args.vqa_data if hasattr(args, 'vqa_data') and args.vqa_data else "llava_instruct_150k"
        DATA_DIR = os.path.join(self.base_image_dir, "llava_dataset")
        self.vqa_image_root = os.path.join(self.base_image_dir, "coco/train2017")
        
        try:
            with open(os.path.join(DATA_DIR, "{}.json".format(vqa_data_type))) as f:
                vqa_data = json.load(f)
            self.vqa_data = vqa_data
            print(f"VQAデータセット: {len(self.vqa_data)} サンプル")
        except FileNotFoundError:
            print(f"警告: VQAデータセットファイルが見つかりません: {os.path.join(DATA_DIR, vqa_data_type)}.json")
            self.vqa_data = []

    def __len__(self):
        return self.samples_per_epoch if len(self.vqa_data) > 0 else 0

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        # データセットが空の場合はエラー
        if len(self.vqa_data) == 0:
            raise ValueError("VQAデータセットが空です")
            
        # ランダムにサンプルを選択
        idx = random.randint(0, len(self.vqa_data) - 1)
        item = self.vqa_data[idx]
        image_path = os.path.join(self.vqa_image_root, item["image"])
        
        # 画像の読み込み
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]
        
        # 画像の前処理
        if hasattr(self, 'processor') and self.processor is not None:
            # Gemma3のプロセッサを使用
            image_clip = self.processor.image_processor(image, return_tensors="pt").pixel_values[0]
        else:
            # CLIPイメージプロセッサを使用
            image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        
        # SAM用の画像前処理
        image_sam = self.transform.apply_image(image)
        resize = image_sam.shape[:2]

        # 会話データの処理
        if self.is_gemma3:
            # Gemma3用の会話テンプレート
            conv = get_default_conv_template(self.conv_type)
            
            # マルチモーダル前処理
            source = item["conversations"]
            source = preprocess_multimodal(source, mm_use_im_start_end=False, conv_version=self.conv_type)
            
            # 役割のマッピング
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
            
            # 会話の構築
            conversations = []
            if roles[source[0]["from"]] != conv.roles[0]:
                # humanからの発話でなければスキップ
                source = source[1:]
                
            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"役割が一致しません: {idx}, {j}"
                conv.append_message(role, sentence["value"])
                
            conversations.append(conv.get_prompt())
            
        else:
            # LLaVA用の会話テンプレート
            if LLAVA_AVAILABLE:
                conv = conversation_lib.default_conversation.copy()
            else:
                conv = get_default_conv_template("llava_v1")
                
            # マルチモーダル前処理
            source = item["conversations"]
            source = preprocess_multimodal(
                source,
                mm_use_im_start_end=(conv.sep_style == conversation_lib.SeparatorStyle.TWO if hasattr(conversation_lib, 'SeparatorStyle') else False)
            )
            
            # 役割のマッピング
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
            
            # 会話の構築
            conversations = []
            if roles[source[0]["from"]] != conv.roles[0]:
                # humanからの発話でなければスキップ
                source = source[1:]
                
            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"役割が一致しません: {idx}, {j}"
                conv.append_message(role, sentence["value"])
                
            conversations.append(conv.get_prompt())

        # 質問と回答の整理
        questions = conversations
        sampled_classes = conversations

        # SAM用の画像をテンソルに変換
        image_sam = self.preprocess(torch.from_numpy(image_sam).permute(2, 0, 1).contiguous())

        # マスクは空（VQAタスクでは不要）
        masks = torch.rand(0, *ori_size)
        
        # ラベルの準備
        label = torch.ones(ori_size) * self.ignore_label
        
        # 学習/推論モードのフラグ
        inference = False
        
        return image_path, image_sam, image_clip, conversations, masks, label, resize, questions, sampled_classes, inference

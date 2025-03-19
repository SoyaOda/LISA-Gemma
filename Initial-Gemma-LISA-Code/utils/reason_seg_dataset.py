import glob
import json
import os
import random

import cv2
import numpy as np
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

from .data_processing import get_mask_from_json
from .utils import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN,
                    EXPLANATORY_QUESTION_LIST, LONG_QUESTION_LIST,
                    SHORT_QUESTION_LIST)
from .conversation import get_default_conv_template


class ReasonSegDataset(torch.utils.data.Dataset):
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
        self.reason_seg_data_str = args.reason_seg_data if hasattr(args, 'reason_seg_data') else "ReasonSeg|train"
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = args.explanatory if hasattr(args, 'explanatory') else 0.1
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

        # 質問と回答のテンプレート
        self.short_question_list = SHORT_QUESTION_LIST
        self.long_question_list = LONG_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        # データセットの初期化
        try:
            reason_seg_data, splits = self.reason_seg_data_str.split("|")
            splits = splits.split("_")
            images = []
            for split in splits:
                images_split = glob.glob(
                    os.path.join(
                        self.base_image_dir, "reason_seg", reason_seg_data, split, "*.jpg"
                    )
                )
                images.extend(images_split)
            jsons = [path.replace(".jpg", ".json") for path in images]
            self.reason_seg_data = (images, jsons)

            print(f"推論セグメンテーションデータセット サンプル数: {len(images)}")

            # 説明的なセグメンテーション用のデータ
            if self.explanatory != -1:
                self.explanatory_question_list = EXPLANATORY_QUESTION_LIST
                self.img_to_explanation = {}
                explanatory_path = os.path.join(
                    self.base_image_dir,
                    "reason_seg",
                    reason_seg_data,
                    "explanatory",
                    "train.json",
                )
                
                if os.path.exists(explanatory_path):
                    with open(explanatory_path) as f:
                        items = json.load(f)
                    for item in items:
                        img_name = item["image"]
                        self.img_to_explanation[img_name] = {
                            "query": item["query"],
                            "outputs": item["outputs"],
                        }

                    print(f"説明データセット サンプル数: {len(self.img_to_explanation)}")
                else:
                    print(f"警告: 説明データファイルが見つかりません: {explanatory_path}")
                    self.img_to_explanation = {}
        except Exception as e:
            print(f"推論セグメンテーションデータセットの初期化エラー: {e}")
            self.reason_seg_data = ([], [])
            self.img_to_explanation = {}

    def __len__(self):
        images, _ = self.reason_seg_data
        return self.samples_per_epoch if len(images) > 0 else 0

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
        images, jsons = self.reason_seg_data
        if len(images) == 0:
            raise ValueError("推論セグメンテーションデータセットが空です")
            
        # ランダムにサンプルを選択
        idx = random.randint(0, len(images) - 1)
        image_path = images[idx]
        json_path = jsons[idx]

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
            image_clip = self.clip_image_processor.preprocess(
                image, return_tensors="pt"
            )["pixel_values"][0]
        
        # マスクとテキストデータの取得
        mask, sents, is_sentence = get_mask_from_json(json_path, image)
        
        # サンプル数の調整
        if len(sents) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(sents))), size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(sents)))
            
        # サンプリングされた文章とマスクを取得
        sampled_sents = [sents[ind] for ind in sampled_inds]
        sampled_masks = [
            (mask == 1).astype(np.float32) for _ in range(len(sampled_inds))
        ]

        # SAM用の画像前処理
        image_sam = self.transform.apply_image(image)
        resize = image_sam.shape[:2]

        # 説明的な文章の選択
        image_name = image_path.split("/")[-1]
        choice = 0  # デフォルトは [SEG] トークンのみ
        if self.explanatory != -1 and image_name in self.img_to_explanation:
            if random.random() < self.explanatory:
                choice = 2  # テキスト回答のみ
            else:
                choice = random.randint(0, 1)  # 0: [SEG]トークンのみ、1: [SEG]トークン+テキスト

        # 会話テンプレートと質問・回答の作成
        questions = []
        answers = []
        
        for text in sampled_sents:
            # 質問テンプレートの選択
            if is_sentence:
                question_template = random.choice(self.long_question_list)
                question = question_template.format(sent=text)
            else:
                question_template = random.choice(self.short_question_list)
                question = question_template.format(class_name=text.lower())
                
            questions.append(question)

            # 説明が利用可能な場合は回答を調整
            img_name = image_path.split("/")[-1]
            if self.explanatory != -1 and img_name in self.img_to_explanation:
                if choice == 0:  # [SEG] トークンのみ
                    answers.append(random.choice(self.answer_list))
                elif choice == 1:  # [SEG] トークン + テキスト回答
                    image_name = image_path.split("/")[-1]
                    answer_text = self.img_to_explanation[image_name]["outputs"]
                    answer = random.choice(self.answer_list) + " {}".format(answer_text)
                    questions[-1] = (
                        DEFAULT_IMAGE_TOKEN
                        + "\n"
                        + text
                        + " {}".format(random.choice(self.explanatory_question_list))
                    )
                    answers.append(answer)
                elif choice == 2:  # テキスト回答のみ
                    image_name = image_path.split("/")[-1]
                    answer_text = self.img_to_explanation[image_name]["outputs"]
                    questions[-1] = DEFAULT_IMAGE_TOKEN + "\n" + text
                    answers.append(answer_text)
                else:
                    raise ValueError("未実装の選択肢です")
            else:
                answers.append(random.choice(self.answer_list))

        # 会話プロンプトの作成
        conversations = []
        
        for i in range(len(questions)):
            # 会話テンプレートの選択
            if self.is_gemma3:
                conv = get_default_conv_template(self.conv_type)
            else:
                if LLAVA_AVAILABLE:
                    conv = conversation_lib.default_conversation.copy()
                else:
                    conv = get_default_conv_template(self.conv_type)
            
            # プロンプトの生成
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())

        # SAM用の画像をテンソルに変換
        image_sam = self.preprocess(torch.from_numpy(image_sam).permute(2, 0, 1).contiguous())

        # マスクの処理（テキスト回答のみの場合は空のマスク）
        image_name = image_path.split("/")[-1]
        if (
            self.explanatory != -1
            and image_name in self.img_to_explanation
            and choice == 2
        ):
            # テキスト回答のみの場合はマスクなし
            masks = torch.rand(0, *ori_size)
            label = torch.ones(ori_size) * self.ignore_label
        else:
            # 通常のマスク処理
            masks = np.stack(sampled_masks, axis=0)
            masks = torch.from_numpy(masks)
            label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        
        # 学習/推論モードのフラグ
        inference = False
        
        return image_path, image_sam, image_clip, conversations, masks, label, resize, questions, sampled_sents, inference

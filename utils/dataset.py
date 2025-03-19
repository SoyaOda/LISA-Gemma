# import argparse
import glob
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor, AutoProcessor

# Gemma3のモジュールをインポート
try:
    from model.gemma3 import conversation as gemma_conversation_lib
    from model.gemma3.mm_utils import GemmaImageProcessor
    from model.gemma3.constants import (DEFAULT_IMAGE_TOKEN as GEMMA_IMAGE_TOKEN,
                                      DEFAULT_IM_START_TOKEN as GEMMA_IM_START_TOKEN,
                                      DEFAULT_IM_END_TOKEN as GEMMA_IM_END_TOKEN,
                                      SYSTEM_PROMPT_GEMMA)
    GEMMA_AVAILABLE = True
except ImportError:
    GEMMA_AVAILABLE = False

# LLaVA関連モジュールをインポート（互換性のため）
try:
    from model.llava import conversation as conversation_lib
    from model.llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                                     IMAGE_TOKEN_INDEX)
    from model.llava.mm_utils import tokenizer_image_token
    LLAVA_AVAILABLE = True
except ImportError:
    LLAVA_AVAILABLE = False
    # LLaVAがない場合のフォールバック定義
    from .constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
    
    def tokenizer_image_token(prompt, tokenizer, return_tensors=None):
        """LLaVAのtokenizer_image_token関数を模倣"""
        input_ids = tokenizer(prompt, return_tensors=return_tensors).input_ids
        return input_ids

from model.segment_anything.utils.transforms import ResizeLongestSide

from .constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, SYSTEM_PROMPT
from .conversation import get_default_conv_template
from .data_processing import get_mask_from_json
from .reason_seg_dataset import ReasonSegDataset
from .refer import REFER
from .refer_seg_dataset import ReferSegDataset
from .sem_seg_dataset import SemSegDataset
from .vqa_dataset import VQADataset

# Gemma3のシステムプロンプトを使用 - 定数ファイルに移動
# SYSTEM_PROMPT定義を削除（constants.pyに移動済み）
    
# Gemma3とLLaVAで共通に使用される画像トークン - 定数ファイルに移動
# 以下の定義を削除（constants.pyに移動済み）
# DEFAULT_IMAGE_TOKEN = "<image>"
# DEFAULT_IM_START_TOKEN = "<im_start>"  
# DEFAULT_IM_END_TOKEN = "<im_end>"

def tokenizer_image_token(text, tokenizer, return_tensors=None):
    """テキスト内の画像トークンを処理し、トークナイズします"""
    # 画像トークンをプレースホルダに置き換え
    for img_token in [DEFAULT_IMAGE_TOKEN]:
        if img_token in text:
            text = text.replace(img_token, tokenizer.pad_token)
    
    # トークナイズ
    tokens = tokenizer(
        text, 
        return_tensors=return_tensors,
        padding="longest"
    ).input_ids
    
    # 画像トークンのインデックスを置き換え
    return tokens

def collate_fn(
    batch, tokenizer=None, conv_type="gemma_v1", use_mm_start_end=True, local_rank=-1
):
    """バッチを整形するcollate関数"""
    image_path_list = []
    images_list = []  # SAM用高解像度画像
    images_gemma_list = []  # Gemma3視覚モデル用画像
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    
    for (
        image_path,
        images,
        images_gemma,
        conversations,
        masks,
        label,
        resize,
        questions,
        sampled_classes,
        inference,
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_gemma_list.append(images_gemma)
        conversation_list.extend(conversations)
        label_list.append(label)
        masks_list.append(masks.float())
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)

    # 画像トークンの置き換え処理
    if use_mm_start_end:
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            conversation_list[i] = conversation_list[i].replace(DEFAULT_IMAGE_TOKEN, replace_token)
    
    # テキストのトークナイズ
    input_ids = []
    max_length = 0
    
    # まず各会話をトークナイズして最大長を確認
    conversation_tokens = []
    for prompt in conversation_list:
        try:
            # トークナイズ（テンソル変換なし）
            tokens = tokenizer_image_token(prompt, tokenizer, return_tensors=None)
            conversation_tokens.append(tokens)
            
            # 最大長を更新
            if len(tokens) > max_length:
                max_length = len(tokens)
        except Exception as e:
            print(f"トークナイズエラー: {e}")
            # エラー発生時は空のトークン列を使用
            empty_tokens = tokenizer("", return_tensors=None).input_ids
            conversation_tokens.append(empty_tokens)
    
    # 各会話をPyTorchテンソルに変換して最大長に統一
    for tokens in conversation_tokens:
        # トークンをテンソルに変換
        token_tensor = torch.tensor(tokens, dtype=torch.long)
        input_ids.append(token_tensor)
    
    # パディングを適用して全てのシーケンスを同じ長さに
    try:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
    except RuntimeError as e:
        print(f"パディングエラー: {e}")
        # 緊急措置: 手動でパディングを適用
        padded_ids = []
        for ids in input_ids:
            if len(ids) < max_length:
                padding = torch.full((max_length - len(ids),), tokenizer.pad_token_id, dtype=torch.long)
                padded = torch.cat([ids, padding], dim=0)
            else:
                padded = ids[:max_length]  # 切り詰め
            padded_ids.append(padded)
        input_ids = torch.stack(padded_ids, dim=0)
    
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    # ターゲットラベルの作成
    targets = input_ids.clone()
    
    # ラベルの作成（教師強制用）
    # Gemma3の会話形式に合わせてセパレータとロールを設定
    if conv_type == "gemma_v1":
        sep = "\n\nAssistant: "
    else:
        sep = "[/INST] "
    
    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        
        # システムプロンプト部分とユーザークエリ部分はIGNORE_INDEXに設定
        # Gemma3の会話形式に応じて調整
        # Gemma3のプロンプト形式: "System: {system}\n\nUser: {query}\n\nAssistant: {response}"
        user_assistant_sep = sep
        system_user_sep = "\n\nUser: "
        
        if system_user_sep in conversation and user_assistant_sep in conversation:
            # システムプロンプトの開始からアシスタント応答の開始までをIGNORE_INDEXに
            system_start = 0
            user_start = conversation.find(system_user_sep) + len(system_user_sep)
            assistant_start = conversation.find(user_assistant_sep) + len(user_assistant_sep)
            
            # トークン位置に変換
            user_token_start = len(tokenizer(conversation[:user_start]).input_ids) - 1
            assistant_token_start = len(tokenizer(conversation[:assistant_start]).input_ids) - 1
            
            # システム・ユーザー部分はIGNORE_INDEXに設定
            target[:assistant_token_start] = IGNORE_INDEX
        
        # パディング部分もIGNORE_INDEXに設定
        target[total_len:] = IGNORE_INDEX
    
    # 長いシーケンスのトランケーション
    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length - 255  # 画像トークン用の余裕

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    # SAM用画像とGemma用画像の形状を確認し整形
    valid_sam_images = []
    valid_gemma_images = []
    
    for sam_img, gemma_img in zip(images_list, images_gemma_list):
        # 両方のテンソルが有効であることを確認
        if sam_img is not None and isinstance(sam_img, torch.Tensor) and sam_img.dim() == 3:
            if gemma_img is not None and isinstance(gemma_img, torch.Tensor) and gemma_img.dim() == 3:
                valid_sam_images.append(sam_img)
                valid_gemma_images.append(gemma_img)
            else:
                # Gemma画像が無効な場合はゼロテンソルで置き換え
                print(f"警告: 無効なGemma画像形状: {type(gemma_img)}")
                valid_sam_images.append(sam_img)
                valid_gemma_images.append(torch.zeros(3, 224, 224))
        else:
            # どちらも無効な場合はスキップ
            print(f"警告: 無効な画像ペア、このサンプルをスキップします")
    
    # 有効な画像がない場合は、デフォルトのゼロテンソルを作成
    if not valid_sam_images:
        valid_sam_images = [torch.zeros(3, 1024, 1024)]
        valid_gemma_images = [torch.zeros(3, 224, 224)]
    
    # 有効な画像のみでバッチを構成
    try:
        images_tensor = torch.stack(valid_sam_images, dim=0)
        pixel_values_tensor = torch.stack(valid_gemma_images, dim=0)
    except RuntimeError as e:
        print(f"画像スタックエラー: {e}")
        # サイズが不一致の場合、最初の有効なサイズに合わせる
        if valid_sam_images and valid_gemma_images:
            target_sam_shape = valid_sam_images[0].shape
            target_gemma_shape = valid_gemma_images[0].shape
            
            # すべての画像を同じサイズに調整
            adjusted_sam_images = []
            adjusted_gemma_images = []
            
            for sam_img, gemma_img in zip(valid_sam_images, valid_gemma_images):
                if sam_img.shape != target_sam_shape:
                    # リサイズまたはパディングで調整
                    sam_img = F.interpolate(sam_img.unsqueeze(0), size=target_sam_shape[1:], mode='bilinear').squeeze(0)
                adjusted_sam_images.append(sam_img)
                
                if gemma_img.shape != target_gemma_shape:
                    gemma_img = F.interpolate(gemma_img.unsqueeze(0), size=target_gemma_shape[1:], mode='bilinear').squeeze(0)
                adjusted_gemma_images.append(gemma_img)
            
            images_tensor = torch.stack(adjusted_sam_images, dim=0)
            pixel_values_tensor = torch.stack(adjusted_gemma_images, dim=0)
        else:
            # 最終手段: デフォルト値で埋める
            images_tensor = torch.zeros(len(image_path_list), 3, 1024, 1024)
            pixel_values_tensor = torch.zeros(len(image_path_list), 3, 224, 224)

    return {
        "image_paths": image_path_list,
        "images": images_tensor,
        "pixel_values": pixel_values_tensor,  # Gemma3用の画像テンソル
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
    }


class HybridDataset(torch.utils.data.Dataset):
    """Gemma3モデル用のハイブリッドデータセット"""
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        model_name,  # Gemma3モデル名を指定
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        dataset="sem_seg||refer_seg||vqa||reason_seg",
        sample_rate=[9, 3, 3, 1],
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        vqa_data="llava_instruct_150k",
        reason_seg_data="ReasonSeg|train",
        explanatory=0.1,
    ):
        self.exclude_val = exclude_val
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        
        # Gemma3のプロセッサを初期化
        self.processor = None
        self.image_processor = None
        
        try:
            # 直接transformersからAutoProcessorを使用
            from transformers import AutoProcessor, AutoImageProcessor
            
            # trust_remote_code=Trueを指定してプロセッサを取得
            self.processor = AutoProcessor.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            
            # 改善されたGemmaImageProcessorを使用
            from model.gemma3.mm_utils import GemmaImageProcessor
            self.image_processor = GemmaImageProcessor(self.processor)
            
            # バックアップとしてCLIPのイメージプロセッサも取得
            self.image_processor_fallback = AutoImageProcessor.from_pretrained(
                "openai/clip-vit-large-patch14",
                trust_remote_code=True
            )
            
            print(f"Gemma3プロセッサを初期化: {model_name}")
        except Exception as e:
            print(f"プロセッサの初期化エラー: {e}")
            print("警告: 標準的な画像前処理を使用します")
        
        # SAM用の画像処理
        self.transform = ResizeLongestSide(self.img_size)

        self.datasets = dataset.split("||")

        self.all_datasets = []
        for dataset in self.datasets:
            if dataset == "sem_seg":
                self.all_datasets.append(
                    SemSegDataset(
                        base_image_dir,
                        tokenizer,
                        model_name,  # vision_towerの代わりにmodel_nameを渡す
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        sem_seg_data,
                    )
                )
            elif dataset == "refer_seg":
                self.all_datasets.append(
                    ReferSegDataset(
                        base_image_dir,
                        tokenizer,
                        model_name,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        refer_seg_data,
                    )
                )
            elif dataset == "vqa":
                self.all_datasets.append(
                    VQADataset(
                        base_image_dir,
                        tokenizer,
                        model_name,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        vqa_data,
                    )
                )
            elif dataset == "reason_seg":
                self.all_datasets.append(
                    ReasonSegDataset(
                        base_image_dir,
                        tokenizer,
                        model_name,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        reason_seg_data,
                        explanatory,
                    )
                )

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # ランダムにサブデータセットを選択
        ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
        data = self.all_datasets[ind]
        inference = False
        return *data[0], inference


class ValDataset(torch.utils.data.Dataset):
    """評価用データセット"""
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        model_name,  # Gemma3モデル名
        val_dataset,
        image_size=1024,
    ):
        self.base_image_dir = base_image_dir
        splits = val_dataset.split("|")
        if len(splits) == 2:
            ds, split = splits
            images = glob.glob(
                os.path.join(self.base_image_dir, "reason_seg", ds, split, "*.jpg")
            )
            self.images = images
            self.data_type = "reason_seg"
        elif len(splits) == 3:
            ds, splitBy, split = splits
            refer_api = REFER(self.base_image_dir, ds, splitBy)
            ref_ids_val = refer_api.getRefIds(split=split)
            images_ids_val = refer_api.getImgIds(ref_ids=ref_ids_val)
            refs_val = refer_api.loadRefs(ref_ids=ref_ids_val)
            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_val)
            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        base_image_dir, "images/saiapr_tc-12", item["file_name"]
                    )
                elif ds in ["refcoco", "refcoco+", "refcocog", "grefcoco"]:
                    item["file_name"] = os.path.join(
                        base_image_dir,
                        "images/mscoco/images/train2014",
                        item["file_name"],
                    )
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_val

            img2refs = {}
            for ref in refs_val:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_ds = refer_seg_ds
            self.data_type = "refer_seg"

        self.ds = ds
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        
        # Gemma3のプロセッサを初期化
        try:
            # モデル名からプロセッサを取得
            self.processor = get_gemma_processor(model_name)
            self.image_processor = GemmaImageProcessor(self.processor)
        except Exception as e:
            print(f"プロセッサの初期化エラー: {e}")
            # フォールバック: 標準的な前処理を使用
            self.processor = None
            self.image_processor = None
            print("警告: 標準的な画像前処理を使用します")

    def __len__(self):
        if self.data_type == "refer_seg":
            return len(self.refer_seg_ds["images"])
        else:
            return len(self.images)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """SAM用にピクセル値を正規化しパディングする"""
        # 正規化
        x = (x - self.pixel_mean) / self.pixel_std

        # パディング
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        """データセットから要素を取得"""
        if self.data_type == "refer_seg":
            refer_seg_ds = self.refer_seg_ds
            images = refer_seg_ds["images"]
            annotations = refer_seg_ds["annotations"]
            img2refs = refer_seg_ds["img2refs"]

            image_info = images[idx]
            image_path = image_info["file_name"]
            image_id = image_info["id"]

            refs = img2refs[image_id]
            if len(refs) == 0:
                raise ValueError(f"画像 {image_id} に対する参照がありません")

            sents = []
            ann_ids = []
            for ref in refs:
                for sent in ref["sentences"]:
                    sents.append(sent["sent"].strip().lower())
                    ann_ids.append(ref["ann_id"])

            sampled_sents = sents
            sampled_ann_ids = ann_ids
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            is_sentence = False
        else:
            image_path = self.images[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            json_path = image_path.replace(".jpg", ".json")
            mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, image)
            sampled_sents = [sampled_sents[0]]

        # Gemma3用の会話形式を作成
        conversations = []
        template = get_default_conv_template("gemma_v1")  # Gemma3用テンプレート
        
        i = 0
        while i < len(sampled_sents):
            template.messages = []
            text = sampled_sents[i].strip()
            
            # システムプロンプトを設定
            template.system = SYSTEM_PROMPT
            
            if is_sentence:
                # 文章ベースのセグメンテーション
                query = f"{DEFAULT_IMAGE_TOKEN}\n{text} Please output segmentation mask."
                template.append_message(template.roles[0], query)
                template.append_message(template.roles[1], "[SEG].")
            else:
                # 物体ベースのセグメンテーション
                query = f"{DEFAULT_IMAGE_TOKEN}\nWhat is {text} in this image? Please output segmentation mask."
                template.append_message(template.roles[0], query)
                template.append_message(template.roles[1], "[SEG].")
            
            conversations.append(template.get_prompt())
            i += 1

        # Gemma3用の画像処理
        if self.image_processor is not None:
            # Gemma3の視覚モデル用の前処理
            images_gemma = self.image_processor(image)
        else:
            # フォールバック処理: 標準的なリサイズと正規化
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
        if self.data_type == "refer_seg":
            masks = []
            for i, ann_id in enumerate(sampled_ann_ids):
                ann = annotations[ann_id]
                if len(ann["segmentation"]) == 0 and sampled_sents[i] != "":
                    m = np.zeros((image_info["height"], image_info["width"], 1))
                else:
                    if type(ann["segmentation"][0]) == list:  # polygon
                        rle = mask.frPyObjects(
                            ann["segmentation"],
                            image_info["height"],
                            image_info["width"],
                        )
                    else:
                        rle = ann["segmentation"]
                        for i in range(len(rle)):
                            if not isinstance(rle[i]["counts"], bytes):
                                rle[i]["counts"] = rle[i]["counts"].encode()
                    m = mask.decode(rle)
                m = np.sum(m, axis=2)  # 複数のバイナリマップを合成
                m = m.astype(np.uint8)  # np.uint8に変換
                masks.append(m)
        else:
            masks = [mask_json]

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

        return (
            image_path,
            image_sam,
            images_gemma,
            conversations,
            masks,
            labels,
            resize,
            None,
            None,
            inference,
        )

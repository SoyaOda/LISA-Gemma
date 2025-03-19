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
                                      DEFAULT_IM_END_TOKEN as GEMMA_IM_END_TOKEN)
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
    DEFAULT_IMAGE_TOKEN = "<image>"
    IGNORE_INDEX = -100
    IMAGE_TOKEN_INDEX = -200
    
    def tokenizer_image_token(prompt, tokenizer, return_tensors=None):
        """LLaVAのtokenizer_image_token関数を模倣"""
        input_ids = tokenizer(prompt, return_tensors=return_tensors).input_ids
        return input_ids

from model.segment_anything.utils.transforms import ResizeLongestSide

from .conversation import get_default_conv_template
from .data_processing import get_mask_from_json
from .reason_seg_dataset import ReasonSegDataset
from .refer import REFER
from .refer_seg_dataset import ReferSegDataset
from .sem_seg_dataset import SemSegDataset
from .utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_TOKEN)
from .vqa_dataset import VQADataset


def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
    image_path_list = []
    images_list = []
    images_clip_list = []
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
        images_clip,
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
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        label_list.append(label)
        masks_list.append(masks.float())
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)

    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
            
    # 会話テンプレートの種類に基づいて処理
    if "gemma" in conv_type and GEMMA_AVAILABLE:
        # Gemma3モデル用の処理
        # GemmaモデルのプロンプトにはAutoProcessorを使用せず、
        # Gemma用の特殊トークンを考慮した基本的なトークナイズを行う
        input_ids = [
            tokenizer(prompt, return_tensors="pt").input_ids.squeeze(0)
            for prompt in conversation_list
        ]
    else:
        # 従来のLLaVA処理
        input_ids = [
            tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
            for prompt in conversation_list
        ]
        
    # パディング処理
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    # 会話テンプレートに応じて適切な会話クラスを選択
    if "gemma" in conv_type and GEMMA_AVAILABLE:
        conv = gemma_conversation_lib.get_default_conv_template(conv_type)
        sep = conv.roles[1] + "\n"  # Gemma3形式のセパレータ
    elif LLAVA_AVAILABLE:
        conv = conversation_lib.default_conversation.copy()
        if conv_type == "llava_v1":
            sep = conv.sep + conv.roles[1] + ": "
        else:
            sep = "[/INST] "
    else:
        # 両方のモデルが利用できない場合のフォールバック
        sep = "ASSISTANT: "
        
    targets = input_ids.clone()
    
    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        # 会話をターンごとに分割してターゲットを設定
        if "gemma" in conv_type and GEMMA_AVAILABLE:
            rounds = conversation.split(conv.roles[0])
        else:
            rounds = conversation.split(conv.sep2 if hasattr(conv, 'sep2') else "\n")
            
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        
        for i, rou in enumerate(rounds):
            if len(rou) == 0:
                continue

            if "gemma" in conv_type and GEMMA_AVAILABLE:
                if i == 0 and conv.system in rou:
                    # システムプロンプトはIGNORE_INDEXにする
                    parts = rou.split(conv.roles[1], 1)
                    if len(parts) > 1:
                        parts = [parts[0] + conv.roles[1], parts[1]]
                    else:
                        parts = [rou]
                        
                    if len(parts[0]) > 0:
                        round_len = len(tokenizer(parts[0], return_tensors="pt").input_ids[0])
                        target[cur_len : cur_len + round_len] = IGNORE_INDEX
                        cur_len += round_len
                        
                    if len(parts) > 1 and len(parts[1]) > 0:
                        round_len = len(tokenizer(parts[1], return_tensors="pt").input_ids[0])
                        cur_len += round_len
                else:
                    if conv.roles[0] in rou:
                        parts = rou.split(conv.roles[1], 1)
                        
                        if len(parts[0]) > 0:
                            round_len = len(tokenizer(parts[0], return_tensors="pt").input_ids[0])
                            target[cur_len : cur_len + round_len] = IGNORE_INDEX
                            cur_len += round_len
                            
                        if len(parts) > 1 and len(parts[1]) > 0:
                            round_len = len(tokenizer(parts[1], return_tensors="pt").input_ids[0])
                            cur_len += round_len
                    else:
                        round_len = len(tokenizer(rou, return_tensors="pt").input_ids[0])
                        cur_len += round_len
            else:
                # 従来のLLaVA処理
                if rou.startswith(conv.roles[0]):
                    round_len = len(tokenizer_image_token(rou, tokenizer, return_tensors="pt"))
                    target[cur_len : cur_len + round_len] = IGNORE_INDEX
                    cur_len += round_len
                elif rou.startswith(conv.roles[1]):
                    round_len = len(tokenizer_image_token(rou, tokenizer, return_tensors="pt"))
                    cur_len += round_len
                else:
                    round_len = len(tokenizer_image_token(rou, tokenizer, return_tensors="pt"))
                    target[cur_len : cur_len + round_len] = IGNORE_INDEX
                    cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < total_len:
            if cur_len == 0:
                target[:] = IGNORE_INDEX
            else:
                target[cur_len:total_len] = IGNORE_INDEX

    return {
        "image_paths": image_path_list,
        "images": torch.cat(images_list),
        "images_clip": torch.stack(images_clip_list),
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "offset": torch.LongTensor(offset_list),
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
    }


class HybridDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        args,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision="fp32",
        image_size=224,
        num_classes_per_sample=3,
    ):
        self.tokenizer = tokenizer
        self.precision = precision
        self.samples_per_epoch = samples_per_epoch
        self.image_size = image_size
        self.num_classes_per_sample = num_classes_per_sample
        self.is_gemma3 = "gemma-3" in args.version if args.version else False

        # タスクごとのデータセットの初期化
        self.sem_seg_dataset = None
        self.sem_seg_dataset_length = 0
        self.refer_seg_dataset = None
        self.refer_seg_dataset_length = 0
        self.vqa_dataset = None
        self.vqa_dataset_length = 0
        self.reason_seg_dataset = None
        self.reason_seg_dataset_length = 0

        # Gemma3用の画像プロセッサを初期化
        if self.is_gemma3:
            try:
                from transformers import AutoProcessor
                self.processor = AutoProcessor.from_pretrained(args.version)
                print(f"初期化しました: Gemma3プロセッサ (from {args.version})")
            except Exception as e:
                print(f"Gemma3プロセッサの初期化に失敗しました: {e}")
                print("フォールバック: DummyProcessorを使用します")
                # フォールバックとしてダミープロセッサを使用
                class DummyProcessor:
                    def __call__(self, images, **kwargs):
                        return {"pixel_values": images}
                self.processor = DummyProcessor()
        else:
            # LLaVA用にCLIPイメージプロセッサを初期化
            from transformers import CLIPImageProcessor
            self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
            
        # SAM用の画像処理のためのリサイザー
        self.sam_transform = ResizeLongestSide(self.img_size)

        # タスクの比率を計算
        self.cum_samples = None
        self.datasets_names = []
        self.sample_rate = []
        
        # データセットを初期化
        # 各タスクのデータセットを設定（設定されている場合）
        
        # セマンティックセグメンテーションデータセット
        if args.sem_seg_data:
            try:
                from utils.sem_seg_dataset import SemSegDataset
                self.sem_seg_dataset = SemSegDataset(args, tokenizer, self.sam_transform)
                self.sem_seg_dataset_length = len(self.sem_seg_dataset)
                self.datasets_names.append("sem_seg")
                self.sample_rate.append(args.sem_seg_sample_rate if hasattr(args, 'sem_seg_sample_rate') else 0.25)
                print(f"セマンティックセグメンテーションデータセットが初期化されました。サイズ: {self.sem_seg_dataset_length}")
            except Exception as e:
                print(f"セマンティックセグメンテーションデータセットの初期化に失敗しました: {e}")
                
        # 参照セグメンテーションデータセット
        if args.refer_seg_data:
            try:
                from utils.refer_seg_dataset import ReferSegDataset
                self.refer_seg_dataset = ReferSegDataset(args, tokenizer, self.sam_transform)
                self.refer_seg_dataset_length = len(self.refer_seg_dataset)
                self.datasets_names.append("refer_seg")
                self.sample_rate.append(args.refer_seg_sample_rate if hasattr(args, 'refer_seg_sample_rate') else 0.25)
                print(f"参照セグメンテーションデータセットが初期化されました。サイズ: {self.refer_seg_dataset_length}")
            except Exception as e:
                print(f"参照セグメンテーションデータセットの初期化に失敗しました: {e}")
                
        # VQAデータセット
        if args.vqa_data:
            try:
                from utils.vqa_dataset import VQADataset
                self.vqa_dataset = VQADataset(args, tokenizer, self.sam_transform)
                self.vqa_dataset_length = len(self.vqa_dataset)
                self.datasets_names.append("vqa")
                self.sample_rate.append(args.vqa_sample_rate if hasattr(args, 'vqa_sample_rate') else 0.25)
                print(f"VQAデータセットが初期化されました。サイズ: {self.vqa_dataset_length}")
            except Exception as e:
                print(f"VQAデータセットの初期化に失敗しました: {e}")
                
        # Reasoning Segmentationデータセット
        if args.reason_seg_data:
            try:
                from utils.reason_seg_dataset import ReasonSegDataset
                self.reason_seg_dataset = ReasonSegDataset(args, tokenizer, self.sam_transform)
                self.reason_seg_dataset_length = len(self.reason_seg_dataset)
                self.datasets_names.append("reason_seg")
                self.sample_rate.append(args.reason_seg_sample_rate if hasattr(args, 'reason_seg_sample_rate') else 0.25)
                print(f"Reasoning Segmentationデータセットが初期化されました。サイズ: {self.reason_seg_dataset_length}")
            except Exception as e:
                print(f"Reasoning Segmentationデータセットの初期化に失敗しました: {e}")

        # サンプリング率を正規化し、累積確率を計算
        total_rate = sum(self.sample_rate)
        if total_rate > 0:
            self.sample_rate = [rate / total_rate for rate in self.sample_rate]
            self.cum_samples = [sum(self.sample_rate[:i+1]) for i in range(len(self.sample_rate))]
        else:
            self.sample_rate = []
            self.cum_samples = []

        print(f"データセット名: {self.datasets_names}")
        print(f"サンプリング率: {self.sample_rate}")
        print(f"累積サンプリング率: {self.cum_samples}")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # ランダムにデータセットを選択
        r = random.random()
        dataset_idx = 0
        for i, rate in enumerate(self.cum_samples):
            if r <= rate:
                dataset_idx = i
                break
                
        # 選択されたデータセットからサンプルを取得
        if dataset_idx == 0 and self.sem_seg_dataset is not None:
            data = self.sem_seg_dataset[random.randint(0, self.sem_seg_dataset_length - 1)]
        elif dataset_idx == 1 and self.refer_seg_dataset is not None:
            data = self.refer_seg_dataset[random.randint(0, self.refer_seg_dataset_length - 1)]
        elif dataset_idx == 2 and self.vqa_dataset is not None:
            data = self.vqa_dataset[random.randint(0, self.vqa_dataset_length - 1)]
        elif dataset_idx == 3 and self.reason_seg_dataset is not None:
            data = self.reason_seg_dataset[random.randint(0, self.reason_seg_dataset_length - 1)]
        else:
            # フォールバック: 利用可能な最初のデータセットを使用
            if self.sem_seg_dataset is not None:
                data = self.sem_seg_dataset[random.randint(0, self.sem_seg_dataset_length - 1)]
            elif self.refer_seg_dataset is not None:
                data = self.refer_seg_dataset[random.randint(0, self.refer_seg_dataset_length - 1)]
            elif self.vqa_dataset is not None:
                data = self.vqa_dataset[random.randint(0, self.vqa_dataset_length - 1)]
            elif self.reason_seg_dataset is not None:
                data = self.reason_seg_dataset[random.randint(0, self.reason_seg_dataset_length - 1)]
            else:
                raise ValueError("No dataset available")
                
        return data


class ValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
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
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

    def __len__(self):
        if self.data_type == "refer_seg":
            return len(self.refer_seg_ds["images"])
        else:
            return len(self.images)

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
                raise ValueError("image {} has no refs".format(image_id))

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

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        i = 0
        while i < len(sampled_sents):
            conv.messages = []
            text = sampled_sents[i].strip()
            if is_sentence:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n {} Please output segmentation mask.".format(text),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            else:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n What is {} in this image? Please output segmentation mask.".format(
                        text
                    ),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            conversations.append(conv.get_prompt())
            i += 1

        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        # preprocess image for sam
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

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
                m = np.sum(
                    m, axis=2
                )  # sometimes there are multiple binary map (corresponding to multiple segs)
                m = m.astype(np.uint8)  # convert to np.uint8
                masks.append(m)
        else:
            masks = [mask_json]

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            labels,
            resize,
            None,
            None,
            inference,
        )

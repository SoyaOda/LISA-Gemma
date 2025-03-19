import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
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

from .grefer import G_REFER
from .refer import REFER
from .utils import ANSWER_LIST, SHORT_QUESTION_LIST
from .conversation import get_default_conv_template


class ReferSegDataset(torch.utils.data.Dataset):
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

        # 質問と回答のテンプレート
        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        # データセットの初期化
        DATA_DIR = os.path.join(self.base_image_dir, "refer_seg")
        self.refer_seg_ds_list = args.refer_seg_data.split("||") if hasattr(args, 'refer_seg_data') and args.refer_seg_data else []
        self.refer_seg_data = {}
        
        for ds in self.refer_seg_ds_list:
            if ds == "refcocog":
                splitBy = "umd"
            else:
                splitBy = "unc"

            if ds == "grefcoco":
                refer_api = G_REFER(DATA_DIR, ds, splitBy)
            else:
                refer_api = REFER(DATA_DIR, ds, splitBy)
                
            ref_ids_train = refer_api.getRefIds(split="train")
            images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)
            refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)

            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_train)

            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        DATA_DIR, "images/saiapr_tc-12", item["file_name"]
                    )
                else:
                    item["file_name"] = os.path.join(
                        DATA_DIR, "images/mscoco/images/train2014", item["file_name"]
                    )
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_train

            print(
                "データセット {} (refs {}) (train split) - 画像: {} 件、アノテーション: {} 件".format(
                    ds,
                    splitBy,
                    len(refer_seg_ds["images"]),
                    len(refer_seg_ds["annotations"]),
                )
            )

            img2refs = {}
            for ref in refs_train:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_data[ds] = refer_seg_ds

    def __len__(self):
        return self.samples_per_epoch

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
        # データセットが空の場合は処理をスキップ
        if len(self.refer_seg_ds_list) == 0:
            raise ValueError("参照セグメンテーションデータセットが初期化されていません")
            
        # ランダムにデータセットを選択
        ds = random.randint(0, len(self.refer_seg_ds_list) - 1)
        ds = self.refer_seg_ds_list[ds]
        refer_seg_ds = self.refer_seg_data[ds]
        
        # データを取得
        images = refer_seg_ds["images"]
        annotations = refer_seg_ds["annotations"]
        img2refs = refer_seg_ds["img2refs"]
        
        # ランダムに画像を選択
        idx = random.randint(0, len(images) - 1)
        image_info = images[idx]
        image_path = image_info["file_name"]
        image_id = image_info["id"]
        
        # 参照文を取得
        refs = img2refs[image_id]
        if len(refs) == 0:
            return self.__getitem__(0)

        # 文章とアノテーションIDを収集
        sents = []
        ann_ids = []
        for ref in refs:
            for sent in ref["sentences"]:
                text = sent["sent"]
                sents.append(text)
                ann_ids.append(ref["ann_id"])
                
        # サンプル数の調整
        if len(sents) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(sents))), size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(sents)))
            
        # サンプリングされた文章とアノテーションIDを取得
        sampled_sents = [sents[ind] for ind in sampled_inds]
        sampled_ann_ids = [ann_ids[ind] for ind in sampled_inds]
        sampled_classes = sampled_sents
        
        # 画像を読み込み
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 画像の前処理
        if hasattr(self, 'processor') and self.processor is not None:
            # Gemma3のプロセッサを使用
            image_clip = self.processor.image_processor(image, return_tensors="pt").pixel_values[0]
        else:
            # CLIPイメージプロセッサを使用
            image_clip = self.clip_image_processor.preprocess(
                image, return_tensors="pt"
            )["pixel_values"][0]
        
        # SAM用の画像前処理
        image_sam = self.transform.apply_image(image)
        resize = image_sam.shape[:2]

        # 会話テンプレートの作成
        conversations = []
        questions = []
        
        for text in sampled_classes:
            # テキストの前処理
            text = text.strip()
            
            # 会話テンプレートの選択
            if self.is_gemma3:
                conv = get_default_conv_template(self.conv_type)
            else:
                if LLAVA_AVAILABLE:
                    conv = conversation_lib.default_conversation.copy()
                else:
                    conv = get_default_conv_template(self.conv_type)
            
            # 質問の作成
            question_idx = random.randint(0, len(self.short_question_list) - 1)
            question = self.short_question_list[question_idx].format(text.lower())
            questions.append(question)
            
            # 回答の作成
            answer_idx = random.randint(0, len(self.answer_list) - 1)
            answer = self.answer_list[answer_idx]
            
            # プロンプトの生成
            if self.is_gemma3:
                # Gemma3形式のプロンプト
                conv.append_message(conv.roles[0], "<image>\n" + question)
                conv.append_message(conv.roles[1], answer)
            else:
                # LLaVA形式のプロンプト
                conv.append_message(conv.roles[0], "<image>\n" + question)
                conv.append_message(conv.roles[1], answer)
            
            conversations.append(conv.get_prompt())

        # SAM用の画像をテンソルに変換
        image_sam = self.preprocess(torch.from_numpy(image_sam).permute(2, 0, 1).contiguous())

        # マスクの処理
        masks = []
        for ann_id in sampled_ann_ids:
            if isinstance(ann_id, list):
                # 複数のアノテーションIDがある場合
                if -1 in ann_id:
                    # -1は無効なアノテーションを示す
                    assert len(ann_id) == 1
                    m = np.zeros((image_info["height"], image_info["width"])).astype(
                        np.uint8
                    )
                else:
                    # 複数のアノテーションを組み合わせる
                    m_final = np.zeros(
                        (image_info["height"], image_info["width"])
                    ).astype(np.uint8)
                    for ann_id_i in ann_id:
                        ann = annotations[ann_id_i]

                        if len(ann["segmentation"]) == 0:
                            m = np.zeros(
                                (image_info["height"], image_info["width"])
                            ).astype(np.uint8)
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
                        m_final = m_final | m
                    m = m_final
                masks.append(m)
                continue

            # 単一のアノテーションID
            ann = annotations[ann_id]

            if len(ann["segmentation"]) == 0:
                m = np.zeros((image_info["height"], image_info["width"])).astype(
                    np.uint8
                )
                masks.append(m)
                continue

            if type(ann["segmentation"][0]) == list:  # polygon
                rle = mask.frPyObjects(
                    ann["segmentation"], image_info["height"], image_info["width"]
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

        # マスクをテンソルに変換
        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        
        # ラベルの準備
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        
        # 学習/推論モードのフラグ
        inference = False
        
        return image_path, image_sam, image_clip, conversations, masks, label, resize, questions, sampled_classes, inference

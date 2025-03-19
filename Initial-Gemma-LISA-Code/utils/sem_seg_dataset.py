import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
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
from .utils import ANSWER_LIST, SHORT_QUESTION_LIST
from .conversation import get_default_conv_template

from .utils import ANSWER_LIST, SHORT_QUESTION_LIST


def init_mapillary(base_image_dir):
    mapillary_data_root = os.path.join(base_image_dir, "mapillary")
    with open(os.path.join(mapillary_data_root, "config_v2.0.json")) as f:
        mapillary_classes = json.load(f)["labels"]
    mapillary_classes = [x["readable"].lower() for x in mapillary_classes]
    mapillary_classes = np.array(mapillary_classes)
    mapillary_labels = sorted(
        glob.glob(
            os.path.join(mapillary_data_root, "training", "v2.0", "labels", "*.png")
        )
    )
    mapillary_images = [
        x.replace(".png", ".jpg").replace("v2.0/labels", "images")
        for x in mapillary_labels
    ]
    print("mapillary: ", len(mapillary_images))
    return mapillary_classes, mapillary_images, mapillary_labels


def init_ade20k(base_image_dir):
    with open("utils/ade20k_classes.json", "r") as f:
        ade20k_classes = json.load(f)
    ade20k_classes = np.array(ade20k_classes)
    image_ids = sorted(
        os.listdir(os.path.join(base_image_dir, "ade20k/images", "training"))
    )
    ade20k_image_ids = []
    for x in image_ids:
        if x.endswith(".jpg"):
            ade20k_image_ids.append(x[:-4])
    ade20k_images = []
    for image_id in ade20k_image_ids:  # self.descriptions:
        ade20k_images.append(
            os.path.join(
                base_image_dir,
                "ade20k",
                "images",
                "training",
                "{}.jpg".format(image_id),
            )
        )
    ade20k_labels = [
        x.replace(".jpg", ".png").replace("images", "annotations")
        for x in ade20k_images
    ]
    print("ade20k: ", len(ade20k_images))
    return ade20k_classes, ade20k_images, ade20k_labels


def init_cocostuff(base_image_dir):
    cocostuff_classes = []
    with open("utils/cocostuff_classes.txt") as f:
        for line in f.readlines()[1:]:
            cocostuff_classes.append(line.strip().split(": ")[-1])
    cocostuff_classes = np.array(cocostuff_classes)
    cocostuff_images = []

    cocostuff_labels = glob.glob(
        os.path.join(base_image_dir, "cocostuff", "train2017", "*.png")
    )
    cocostuff_images = [
        x.replace(".png", ".jpg").replace("cocostuff", "coco") for x in cocostuff_labels
    ]

    print("cocostuff: ", len(cocostuff_images))
    return cocostuff_classes, cocostuff_images, cocostuff_labels


def init_paco_lvis(base_image_dir):
    coco_api_paco_lvis = COCO(
        os.path.join(
            base_image_dir, "vlpart", "paco", "annotations", "paco_lvis_v1_train.json"
        )
    )
    all_classes = coco_api_paco_lvis.loadCats(coco_api_paco_lvis.getCatIds())
    class_map_paco_lvis = {}
    for cat in all_classes:
        cat_split = cat["name"].strip().split(":")
        if len(cat_split) == 1:
            name = cat_split[0].split("_(")[0]
        else:
            assert len(cat_split) == 2
            obj, part = cat_split
            obj = obj.split("_(")[0]
            part = part.split("_(")[0]
            name = (obj, part)
        class_map_paco_lvis[cat["id"]] = name
    img_ids = coco_api_paco_lvis.getImgIds()
    print("paco_lvis: ", len(img_ids))
    return class_map_paco_lvis, img_ids, coco_api_paco_lvis


def init_pascal_part(base_image_dir):
    coco_api_pascal_part = COCO(
        os.path.join(base_image_dir, "vlpart", "pascal_part", "train.json")
    )
    all_classes = coco_api_pascal_part.loadCats(coco_api_pascal_part.getCatIds())
    class_map_pascal_part = {}
    for cat in all_classes:
        cat_main, cat_part = cat["name"].strip().split(":")
        name = (cat_main, cat_part)
        class_map_pascal_part[cat["id"]] = name
    img_ids = coco_api_pascal_part.getImgIds()
    print("pascal_part: ", len(img_ids))
    return class_map_pascal_part, img_ids, coco_api_pascal_part


class SemSegDataset(torch.utils.data.Dataset):
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
        self.exclude_val = args.exclude_val
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

        self.data2list = {}
        self.data2classes = {}

        # データセットの初期化
        self.sem_seg_datas = args.sem_seg_data.split("||") if hasattr(args, 'sem_seg_data') and args.sem_seg_data else []
        for ds in self.sem_seg_datas:
            classes, images, labels = eval("init_{}".format(ds))(self.base_image_dir)
            self.data2list[ds] = (images, labels)
            self.data2classes[ds] = classes

        if "cocostuff" in self.sem_seg_datas:
            self.cocostuff_class2index = {
                c: i for i, c in enumerate(self.data2classes["cocostuff"])
            }

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
        # データセットをランダムに選択
        ds = random.randint(0, len(self.sem_seg_datas) - 1)
        ds = self.sem_seg_datas[ds]

        # データロード処理（既存コードと同じ）
        if ds in ["paco_lvis", "pascal_part"]:
            class_map = self.data2classes[ds]
            img_ids, coco_api = self.data2list[ds]
            idx = random.randint(0, len(img_ids) - 1)
            img_id = img_ids[idx]
            image_info = coco_api.loadImgs([img_id])[0]
            file_name = image_info["file_name"]
            if ds == "pascal_part":
                file_name = os.path.join(
                    "VOCdevkit", "VOC2010", "JPEGImages", file_name
                )
                image_path = os.path.join(self.base_image_dir, "vlpart", ds, file_name)
            elif ds == "paco_lvis":
                image_path = os.path.join(self.base_image_dir, "coco", file_name)
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
            
            # アノテーションの取得
            annIds = coco_api.getAnnIds(imgIds=image_info["id"])
            anns = coco_api.loadAnns(annIds)
            if len(anns) == 0:
                return self.__getitem__(0)
            
            # サンプル数の調整
            if len(anns) >= self.num_classes_per_sample:
                sampled_anns = np.random.choice(
                    anns, size=self.num_classes_per_sample, replace=False
                ).tolist()
            else:
                sampled_anns = anns
                
            # クラス名の取得
            sampled_classes = []
            for ann in sampled_anns:
                sampled_cls = class_map[ann["category_id"]]
                if isinstance(sampled_cls, tuple):
                    obj, part = sampled_cls
                    if random.random() < 0.5:
                        name = obj + " " + part
                    else:
                        name = "the {} of the {}".format(part, obj)
                else:
                    name = sampled_cls
                sampled_classes.append(name)

        # ade20k, cocostuff, mapillaryデータセットの処理（既存コードと同じ）
        elif ds in ["ade20k", "cocostuff", "mapillary"]:
            image, labels = self.data2list[ds]
            idx = random.randint(0, len(image) - 1)
            image_path = image[idx]
            label_path = labels[idx]
            label = Image.open(label_path)
            label = np.array(label)
            if ds == "ade20k":
                label[label == 0] = 255
                label -= 1
                label[label == 254] = 255
            elif ds == "cocostuff":
                for c, i in self.cocostuff_class2index.items():
                    if "-" in c:
                        label[label == i] = 255
            elif ds == "mapillary":
                # マップラリーデータの処理
                mapping_cocostuff_to_mapillary = {
                    83: 20,  # 'bus': 'bus',
                    41: 22,  # 'car': 'car',
                    179: 24,  # 'pavement': 'road',
                    125: 23,  # 'person': 'person',
                    65: 21,  # 'truck': 'truck',
                }
                segmentation = np.ones(label.shape).astype(np.uint8) * 255
                for s, t in mapping_cocostuff_to_mapillary.items():
                    mask = label == t
                    segmentation[mask] = 254
                label = segmentation

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
            
            # ランダムにクラスをサンプリング
            classes = self.data2classes[ds]
            unique_labels = np.unique(label)
            unique_labels = [l for l in unique_labels if l != 255]
            
            # 十分なラベルがない場合は別のサンプルを取得
            if len(unique_labels) == 0:
                return self.__getitem__(0)
                
            # クラスのサンプリング
            if len(unique_labels) >= self.num_classes_per_sample:
                sampled_cls_idx = np.random.choice(
                    unique_labels, size=self.num_classes_per_sample, replace=False
                ).tolist()
            else:
                sampled_cls_idx = unique_labels
                
            # クラス名の取得
            sampled_classes = [classes[i] for i in sampled_cls_idx]
            
            # マスク画像の作成
            masks = []
            for i, cls_idx in enumerate(sampled_cls_idx):
                mask = label == cls_idx
                masks.append(mask)
                
            # マスクの結合
            masks = np.stack(masks, axis=0)

        # 会話テンプレートの作成
        conversations = []
        
        for i, sampled_class in enumerate(sampled_classes):
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
            question = self.short_question_list[question_idx].format(sampled_class)
            
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
            
        # 出力データの準備
        if ds in ["paco_lvis", "pascal_part"]:
            masks = []
            for ann in sampled_anns:
                if type(ann["segmentation"]) == list:  # polygon
                    rle = mask.frPyObjects(
                        ann["segmentation"],
                        image_info["height"],
                        image_info["width"],
                    )
                else:
                    rle = ann["segmentation"]
                m = mask.decode(rle)
                if len(m.shape) == 3:
                    m = np.sum(m, axis=2)  # multi-part segments
                m = m.astype(np.uint8)
                masks.append(m)
            masks = np.stack(masks, axis=0)
        
        # SAM用の画像をテンソルに変換
        image_sam = self.preprocess(torch.from_numpy(image_sam).permute(2, 0, 1).contiguous())
        
        # マスクをテンソルに変換
        masks = torch.from_numpy(masks)
        
        # ラベルの準備
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        
        # 学習/推論モードのフラグ
        inference = False
        
        return image_path, image_sam, image_clip, conversations, masks, label, resize, question, sampled_classes, inference

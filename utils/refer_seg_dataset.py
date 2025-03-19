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
from .constants import ANSWER_LIST, SHORT_QUESTION_LIST, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, SYSTEM_PROMPT
from .conversation import get_default_conv_template


class ReferSegDataset(torch.utils.data.Dataset):
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
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
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
            refer_seg_data: 参照セグメンテーションデータ
        """
        self.base_image_dir = base_image_dir
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.precision = precision
        self.image_size = image_size
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample
        self.exclude_val = exclude_val
        
        # データセットの初期化
        self.all_datasets = refer_seg_data.split("||")
        
        # 各データセットの設定
        self.images = []
        self.refs_per_image = {}
        
        # 初期化関数
        for ds in self.all_datasets:
            # 利用可能なデータセット：refclef, refcoco, refcoco+, refcocog
            if ds == "refclef":
                self.init_refclef()
            elif ds == "refcoco":
                self.init_refcoco()
            elif ds == "refcoco+":
                self.init_refcocoplus()
            elif ds == "refcocog":
                self.init_refcocog()
        
        print(f"合計画像数: {len(self.images)}")
        
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
    
    def init_refclef(self):
        """RefClefデータセットを初期化"""
        refer_api = REFER(self.base_image_dir, "refclef", "unc")
        
        # 参照IDを取得
        ref_ids = refer_api.getRefIds(split="train")
        
        # valデータが含まれないようにする場合
        if self.exclude_val:
            val_ref_ids = refer_api.getRefIds(split="val")
            ref_ids = [ref_id for ref_id in ref_ids if ref_id not in val_ref_ids]
        
        # 画像IDを取得
        img_ids = refer_api.getImgIds(ref_ids=ref_ids)
        
        # 参照情報を取得
        refs = refer_api.loadRefs(ref_ids=ref_ids)
        
        # 画像ごとの参照を整理
        img2refs = {}
        for ref in refs:
            img_id = ref["image_id"]
            if img_id in img2refs:
                img2refs[img_id].append(ref)
            else:
                img2refs[img_id] = [ref]
        
        # 画像情報を取得
        images = refer_api.loadImgs(img_ids)
        
        # 画像パスとアノテーションを保存
        for image in images:
            image_path = os.path.join(
                self.base_image_dir, "images/saiapr_tc-12", image["file_name"]
            )
            self.images.append(image_path)
            self.refs_per_image[image_path] = img2refs[image["id"]]
        
        print(f"RefClef: {len(self.images)}画像")
    
    def init_refcoco(self):
        """RefCOCOデータセットを初期化"""
        refer_api = REFER(self.base_image_dir, "refcoco", "unc")
        
        # 参照IDを取得
        ref_ids = refer_api.getRefIds(split="train")
        
        # valデータが含まれないようにする場合
        if self.exclude_val:
            val_ref_ids = refer_api.getRefIds(split="val")
            ref_ids = [ref_id for ref_id in ref_ids if ref_id not in val_ref_ids]
        
        # 画像IDを取得
        img_ids = refer_api.getImgIds(ref_ids=ref_ids)
        
        # 参照情報を取得
        refs = refer_api.loadRefs(ref_ids=ref_ids)
        
        # 画像ごとの参照を整理
        img2refs = {}
        for ref in refs:
            img_id = ref["image_id"]
            if img_id in img2refs:
                img2refs[img_id].append(ref)
            else:
                img2refs[img_id] = [ref]
        
        # 画像情報を取得
        images = refer_api.loadImgs(img_ids)
        
        # 画像パスとアノテーションを保存
        for image in images:
            image_path = os.path.join(
                self.base_image_dir, "images/mscoco/images/train2014", image["file_name"]
            )
            self.images.append(image_path)
            self.refs_per_image[image_path] = img2refs[image["id"]]
        
        print(f"RefCOCO: {len(self.images)}画像")
    
    def init_refcocoplus(self):
        """RefCOCO+データセットを初期化"""
        refer_api = REFER(self.base_image_dir, "refcoco+", "unc")
        
        # 参照IDを取得
        ref_ids = refer_api.getRefIds(split="train")
        
        # valデータが含まれないようにする場合
        if self.exclude_val:
            val_ref_ids = refer_api.getRefIds(split="val")
            ref_ids = [ref_id for ref_id in ref_ids if ref_id not in val_ref_ids]
        
        # 画像IDを取得
        img_ids = refer_api.getImgIds(ref_ids=ref_ids)
        
        # 参照情報を取得
        refs = refer_api.loadRefs(ref_ids=ref_ids)
        
        # 画像ごとの参照を整理
        img2refs = {}
        for ref in refs:
            img_id = ref["image_id"]
            if img_id in img2refs:
                img2refs[img_id].append(ref)
            else:
                img2refs[img_id] = [ref]
        
        # 画像情報を取得
        images = refer_api.loadImgs(img_ids)
        
        # 画像パスとアノテーションを保存
        for image in images:
            image_path = os.path.join(
                self.base_image_dir, "images/mscoco/images/train2014", image["file_name"]
            )
            self.images.append(image_path)
            self.refs_per_image[image_path] = img2refs[image["id"]]
        
        print(f"RefCOCO+: {len(self.images)}画像")
    
    def init_refcocog(self):
        """RefCOCOgデータセットを初期化"""
        refer_api = REFER(self.base_image_dir, "refcocog", "umd")
        
        # 参照IDを取得
        ref_ids = refer_api.getRefIds(split="train")
        
        # valデータが含まれないようにする場合
        if self.exclude_val:
            val_ref_ids = refer_api.getRefIds(split="val")
            ref_ids = [ref_id for ref_id in ref_ids if ref_id not in val_ref_ids]
        
        # 画像IDを取得
        img_ids = refer_api.getImgIds(ref_ids=ref_ids)
        
        # 参照情報を取得
        refs = refer_api.loadRefs(ref_ids=ref_ids)
        
        # 画像ごとの参照を整理
        img2refs = {}
        for ref in refs:
            img_id = ref["image_id"]
            if img_id in img2refs:
                img2refs[img_id].append(ref)
            else:
                img2refs[img_id] = [ref]
        
        # 画像情報を取得
        images = refer_api.loadImgs(img_ids)
        
        # 画像パスとアノテーションを保存
        for image in images:
            image_path = os.path.join(
                self.base_image_dir, "images/mscoco/images/train2014", image["file_name"]
            )
            self.images.append(image_path)
            self.refs_per_image[image_path] = img2refs[image["id"]]
        
        print(f"RefCOCOg: {len(self.images)}画像")
    
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
        # ランダムに画像を選択
        image_path = random.choice(self.images)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 参照を取得
        refs = self.refs_per_image[image_path]
        
        # ランダムに参照を選択
        if len(refs) >= self.num_classes_per_sample:
            sampled_refs = random.sample(refs, self.num_classes_per_sample)
        else:
            sampled_refs = refs
        
        # 文章と注釈IDを抽出
        sents = []
        ann_ids = []
        for ref in sampled_refs:
            for sent in ref["sentences"]:
                sents.append(sent["sent"].strip().lower())
                ann_ids.append(ref["ann_id"])
        
        # 会話を生成
        conversations = []
        
        # Gemma3用のテンプレート取得
        template = get_default_conv_template("gemma_v1")
        template.system = SYSTEM_PROMPT
        
        # 各文章について会話を生成
        for text in sents:
            template.messages = []
            
            # ユーザーからの質問を生成
            query = f"{DEFAULT_IMAGE_TOKEN}\n{text} Please output segmentation mask."
            template.append_message(template.roles[0], query)
            
            # アシスタントの回答を生成
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
        masks = []
        for ann_id in ann_ids:
            # 対応するアノテーションを見つける
            ann = None
            for ref in sampled_refs:
                if ref["ann_id"] == ann_id:
                    ann = ref["ann"]
                    break
            
            if ann is None:
                # アノテーションが見つからない場合は空のマスク
                m = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            else:
                # RLEからマスクを復元
                if type(ann["segmentation"]) == list:  # ポリゴン
                    rle = mask.frPyObjects(
                        ann["segmentation"],
                        ann["height"],
                        ann["width"],
                    )
                else:
                    rle = ann["segmentation"]
                    # バイトエンコードを確認
                    if not isinstance(rle["counts"], bytes):
                        rle["counts"] = rle["counts"].encode()
                
                m = mask.decode(rle)
                if len(m.shape) == 3:
                    m = np.sum(m, axis=2)  # 複数パートのセグメントを合成
                
                m = m.astype(np.uint8)
            
            masks.append(m)
        
        # マスクをテンソルに変換
        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        
        # ダミーラベルを作成
        label = torch.ones(image.shape[0], image.shape[1]) * self.ignore_label
        
        # 推論フラグと質問のプレースホルダー
        inference = False
        questions = None
        sampled_classes = None
        
        return (
            image_path,
            image_sam,
            images_gemma,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_classes,
            inference,
        )

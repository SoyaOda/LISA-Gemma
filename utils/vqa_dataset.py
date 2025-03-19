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

from .constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, SYSTEM_PROMPT
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
    """ビジュアルQAデータセット"""
    
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
        vqa_data="llava_instruct_150k",
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
            vqa_data: VQAデータセット名
        """
        self.base_image_dir = base_image_dir
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        
        # データセットの初期化
        self.all_datasets = vqa_data.split("||")
        
        # 各データセットの設定
        self.images = []
        self.questions = []
        self.answers = []
        
        for ds in self.all_datasets:
            if ds == "llava_instruct_150k":
                self.init_llava_dataset()
        
        print(f"VQAデータセット合計: {len(self.questions)}件")
        
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
    
    def init_llava_dataset(self):
        """LLaVAインストラクションデータセットを初期化"""
        # データセットパス
        dataset_path = os.path.join(self.base_image_dir, "llava_dataset", "llava_instruct_150k.json")
        print(f"LLaVAデータセットのロード: {dataset_path}")
        
        with open(dataset_path, "r") as f:
            data = json.load(f)
        
        # 質問と回答を抽出
        for item in data:
            if "image" in item and item["image"]:
                image_path = os.path.join(self.base_image_dir, item["image"])
                
                # 画像が存在するか確認
                if os.path.exists(image_path):
                    conversations = item["conversations"]
                    if len(conversations) >= 2:
                        question = conversations[0]["value"]
                        answer = conversations[1]["value"]
                        
                        # 画像参照記号を含むか確認
                        if "<image>" in question:
                            self.images.append(image_path)
                            self.questions.append(question)
                            self.answers.append(answer)
    
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
        # ランダムに画像・質問・回答を選択
        idx = random.randint(0, len(self.images) - 1)
        image_path = self.images[idx]
        question = self.questions[idx]
        answer = self.answers[idx]
        
        # 画像を読み込み
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 会話を生成
        conversations = []
        
        # Gemma3用のテンプレート取得
        template = get_default_conv_template("gemma_v1")
        template.system = SYSTEM_PROMPT
        
        # 質問に<image>トークンがある場合は置き換え
        question = question.replace("<image>", DEFAULT_IMAGE_TOKEN)
        
        # メッセージを追加
        template.messages = []
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], answer)
        
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
        
        # VQAではマスクは不要なので空のテンソルを用意
        masks = torch.zeros(1, resize[0], resize[1])
        
        # ダミーラベルを作成
        label = torch.ones(image.shape[0], image.shape[1]) * self.ignore_label
        
        # 推論フラグと質問のプレースホルダー
        inference = False
        questions = question
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

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass
import math

import transformers
from transformers import PreTrainedModel, GenerationMixin
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers import Gemma3Config, Gemma3TextConfig, Gemma3ForConditionalGeneration
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide

from model.gemma3.constants import (DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN,
                                 DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN,
                                 IMAGE_TOKEN_INDEX, IGNORE_INDEX)
from model.segment_anything import build_sam_vit_h, sam_model_registry
from model.segment_anything.modeling import Sam


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    DICE損失を計算します（マスク用のIOU損失に類似）
    Args:
        inputs: 任意の形状の浮動小数点テンソル
                各サンプルの予測値
        targets: inputsと同じ形状の浮動小数点テンソル
                各要素のバイナリ分類ラベルを格納
                (0: ネガティブクラス、1: ポジティブクラス)
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    シグモイドクロスエントロピー損失を計算します
    Args:
        inputs: 任意の形状の浮動小数点テンソル
                各サンプルの予測値
        targets: inputsと同じ形状の浮動小数点テンソル
                各要素のバイナリ分類ラベルを格納
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.mean(1).sum() / (num_masks + 1e-8)
    return loss


class GemmaLISAMetaModel(nn.Module):
    """GemmaとLISAのメタモデル"""
    
    def __init__(self):
        super().__init__()
        self.seg_token_idx = None
        self.visual_model = None
        self.text_hidden_fcs = None

    def initialize_vision_modules(
        self, 
        vision_tower: str,
        mm_vision_select_layer: int,
        pretrain_mm_mlp_adapter: Optional[str] = None,
        mm_projector_type: Optional[str] = None,
        tune_mm_mlp_adapter: bool = False,
        freeze_vision_tower: bool = True,
    ):
        """Vision Tower（画像エンコーダ）を初期化
        
        Gemma3では内部に画像エンコーダを持っているため、
        この関数は必要最低限の処理のみを行います
        """
        # Gemma3では視覚エンコーダが内蔵されているため特に処理は不要
        pass

    def initialize_lisa_modules(
        self,
        config,
        vision_pretrained=None,
        freeze_lm=True,
        freeze_vision_encoder=True,
        freeze_mask_decoder=False,
        out_dim=256,
        train_mask_decoder=True,
        **kwargs,
    ):
        """
        LISAモジュールを初期化します。
        
        Args:
            config: モデル設定
            vision_pretrained: SAMモデルの事前学習済み重みへのパス
            freeze_lm: 言語モデル部分を凍結するかどうか
            freeze_vision_encoder: 視覚エンコーダを凍結するかどうか
            freeze_mask_decoder: マスクデコーダを凍結するかどうか
            out_dim: テキスト→SAMプロンプト変換層の出力次元
            train_mask_decoder: マスクデコーダを学習するかどうか
        
        Returns:
            self: 初期化されたモデル
        """
        if vision_pretrained is None:
            raise ValueError("SAMモデルのパスを指定してください (vision_pretrained)")
            
        # SAMモデルをロード
        print(f"SAM vit-h モデルを {vision_pretrained} からロードしています...")
        if not os.path.exists(vision_pretrained):
            raise FileNotFoundError(
                f"SAMモデルの重みファイル {vision_pretrained} が見つかりません。"
                "正しいパスを指定してください。"
            )
        
        # SAMモデルの構築
        self.visual_model = build_sam_vit_h(vision_pretrained)
        
        # configの型を検出して適切に隠れ層次元を取得
        print(f"設定オブジェクトの型: {type(config).__name__}")
        
        if hasattr(config, "text_config"):
            # Gemma3Configの場合（マルチモーダル設定）
            hidden_size = getattr(config.text_config, "hidden_size", 4096)
            print(f"Gemma3Configからtext_config.hidden_size={hidden_size}を取得")
        else:
            # すでにGemma3TextConfigの場合（テキスト設定のみ）
            hidden_size = getattr(config, "hidden_size", 4096)
            print(f"設定から直接hidden_size={hidden_size}を取得")
        
        # テキスト埋め込みからSAMプロンプト埋め込みへの変換モジュール
        # オリジナルLISAでは複数層のMLPを使用
        self.text_hidden_fcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.GELU(),
                nn.Linear(hidden_size // 2, out_dim),
                nn.LayerNorm(out_dim)
            )
        ])
        
        # SAMモデルのコンポーネントを直接アクセスできるようにする
        self.image_encoder = self.visual_model.image_encoder
        self.prompt_encoder = self.visual_model.prompt_encoder
        self.mask_decoder = self.visual_model.mask_decoder
        
        # フリーズ設定
        # 1. 視覚エンコーダのフリーズ
        if freeze_vision_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        
        # 2. マスクデコーダの設定
        if not train_mask_decoder or freeze_mask_decoder:
            for param in self.mask_decoder.parameters():
                param.requires_grad = False
        
        # 3. 言語モデル部分のフリーズ
        if freeze_lm and hasattr(self, "gemma_model"):
            for param in self.gemma_model.parameters():
                param.requires_grad = False
        
        # 初期化情報の出力
        print(f"SAMモデルとテキスト射影層の初期化が完了しました")
        print(f"  隠れ層次元: {hidden_size}")
        print(f"  出力次元: {out_dim}")
        print(f"  視覚エンコーダフリーズ: {freeze_vision_encoder}")
        print(f"  マスクデコーダフリーズ: {freeze_mask_decoder}")
        print(f"  言語モデルフリーズ: {freeze_lm}")
        
        return self

    def get_visual_embs(self, x):
        """SAMの視覚エンコーダを使用して画像埋め込みを取得"""
        # SAMの画像エンコーダを使用
        return self.visual_model.image_encoder(x)


class LISAPreTrainedModel(PreTrainedModel, GenerationMixin):
    """
    LISA用のPreTrainedModelの抽象クラス
    """
    config_class = AutoConfig
    base_model_prefix = "gemma_model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SamEncoder", "MaskDecoder"]

    def _init_weights(self, module):
        """モデルの重みを初期化する関数"""
        # 実装なし - ベースモデルが初期化を行う
        pass


class LISAModel(GemmaLISAMetaModel, PreTrainedModel):
    """LISAのメインモデルクラス"""
    
    config_class = AutoConfig
    
    def __init__(self, config):
        super(LISAModel, self).__init__()
        self.config = config

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # configとベースモデルを取得するため、辞書からビジョン関連の引数を抽出
        seg_token_idx = kwargs.pop("seg_token_idx", None)
        vision_pretrained = kwargs.pop("vision_pretrained", None)
        freeze_model = kwargs.pop("freeze_model", True)
        freeze_maskdecoder = kwargs.pop("freeze_maskdecoder", True)
        out_dim = kwargs.pop("out_dim", 256)
        
        # Gemma3モデルをロード
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        
        # configにtext_configがあればvocab_sizeなどを直接アクセス可能にする
        if hasattr(config, 'text_config'):
            # text_configから主要なパラメータをトップレベルにコピー
            for key, value in vars(config.text_config).items():
                setattr(config, key, value)
        
        model = super().from_pretrained(pretrained_model_name_or_path, config=config, *model_args, **kwargs)
        
        # LISA特有の設定
        model.seg_token_idx = seg_token_idx
        model.initialize_lisa_modules(
            config=config.text_config if hasattr(config, 'text_config') else config,
            vision_pretrained=vision_pretrained,
            freeze_model=freeze_model,
            freeze_maskdecoder=freeze_maskdecoder,
            out_dim=out_dim,
        )
        
        return model


class LISAForCausalLM(LISAPreTrainedModel):
    """LISA生成モデル"""
    
    def __init__(self, config, seg_token_idx=None, vision_pretrained=None, out_dim=256, train_mask_decoder=False):
        super().__init__(config)
        self.config = config
        self.seg_token_idx = seg_token_idx
        self.visual_model = None  # SAMモデル
        self.text_hidden_fcs = None  # テキスト→SAMプロンプト変換層
        self.train_mask_decoder = train_mask_decoder
        
        # 損失関数の重みを明示的に初期化
        self.ce_loss_weight = None
        self.dice_loss_weight = None
        self.bce_loss_weight = None
        
        # Gemmaモデルが渡された場合は登録
        if isinstance(config, (Gemma3Config, Gemma3TextConfig)):
            self.gemma_model = Gemma3ForConditionalGeneration(config)
        
        # SAMモデルを初期化する場合
        if vision_pretrained is not None:
            self.initialize_lisa_modules(
                config=config.text_config if hasattr(config, 'text_config') else config,
                vision_pretrained=vision_pretrained,
                out_dim=out_dim,
                train_mask_decoder=train_mask_decoder
            )

    def initialize_lisa_modules(
        self,
        config,
        vision_pretrained=None,
        freeze_model=True,
        freeze_maskdecoder=True,
        out_dim=256,
        train_mask_decoder=False,
        **kwargs,
    ):
        """LISAモジュールを初期化"""
        if vision_pretrained is None:
            raise ValueError("SAMモデルのパスを指定してください (vision_pretrained)")
            
        # SAMモデルをロード
        print(f"SAM vit-h モデルを {vision_pretrained} からロードしています...")
        if not os.path.exists(vision_pretrained):
            raise FileNotFoundError(
                f"SAMモデルの重みファイル {vision_pretrained} が見つかりません。"
                "正しいパスを指定してください。"
            )
            
        # SAMモデルの構築
        self.visual_model = build_sam_vit_h(vision_pretrained)
        
        # SAMモデルのコンポーネントを直接アクセスできるようにする
        self.image_encoder = self.visual_model.image_encoder
        self.prompt_encoder = self.visual_model.prompt_encoder
        self.mask_decoder = self.visual_model.mask_decoder
        
        # マスクデコーダーの学習可否設定
        if not train_mask_decoder:
            if freeze_model:
                for name, param in self.visual_model.named_parameters():
                    param.requires_grad = False
            elif freeze_maskdecoder:
                for name, param in self.visual_model.named_parameters():
                    if "mask_decoder" in name:
                        param.requires_grad = False
        
        # configの型を検出して適切に隠れ層次元を取得
        print(f"設定オブジェクトの型: {type(config).__name__}")
        
        if hasattr(config, "hidden_size"):
            # すでにGemma3TextConfigの場合（テキスト設定のみ）
            hidden_size = getattr(config, "hidden_size", 4096)
            print(f"設定から直接hidden_size={hidden_size}を取得")
        else:
            # 何らかの理由でhidden_sizeが取得できない場合のデフォルト値
            hidden_size = 4096
            print(f"デフォルト値としてhidden_size={hidden_size}を使用")
        
        self.text_hidden_fcs = nn.ModuleList([
            nn.Linear(hidden_size, out_dim)
        ])
        
        print(f"SAMモデルとテキスト射影層(入力次元:{hidden_size}→出力次元:{out_dim})の初期化が完了しました。")
        return self

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        vision_pretrained = kwargs.pop("vision_pretrained", None)
        lisa_config = kwargs.pop("lisa_config", None)
        seg_token_idx = kwargs.pop("seg_token_idx", None)
        out_dim = kwargs.pop("out_dim", 256)
        train_mask_decoder = kwargs.pop("train_mask_decoder", False)
        
        # LISA特有のパラメータを取り除く
        ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        use_mm_start_end = kwargs.pop("use_mm_start_end", True)
        
        # その他のLISA特有のパラメータを必要に応じて除外
        for key in ["freeze_model", "freeze_maskdecoder"]:
            if key in kwargs:
                kwargs.pop(key)
        
        # まずGemmaモデルをロード
        gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path, 
            *model_args, 
            **kwargs
        )
        
        # LISAモデルを生成
        model = cls(
            config=gemma_model.config,
            seg_token_idx=seg_token_idx,
            vision_pretrained=vision_pretrained,
            out_dim=out_dim,
            train_mask_decoder=train_mask_decoder
        )
        
        # 除外したLISA特有のパラメータを設定
        model.ce_loss_weight = ce_loss_weight
        model.dice_loss_weight = dice_loss_weight
        model.bce_loss_weight = bce_loss_weight
        
        # Gemmaモデルの状態をコピー
        model.gemma_model = gemma_model
        
        # 追加のLISA設定があれば初期化
        if lisa_config is not None and vision_pretrained is None:
            model.initialize_lisa_modules(lisa_config)
        
        return model

    def dice_loss(self, inputs, targets, num_masks=1, reduction="mean"):
        """
        Dice損失関数
        
        Args:
            inputs (torch.Tensor): 予測マスク [B, num_masks, H, W]
            targets (torch.Tensor): 正解マスク [B, num_masks, H, W]
            num_masks (int): マスクの数
            reduction (str): 損失値の縮約方法 ('mean'または'sum')
            
        Returns:
            torch.Tensor: 損失値
        """
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)
        
        numerator = 2 * torch.sum(inputs * targets, dim=-1)
        denominator = torch.sum(inputs, dim=-1) + torch.sum(targets, dim=-1) + 1e-6
        
        loss = 1 - (numerator / denominator)
        
        if reduction == "none":
            return loss
        
        # バッチとマスクの平均を取る
        if num_masks == 0:
            return torch.tensor(0.0, device=inputs.device)
        
        return loss.sum() / (loss.shape[0] * num_masks) if reduction == "mean" else loss.sum()
    
    def sigmoid_ce_loss(self, inputs, targets, num_masks=1, reduction="mean"):
        """
        シグモイドクロスエントロピー損失関数
        
        Args:
            inputs (torch.Tensor): 予測マスク [B, num_masks, H, W]
            targets (torch.Tensor): 正解マスク [B, num_masks, H, W]
            num_masks (int): マスクの数
            reduction (str): 損失値の縮約方法 ('mean'または'sum')
            
        Returns:
            torch.Tensor: 損失値
        """
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)
        
        loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        ).mean(dim=-1)
        
        if reduction == "none":
            return loss
        
        # バッチとマスクの平均を取る
        if num_masks == 0:
            return torch.tensor(0.0, device=inputs.device)
        
        return loss.sum() / (loss.shape[0] * num_masks) if reduction == "mean" else loss.sum()
    
    def forward(self, input_ids=None, attention_mask=None, past_key_values=None, pixel_values=None, 
              labels=None, images=None, images_sam=None, inference=False, use_cache=None, 
              return_dict=None, output_hidden_states=None, mask_labels=None, **kwargs):
        """
        LISA モデルの forward メソッド
        
        Args:
            input_ids: 入力テキストのトークン ID
            attention_mask: アテンションマスク
            past_key_values: 過去の key-value 値 (生成時に使用)
            pixel_values: 画像ピクセル値 (Gemma3 の視覚入力用)
            labels: 言語モデリング用のラベル
            images: 画像入力 (images_clip と images_sam が指定されていない場合に使用)
            images_sam: SAM 用に処理された画像
            inference: 推論モードかどうか
            use_cache: キャッシュを使用するかどうか
            return_dict: 辞書形式で結果を返すかどうか
            output_hidden_states: 隠れ状態を出力するかどうか
            mask_labels: セグメンテーションマスクのラベル
            
        Returns:
            言語モデル出力とセグメンテーションマスク
        """
        if return_dict is None:
            return_dict = True
            
        # 隠れ状態を常に出力するようにする（[SEG]トークンの処理に必要）
        output_hidden_states = True
        
        # SAMのセグメンテーション機能が有効かどうか
        sam_enabled = hasattr(self, "visual_model") and self.visual_model is not None and self.seg_token_idx is not None

        # 画像の前処理
        if sam_enabled and images_sam is None and images is not None:
            # 画像が提供されている場合、SAM 用に変換
            sam_transform = ResizeLongestSide(self.image_encoder.img_size)
            images_sam = []

            # バッチ処理：imagesの次元を確認
            if torch.is_tensor(images):
                # テンソルの場合、次元数を確認
                if images.dim() == 4:  # [B, C, H, W]または[B, H, W, C]
                    # torchvisionの期待するフォーマットに変換
                    if images.shape[1] == 3 or images.shape[1] == 1:  # [B, C, H, W]形式
                        # チャネル次元を最後に移動: [B, C, H, W] -> [B, H, W, C]
                        images_np = images.permute(0, 2, 3, 1).cpu().numpy()
                    else:  # [B, H, W, C]形式と仮定
                        images_np = images.cpu().numpy()
                    
                    # 各バッチ要素を個別に処理
                    for i in range(images_np.shape[0]):
                        image_np = images_np[i]  # [H, W, C]
                        transformed_image = sam_transform.apply_image(image_np)
                        images_sam.append(transformed_image)
                elif images.dim() == 3:  # 単一画像 [C, H, W]または[H, W, C]
                    if images.shape[0] == 3 or images.shape[0] == 1:  # [C, H, W]形式
                        image_np = images.permute(1, 2, 0).cpu().numpy()
                    else:  # [H, W, C]形式と仮定
                        image_np = images.cpu().numpy()
                    transformed_image = sam_transform.apply_image(image_np)
                    images_sam.append(transformed_image)
                else:
                    raise ValueError(f"サポートされていない画像次元です: {images.dim()}。2次元、3次元、または4次元のテンソルを期待しています。")
            elif isinstance(images, list):
                # リスト形式の場合、各要素を処理
                for img in images:
                    if torch.is_tensor(img):
                        if img.dim() == 3:  # [C, H, W]または[H, W, C]
                            if img.shape[0] == 3 or img.shape[0] == 1:  # [C, H, W]形式
                                img_np = img.permute(1, 2, 0).cpu().numpy()
                            else:  # [H, W, C]形式と仮定
                                img_np = img.cpu().numpy()
                        elif img.dim() == 2:  # [H, W]
                            img_np = img.cpu().numpy()
                        else:
                            raise ValueError(f"サポートされていない画像次元です: {img.dim()}。2次元または3次元のテンソルを期待しています。")
                    else:
                        img_np = img  # すでにnumpy配列と仮定
                    transformed_image = sam_transform.apply_image(img_np)
                    images_sam.append(transformed_image)
            else:
                raise ValueError("imagesはテンソルまたはテンソルのリストである必要があります。")
            
            # 処理された画像をスタックしてテンソルに変換
            images_sam = torch.stack([torch.from_numpy(image).permute(2, 0, 1).float() for image in images_sam])
            if images_sam.device != self.device:
                images_sam = images_sam.to(self.device)
            
            # SAM画像はそのままにして、Gemma3用の画像を別途準備
            # pixel_valuesがNoneでimagesが提供されている場合のみ
            if pixel_values is None:
                # Gemma3用の画像処理（896x896にリサイズし正規化）
                try:
                    from torchvision.transforms import functional as F
                    from PIL import Image
                    import numpy as np
                    
                    # Gemma3 (SigLIP)のビジョンモデル用の画像サイズ
                    gemma_img_size = 896  # Gemma3の標準入力サイズ
                    
                    try:
                        # 可能であればHugging FaceのAutoProcessorを使用
                        from transformers import AutoProcessor
                        processor = None
                        try:
                            processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")
                            print("Gemma3 AutoProcessorを使用して画像を処理します")
                            
                            if torch.is_tensor(images):
                                # テンソル画像をPIL形式に変換してプロセッサに渡す
                                pixel_values_list = []
                                
                                for i in range(images.shape[0] if images.dim() == 4 else 1):
                                    img = images[i] if images.dim() == 4 else images
                                    
                                    # [C, H, W]形式から[H, W, C]形式に変換
                                    if img.dim() == 3 and (img.shape[0] == 3 or img.shape[0] == 1):
                                        img = img.permute(1, 2, 0)
                                    
                                    # numpy配列に変換してからPIL画像に変換
                                    img_np = img.cpu().numpy()
                                    if img_np.max() <= 1.0:
                                        img_np = (img_np * 255).astype(np.uint8)
                                    else:
                                        img_np = img_np.astype(np.uint8)
                                    
                                    pil_img = Image.fromarray(img_np)
                                    processed = processor(images=pil_img, return_tensors="pt")
                                    pixel_values_list.append(processed.pixel_values[0])
                                
                                pixel_values = torch.stack(pixel_values_list)
                            
                            elif isinstance(images, list):
                                # リスト形式の場合、各要素を処理
                                pixel_values_list = []
                                
                                for img in images:
                                    if torch.is_tensor(img):
                                        # テンソルをPIL画像に変換
                                        if img.dim() == 3 and (img.shape[0] == 3 or img.shape[0] == 1):
                                            img = img.permute(1, 2, 0)
                                        
                                        img_np = img.cpu().numpy()
                                        if img_np.max() <= 1.0:
                                            img_np = (img_np * 255).astype(np.uint8)
                                        else:
                                            img_np = img_np.astype(np.uint8)
                                        
                                        pil_img = Image.fromarray(img_np)
                                    else:
                                        # 既にnumpy配列と仮定
                                        img_np = img
                                        if img_np.max() <= 1.0:
                                            img_np = (img_np * 255).astype(np.uint8)
                                        else:
                                            img_np = img_np.astype(np.uint8)
                                        pil_img = Image.fromarray(img_np)
                                    
                                    processed = processor(images=pil_img, return_tensors="pt")
                                    pixel_values_list.append(processed.pixel_values[0])
                                
                                pixel_values = torch.stack(pixel_values_list)
                            
                            # 正しいデバイスに転送
                            if pixel_values.device != self.device:
                                pixel_values = pixel_values.to(self.device)
                                
                            print(f"画像を自動的に{gemma_img_size}x{gemma_img_size}に処理しました: {pixel_values.shape}")
                            
                        except Exception as e:
                            print(f"AutoProcessorが使用できないため手動リサイズに切り替えます: {e}")
                            processor = None
                    except ImportError:
                        print("transformers.AutoProcessorが利用できないため手動処理を行います")
                        processor = None
                    
                    # AutoProcessorが使えない場合は手動でリサイズ
                    if processor is None:
                        pixel_values_list = []
                        
                        if torch.is_tensor(images):
                            # テンソルの場合
                            for i in range(images.shape[0] if images.dim() == 4 else 1):
                                # 単一画像またはバッチ内の各画像を取得
                                if images.dim() == 4:
                                    img = images[i]
                                else:
                                    img = images
                                
                                # [C, H, W]形式に変換
                                if img.shape[0] == 3 or img.shape[0] == 1:
                                    # すでに[C, H, W]形式
                                    img_tensor = img
                                else:
                                    # [H, W, C]を[C, H, W]に変換
                                    img_tensor = img.permute(2, 0, 1)
                                
                                # 0-1の範囲に正規化（既に0-1の場合はスキップ）
                                if img_tensor.max() > 1.0:
                                    img_tensor = img_tensor / 255.0
                                    
                                # PIL画像に変換してリサイズ
                                pil_img = F.to_pil_image(img_tensor)
                                
                                # 896x896にリサイズ（Gemma3の推奨サイズ）
                                resized_img = F.resize(pil_img, (gemma_img_size, gemma_img_size))
                                
                                # テンソルに戻して正規化（SigLIP方式: 平均0.5、標準偏差0.5）
                                img_tensor = F.to_tensor(resized_img)  # 0-1に正規化
                                img_tensor = F.normalize(img_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # -1～1に正規化
                                
                                pixel_values_list.append(img_tensor)
                        
                        elif isinstance(images, list):
                            # リスト形式の場合、各要素を処理
                            for img in images:
                                if torch.is_tensor(img):
                                    # テンソル画像の処理
                                    if img.dim() == 3:
                                        if img.shape[0] == 3 or img.shape[0] == 1:
                                            # すでに[C, H, W]形式
                                            img_tensor = img
                                        else:
                                            # [H, W, C]を[C, H, W]に変換
                                            img_tensor = img.permute(2, 0, 1)
                                    elif img.dim() == 2:
                                        # グレースケール画像を3チャンネルに拡張
                                        img_tensor = img.unsqueeze(0).repeat(3, 1, 1)
                                    
                                    # 0-1の範囲に正規化（既に0-1の場合はスキップ）
                                    if img_tensor.max() > 1.0:
                                        img_tensor = img_tensor / 255.0
                                
                                else:
                                    # numpy配列と仮定
                                    img_np = img
                                    # [H, W, C]形式ならPIL形式に変換
                                    if len(img_np.shape) == 3 and img_np.shape[2] in [1, 3]:
                                        pil_img = Image.fromarray(
                                            (img_np * 255).astype(np.uint8) 
                                            if img_np.max() <= 1.0 
                                            else img_np.astype(np.uint8)
                                        )
                                    else:
                                        # その他の形式の場合はエラー
                                        raise ValueError(f"サポートされていない画像形式です: {img_np.shape}")
                                    
                                    # リサイズしてテンソルに変換
                                    resized_img = F.resize(pil_img, (gemma_img_size, gemma_img_size))
                                    img_tensor = F.to_tensor(resized_img)
                                    img_tensor = F.normalize(img_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                    pixel_values_list.append(img_tensor)
                                    continue
                                
                                # PIL画像に変換してリサイズ
                                pil_img = F.to_pil_image(img_tensor)
                                resized_img = F.resize(pil_img, (gemma_img_size, gemma_img_size))
                                
                                # テンソルに戻して正規化（SigLIP方式）
                                img_tensor = F.to_tensor(resized_img)
                                img_tensor = F.normalize(img_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                
                                pixel_values_list.append(img_tensor)
                        
                        # バッチ化して適切なデバイスに移動
                        if pixel_values_list:
                            pixel_values = torch.stack(pixel_values_list)
                            if pixel_values.device != self.device:
                                pixel_values = pixel_values.to(self.device)
                            print(f"画像を手動で{gemma_img_size}x{gemma_img_size}にリサイズしました: {pixel_values.shape}")
                
                except ImportError:
                    print("警告: torchvisionが利用できないため、Gemma3用の画像処理を実行できませんでした。")
                    pixel_values = None
                    
        # Gemma モデルでテキスト生成部分を処理
        kwargs_gemma = dict(kwargs)
        
        # SiGLIPビジョンモデルの位置埋め込み補間を有効にする
        if "vision_kwargs" not in kwargs_gemma:
            kwargs_gemma["vision_kwargs"] = {}
        
        # vision_kwargsのネスト構造を設定
        if "embeddings_kwargs" not in kwargs_gemma["vision_kwargs"]:
            kwargs_gemma["vision_kwargs"]["embeddings_kwargs"] = {}
        
        # 位置埋め込みの補間を常に有効化（パッチ数不一致問題の解決策）
        kwargs_gemma["vision_kwargs"]["embeddings_kwargs"]["interpolate_pos_encoding"] = True
        
        # 画像サイズとVisionモデルの設定についてのログ出力
        if pixel_values is not None:
            # 画像サイズの確認
            img_size = pixel_values.shape[-2:]
            print(f"Gemma3/SiGLIPモデルへの入力サイズ: {img_size}")
            
            # 推奨サイズとの比較
            if img_size != (896, 896):
                print(f"警告: 入力サイズ{img_size}はGemma3の推奨サイズ(896, 896)と異なります")
                print(f"位置エンコーディングの補間を有効化しています: interpolate_pos_encoding=True")
            else:
                print(f"入力サイズはGemma3の推奨サイズ(896, 896)と一致しています - 最適な処理が行われます")
                
            # パッチ数の計算（典型的なパッチサイズ14で計算）
            patch_size = 14  # SiGLIPの一般的なパッチサイズ
            expected_patches = (896 // patch_size) ** 2
            actual_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
            print(f"パッチサイズ{patch_size}の場合の推定パッチ数:")
            print(f"  推奨入力(896x896)のパッチ数: {expected_patches}")
            print(f"  実際の入力{img_size}のパッチ数: {actual_patches}")
        
        gemma_outputs = self.gemma_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            pixel_values=pixel_values,
            labels=labels,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs_gemma
        )
        
        # SAM セグメンテーション機能が無効の場合は言語モデル出力のみ返す
        if not sam_enabled:
            return gemma_outputs
            
        # 学習モードと推論モードで [SEG] トークンの検索方法を変える
        if labels is not None and not inference:
            # 学習時: labels 内の [SEG] トークンを検索
            seg_positions = (labels == self.seg_token_idx).nonzero(as_tuple=True)
        else:
            # 推論時: input_ids 内の [SEG] トークンを検索
            seg_positions = (input_ids == self.seg_token_idx).nonzero(as_tuple=True)
            
        # [SEG] トークンがない場合かつ推論モードの場合は言語モデル出力のみ返す
        # 学習モードでは、[SEG]がなくてもマスク損失を計算（0として）
        if len(seg_positions[0]) == 0 and inference:
            return gemma_outputs
        
        # セグメンテーション処理の準備
        # バッチサイズと [SEG] トークンの位置情報を取得
        batch_size = input_ids.shape[0]
        batch_indices, token_indices = seg_positions
        
        # SAM エンコーダで画像埋め込みを生成（一度だけ）
        if images_sam is not None:
            # バッチサイズが異なる場合は対応
            if images_sam.shape[0] != batch_size:
                if images_sam.shape[0] == 1:
                    images_sam = images_sam.repeat(batch_size, 1, 1, 1)
                else:
                    raise ValueError(f"バッチサイズが一致しません: input_ids={batch_size}, images_sam={images_sam.shape[0]}")
            
            # 画像エンコーダで埋め込みを生成（勾配を保持するかどうかを決定）
            with torch.set_grad_enabled(self.image_encoder.training):
                image_embeddings = self.image_encoder(images_sam)
        else:
            # 画像が提供されていない場合
            image_embeddings = None
        
        # LLMの最終隠れ状態を取得
        last_hidden_state = gemma_outputs.hidden_states[-1]
        
        # [SEG]トークンの埋め込みを変換してプロンプト埋め込みを生成
        seg_embeddings = []
        for batch_idx, token_idx in zip(batch_indices, token_indices):
            seg_embedding = last_hidden_state[batch_idx, token_idx]
            # テキスト埋め込みをSAMプロンプト次元に変換
            seg_embeddings.append(self.text_hidden_fcs[0](seg_embedding))
        
        # マスク予測の初期化
        mask_predictions = []
        batch_masks_list = []  # 全バッチ、全[SEG]トークンのマスクを保存
        
        # バッチごとに処理
        for batch_idx in range(batch_size):
            # このバッチインデックスに対応する [SEG] トークンのインデックスを取得
            batch_mask_indices = (batch_indices == batch_idx).nonzero(as_tuple=True)[0]
            
            # [SEG] トークンがないか、画像が提供されていない場合はNoneを追加
            if len(batch_mask_indices) == 0 or image_embeddings is None:
                mask_predictions.append(None)
                batch_masks_list.append(None)
                continue
            
            # この画像の画像埋め込みを取得
            batch_image_embedding = image_embeddings[batch_idx].unsqueeze(0)
            
            # 画像の位置埋め込みを取得
            image_pe = self.prompt_encoder.get_dense_pe()
            
            # この画像の全 [SEG] トークンに対応する埋め込みを取得
            batch_seg_embeddings = [seg_embeddings[i] for i in batch_mask_indices]
            
            # この画像の全 [SEG] トークンに対するマスク予測
            batch_masks = []
            for embedding in batch_seg_embeddings:
                # [SEG]埋め込みをSAMのプロンプトエンコーダ互換形式に変換
                sparse_embeddings = embedding.unsqueeze(0).unsqueeze(0)  # [1, 1, D]
                
                # SAMのマスクデコーダに渡してマスクを生成
                low_res_masks, _ = self.mask_decoder(
                    image_embeddings=batch_image_embedding,
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=None,
                    multimask_output=False,  # 各[SEG]につき1つのマスクのみ生成
                )
                
                batch_masks.append(low_res_masks)
            
            # すべてのマスクを連結（複数の[SEG]トークンがある場合）
            if batch_masks:
                batch_masks_cat = torch.cat(batch_masks, dim=1)
                mask_predictions.append(batch_masks_cat)
                batch_masks_list.append(batch_masks_cat)
            else:
                mask_predictions.append(None)
                batch_masks_list.append(None)
        
        # 総合的な出力を構築
        # 言語モデル部分の損失
        lm_loss = gemma_outputs.loss if hasattr(gemma_outputs, "loss") else None
        
        # セグメンテーション損失
        mask_loss = None
        
        # 学習時にはセグメンテーション損失を計算
        if not inference and mask_labels is not None and any(m is not None for m in batch_masks_list):
            mask_loss = 0.0
            num_valid_masks = 0
            bce_weight = 2.0  # BCE損失の重み
            dice_weight = 0.5  # Dice損失の重み
            
            for i, (pred_mask, gt_mask) in enumerate(zip(batch_masks_list, mask_labels)):
                if pred_mask is None or gt_mask is None:
                    continue
                
                if len(pred_mask.shape) != len(gt_mask.shape):
                    # 予測マスクと正解マスクの形状を揃える
                    gt_mask = gt_mask.unsqueeze(1) if len(pred_mask.shape) > len(gt_mask.shape) else gt_mask
                    pred_mask = pred_mask.unsqueeze(1) if len(gt_mask.shape) > len(pred_mask.shape) else pred_mask
                
                num_masks = pred_mask.shape[1]
                num_valid_masks += num_masks
                
                # BCE損失とDice損失の計算
                bce_loss = self.sigmoid_ce_loss(pred_mask, gt_mask, num_masks)
                dice_loss = self.dice_loss(pred_mask, gt_mask, num_masks)
                
                # 重み付けして合計
                mask_loss += bce_weight * bce_loss + dice_weight * dice_loss
            
            # 有効なマスクがある場合のみ損失を計算
            if num_valid_masks > 0:
                mask_loss = mask_loss / batch_size  # バッチサイズで正規化
            else:
                mask_loss = torch.tensor(0.0, device=lm_loss.device if lm_loss is not None else self.device)
        
        # 最終的な損失
        total_loss = lm_loss
        if mask_loss is not None:
            if lm_loss is not None:
                total_loss = lm_loss + mask_loss
            else:
                total_loss = mask_loss
        
        # 最終的な出力
        output = {
            "loss": total_loss,  # 言語モデル損失とマスク損失の合計
            "lm_loss": lm_loss,
            "mask_loss": mask_loss,
            "logits": gemma_outputs.logits if hasattr(gemma_outputs, "logits") else None,
            "past_key_values": gemma_outputs.past_key_values if hasattr(gemma_outputs, "past_key_values") else None,
            "hidden_states": gemma_outputs.hidden_states if hasattr(gemma_outputs, "hidden_states") else None,
            "attentions": gemma_outputs.attentions if hasattr(gemma_outputs, "attentions") else None,
            "mask_predictions": mask_predictions,
        }
        
        return output

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        """
        生成用の入力を準備
        
        Args:
            input_ids: 入力テキストのトークン ID
            past_key_values: 過去の key-value 値
            attention_mask: アテンションマスク
            **model_kwargs: その他の引数
        
        Returns:
            生成用の入力
        """
        # 画像関連の入力を取得
        images = model_kwargs.get("images", None)
        images_sam = model_kwargs.get("images_sam", None)
        pixel_values = model_kwargs.get("pixel_values", None)
        
        # 最初のパスの場合は画像処理を行う
        if past_key_values is None:
            # 画像のSAM処理
            if images_sam is None and images is not None:
                # SAM 用の画像変換
                if not isinstance(images, list):
                    images = [images]
                sam_transform = ResizeLongestSide(1024)
                images_sam = []
                for image in images:
                    images_sam.append(sam_transform.apply_image(image))
                images_sam = torch.stack([torch.from_numpy(image).permute(2, 0, 1).float() for image in images_sam])
                if images_sam.device != self.device:
                    images_sam = images_sam.to(self.device)
                model_kwargs["images_sam"] = images_sam
                
            # SiGLIPビジョンモデルの位置埋め込み補間を有効にする
            vision_kwargs = model_kwargs.get("vision_kwargs", {})
            if "embeddings_kwargs" not in vision_kwargs:
                vision_kwargs["embeddings_kwargs"] = {}
            vision_kwargs["embeddings_kwargs"]["interpolate_pos_encoding"] = True
            model_kwargs["vision_kwargs"] = vision_kwargs
                
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "pixel_values": pixel_values,  # Gemma3用の画像入力
                "images_sam": images_sam,  # SAM用の画像入力
                "use_cache": True,
                "inference": True,
                "vision_kwargs": vision_kwargs,
            }
        else:
            # 2回目以降のパスでは画像処理をスキップ
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "use_cache": True,
                "inference": True,
            }
        
        return inputs

    def generate(
        self,
        input_ids=None,
        images=None,
        images_sam=None,
        pixel_values=None,
        attention_mask=None,
        **generate_kwargs
    ):
        """
        テキスト生成とセグメンテーション予測を行う
        
        Args:
            input_ids: 入力テキストのトークン ID
            images: 画像入力
            images_sam: SAM 用に処理された画像
            pixel_values: 画像ピクセル値 (Gemma3 の視覚入力用)
            attention_mask: アテンションマスク
            **generate_kwargs: その他の生成オプション
        
        Returns:
            生成されたテキストとセグメンテーションマスク
        """
        # SAM モデルの初期化確認
        if (images is not None or images_sam is not None) and self.visual_model is None:
            raise ValueError("SAM モデルが初期化されていません。initialize_lisa_modules を呼び出してください。")
            
        # SEG トークンが設定されているか確認
        if self.seg_token_idx is None:
            print("警告: seg_token_idx が設定されていません。セグメンテーションは行われません。")
            
        # 生成オプションを設定
        generate_kwargs["output_hidden_states"] = True
        generate_kwargs["return_dict_in_generate"] = True
        generate_kwargs["images"] = images
        generate_kwargs["images_sam"] = images_sam
        generate_kwargs["pixel_values"] = pixel_values
        
        # テキスト生成を実行
        outputs = super().generate(input_ids, attention_mask=attention_mask, **generate_kwargs)
        
        # 生成されたテキストに [SEG] トークンがあるかチェック
        generated_ids = outputs.sequences
        seg_positions = (generated_ids == self.seg_token_idx).nonzero(as_tuple=True)
        
        # [SEG] トークンがない場合はテキスト出力のみ返す
        if len(seg_positions[0]) == 0:
            return outputs
            
        # 生成された [SEG] トークンの隠れ状態を取得
        if not hasattr(outputs, "hidden_states") or not outputs.hidden_states:
            raise ValueError("hidden_states がありません。generate_kwargs で output_hidden_states=True を設定してください。")
            
        # 最後のレイヤーの隠れ状態を取得
        last_hidden_states = outputs.hidden_states[-1][-1]  # 最後の生成ステップ、最後のレイヤー
        
        # [SEG] トークンの埋め込みを抽出
        batch_indices, token_indices = seg_positions
        
        # マスクを生成
        mask_outputs = []
        
        if images_sam is not None:
            # 各 [SEG] トークンについて処理
            pred_embeddings = []
            for batch_idx, token_idx in zip(batch_indices, token_indices):
                seg_embedding = last_hidden_states[batch_idx, token_idx]
                # テキスト埋め込みをSAMプロンプト次元に変換
                pred_embeddings.append(self.text_hidden_fcs[0](seg_embedding))
                
            pred_embeddings = torch.stack(pred_embeddings)
            
            # SAM エンコーダで特徴抽出
            with torch.no_grad():
                image_embeddings = self.image_encoder(images_sam)
                
            # 各バッチ/画像ごとにセグメンテーション
            batch_size = generated_ids.shape[0]
            for batch_idx in range(batch_size):
                batch_mask_indices = (batch_indices == batch_idx).nonzero(as_tuple=True)[0]
                if len(batch_mask_indices) == 0:
                    mask_outputs.append(None)
                    continue
                    
                # この画像の全 [SEG] トークンについて処理
                batch_embeddings = pred_embeddings[batch_mask_indices]
                batch_image_embedding = image_embeddings[batch_idx].unsqueeze(0)
                
                masks = []
                for embedding in batch_embeddings:
                    # SAM のマスクデコーダを使用
                    sparse_embeddings = embedding.unsqueeze(0).unsqueeze(0)
                    mask_predictions, _ = self.mask_decoder(
                        image_embeddings=batch_image_embedding,
                        image_pe=self.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=None,
                        multimask_output=False,
                    )
                    masks.append(mask_predictions)
                    
                mask_outputs.append(torch.cat(masks, dim=1) if masks else None)
                
        # 出力をカスタムクラスにまとめる
        combined_output = {
            "sequences": outputs.sequences,
            "scores": outputs.scores if hasattr(outputs, "scores") else None,
            "hidden_states": outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            "mask_predictions": mask_outputs if mask_outputs else None,
        }
        
        return combined_output

    def get_visual_embs(self, pixel_values):
        """SAMの視覚エンコーダを使用して画像埋め込みを取得"""
        # SAMの画像エンコーダを使用
        return self.image_encoder(pixel_values)

    def generate(self, *args, **kwargs):
        return self.gemma_model.generate(*args, **kwargs)

    def evaluate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        """指定されたテキストで推論を実行し、[SEG]トークンの埋め込みを使ってマスクを生成"""
        # attention_maskが指定されていなければ自動生成
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids).to(input_ids.device)
        
        # 生成呼び出し
        gen_outputs = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            **kwargs,
        )
        
        # 辞書の場合のみseg_embeddingsを取得
        if isinstance(gen_outputs, dict) and 'seg_embeddings' in gen_outputs:
            seg_embeddings = gen_outputs['seg_embeddings']
            
            # 画像がなければマスク生成はスキップ
            if pixel_values is None and images is None:
                return gen_outputs
            
            # SAM用の画像を取得
            sam_images = images if images is not None else pixel_values
            
            # SAMの視覚エンコーダで画像埋め込みを計算
            image_embeddings = self.get_visual_embs(sam_images)
            
            # 各バッチアイテムについてマスクを生成
            masks = []
            for batch_idx, embeds in seg_embeddings.items():
                batch_masks = []
                
                for seg_emb in embeds:
                    # [SEG]埋め込みをSAMプロンプト次元に変換
                    seg_embedding_projected = self.text_hidden_fcs[0](seg_emb).unsqueeze(0).unsqueeze(1)  # [1, 1, 256]
                    
                    # プロンプトエンコーダを使用して埋め込みをSAMの入力形式に変換
                    sparse_embeddings, dense_embeddings = self.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                        text_embeds=seg_embedding_projected,
                    )
                    
                    # SAMのマスクデコーダを使用してマスクを生成
                    low_res_masks, _ = self.mask_decoder(
                        image_embeddings=image_embeddings[batch_idx].unsqueeze(0),
                        image_pe=self.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,  # 単一マスクを出力
                    )
                    
                    batch_masks.append(low_res_masks)
                
                if batch_masks:
                    # バッチアイテムのマスクを結合 (N, 1, H, W)
                    batch_masks = torch.cat(batch_masks, dim=1)
                    masks.append(batch_masks)
            
            if masks:
                # すべてのバッチアイテムのマスクを結合 (B, N, H, W)
                all_masks = torch.cat(masks, dim=0)
                gen_outputs['pred_masks'] = all_masks
        
        return gen_outputs

    def get_image_embeddings(self, image):
        """
        画像をSAMの画像エンコーダでエンコードし、画像埋め込みを返す
        
        Args:
            image (torch.Tensor): 画像テンソル [B, C, H, W]
            
        Returns:
            torch.Tensor: 画像埋め込み
        """
        if not hasattr(self, "visual_model") or self.visual_model is None:
            raise ValueError("SAMモデルが初期化されていません。initialize_lisa_modulesを呼び出してください。")
        
        return self.image_encoder(image)
    
    @property
    def prompt_encoder(self):
        """SAMのプロンプトエンコーダへのアクセスを提供"""
        if not hasattr(self, "visual_model") or self.visual_model is None:
            raise ValueError("SAMモデルが初期化されていません。initialize_lisa_modulesを呼び出してください。")
        
        return self.prompt_encoder
    
    @property
    def mask_decoder(self):
        """SAMのマスクデコーダへのアクセスを提供"""
        if not hasattr(self, "visual_model") or self.visual_model is None:
            raise ValueError("SAMモデルが初期化されていません。initialize_lisa_modulesを呼び出してください。")
        
        return self.mask_decoder
    
    def postprocess_masks(self, masks, input_size, original_size):
        """
        生成されたマスクを後処理して元の画像サイズに戻す
        
        Args:
            masks (torch.Tensor): 生成されたマスク
            input_size (tuple): 入力サイズ (H, W)
            original_size (tuple): 元の画像サイズ (H, W)
            
        Returns:
            torch.Tensor: 後処理されたマスク
        """
        if not hasattr(self, "visual_model") or self.visual_model is None:
            raise ValueError("SAMモデルが初期化されていません。initialize_lisa_modulesを呼び出してください。")
        
        return self.visual_model.postprocess_masks(masks, input_size, original_size)

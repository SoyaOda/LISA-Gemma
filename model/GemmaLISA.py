import os
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass

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
        
        self.visual_model = sam_model_registry['vit_h'](checkpoint=vision_pretrained)
        
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
        # 注: Gemma3TextConfigは直接hidden_sizeを持つが、Gemma3Configはtext_config.hidden_sizeに持つ
        print(f"設定オブジェクトの型: {type(config).__name__}")
        
        if hasattr(config, "text_config"):
            # Gemma3Configの場合（マルチモーダル設定）
            hidden_size = getattr(config.text_config, "hidden_size", 4096)
            print(f"Gemma3Configからtext_config.hidden_size={hidden_size}を取得")
        else:
            # すでにGemma3TextConfigの場合（テキスト設定のみ）
            hidden_size = getattr(config, "hidden_size", 4096)
            print(f"設定から直接hidden_size={hidden_size}を取得")
        
        self.text_hidden_fcs = nn.ModuleList([
            nn.Linear(hidden_size, out_dim)
        ])
        
        print(f"SAMモデルとテキスト射影層(入力次元:{hidden_size}→出力次元:{out_dim})の初期化が完了しました。")
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

    # GemmaLISAMetaModelのinitialize_lisa_modulesメソッドと同等の機能を実装
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
        
        self.visual_model = sam_model_registry['vit_h'](checkpoint=vision_pretrained)
        
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
        # 注: Gemma3TextConfigは直接hidden_sizeを持つが、Gemma3Configはtext_config.hidden_sizeに持つ
        print(f"設定オブジェクトの型: {type(config).__name__}")
        
        if hasattr(config, "text_config"):
            # Gemma3Configの場合（マルチモーダル設定）
            hidden_size = getattr(config.text_config, "hidden_size", 4096)
            print(f"Gemma3Configからtext_config.hidden_size={hidden_size}を取得")
        else:
            # すでにGemma3TextConfigの場合（テキスト設定のみ）
            hidden_size = getattr(config, "hidden_size", 4096)
            print(f"設定から直接hidden_size={hidden_size}を取得")
        
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
        
        # Gemmaモデルの状態をコピー
        model.gemma_model = gemma_model
        
        # 追加のLISA設定があれば初期化
        if lisa_config is not None and vision_pretrained is None:
            model.initialize_lisa_modules(lisa_config)
        
        return model

    def forward(self, input_ids=None, attention_mask=None, past_key_values=None, pixel_values=None, 
             labels=None, images=None, images_clip=None, images_sam=None, inference=False, use_cache=None, 
             return_dict=None, output_hidden_states=None, **kwargs):
        """
        LISA モデルの forward メソッド
        Args:
            input_ids: 入力テキストのトークン ID
            attention_mask: アテンションマスク
            past_key_values: 過去の key-value 値 (生成時に使用)
            pixel_values: 画像ピクセル値 (Gemma3 の視覚入力用)
            labels: 言語モデリング用のラベル
            images: 画像入力 (images_clip と images_sam が指定されていない場合に使用)
            images_clip: CLIP 用に処理された画像
            images_sam: SAM 用に処理された画像
            inference: 推論モードかどうか
            use_cache: キャッシュを使用するかどうか
            return_dict: 辞書形式で結果を返すかどうか
            output_hidden_states: 隠れ状態を出力するかどうか
        Returns:
            言語モデル出力とセグメンテーションマスク
        """
        if return_dict is None:
            return_dict = True
            
        output_hidden_states = True if self.visual_model is not None else output_hidden_states

        # 画像の前処理
        if images_sam is None and images is not None:
            # 画像が提供されている場合、SAM 用に変換
            if not isinstance(images, list):
                images = [images]
            sam_transform = ResizeLongestSide(1024)
            images_sam = []
            for image in images:
                images_sam.append(sam_transform.apply_image(image))
            images_sam = torch.stack([torch.from_numpy(image).permute(2, 0, 1).float() for image in images_sam])
            if images_sam.device != self.device:
                images_sam = images_sam.to(self.device)
            
        # Gemma モデルでテキスト生成部分を処理
        gemma_outputs = self.gemma_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            pixel_values=pixel_values,
            labels=labels,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        
        # SAM モデルの初期化がまだされていない場合は言語モデル出力をそのまま返す
        if self.visual_model is None or self.seg_token_idx is None:
            return gemma_outputs
            
        # [SEG] トークンの位置を見つける
        if labels is not None and not inference:
            # 学習時: labels 内の [SEG] トークンを検索
            seg_positions = (labels == self.seg_token_idx).nonzero(as_tuple=True)
        else:
            # 推論時: input_ids 内の [SEG] トークンを検索
            seg_positions = (input_ids == self.seg_token_idx).nonzero(as_tuple=True)
            
        # [SEG] トークンがない場合は言語モデル出力をそのまま返す
        if len(seg_positions[0]) == 0:
            return gemma_outputs
        
        # 隠れ状態を取得し [SEG] トークンの埋め込みを抽出
        last_hidden_state = gemma_outputs.hidden_states[-1]
        batch_indices, token_indices = seg_positions
        
        # 各 [SEG] トークンについて処理
        pred_embeddings = []
        for batch_idx, token_idx in zip(batch_indices, token_indices):
            seg_embedding = last_hidden_state[batch_idx, token_idx]
            # テキスト埋め込みをSAMプロンプト次元に変換
            pred_embeddings.append(self.text_hidden_fcs[0](seg_embedding))
        
        pred_embeddings = torch.stack(pred_embeddings)
        
        # SAM でセグメンテーション予測
        batch_size = input_ids.shape[0]
        mask_predictions = []
        
        # SAM 用の前処理
        if images_sam is not None:
            # バッチサイズが異なる場合は対応
            if images_sam.shape[0] != batch_size:
                if images_sam.shape[0] == 1:
                    images_sam = images_sam.repeat(batch_size, 1, 1, 1)
                else:
                    raise ValueError(f"バッチサイズが一致しません: input_ids={batch_size}, images_sam={images_sam.shape[0]}")
                    
            # SAM エンコーダで特徴抽出
            with torch.no_grad():
                image_embeddings = self.visual_model.image_encoder(images_sam)
                
            # 予測されたセグメント埋め込みをバッチごとに処理
            for batch_idx in range(batch_size):
                batch_mask_indices = (batch_indices == batch_idx).nonzero(as_tuple=True)[0]
                if len(batch_mask_indices) == 0:
                    # この画像には [SEG] トークンがない
                    mask_predictions.append(None)
                    continue
                
                # この画像の全 [SEG] トークンに対する予測
                batch_embeddings = pred_embeddings[batch_mask_indices]
                batch_image_embedding = image_embeddings[batch_idx].unsqueeze(0)
                
                masks = []
                for embedding in batch_embeddings:
                    # エンベディングをSAMのマスクデコーダに渡す
                    sparse_embeddings = embedding.unsqueeze(0).unsqueeze(0)  # [1, 1, D]
                    
                    # SAM のマスクデコーダを使用してマスク予測
                    mask_predictions_output, _ = self.visual_model.mask_decoder(
                        image_embeddings=batch_image_embedding,
                        image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=None,
                        multimask_output=False,
                    )
                    
                    masks.append(mask_predictions_output)
                
                # この画像の全セグメンテーションマスクを保存
                mask_predictions.append(torch.cat(masks, dim=1) if masks else None)
        
        # 出力を返す
        output = {
            "loss": gemma_outputs.loss if hasattr(gemma_outputs, "loss") else None,
            "logits": gemma_outputs.logits if hasattr(gemma_outputs, "logits") else None,
            "past_key_values": gemma_outputs.past_key_values if hasattr(gemma_outputs, "past_key_values") else None,
            "hidden_states": gemma_outputs.hidden_states if hasattr(gemma_outputs, "hidden_states") else None,
            "mask_predictions": mask_predictions if mask_predictions else None,
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
                
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "pixel_values": pixel_values,  # Gemma3用の画像入力
                "images_sam": images_sam,  # SAM用の画像入力
                "use_cache": True,
                "inference": True,
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
                image_embeddings = self.visual_model.image_encoder(images_sam)
                
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
                    mask_predictions, _ = self.visual_model.mask_decoder(
                        image_embeddings=batch_image_embedding,
                        image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
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
        return self.visual_model.image_encoder(pixel_values)

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
                    sparse_embeddings, dense_embeddings = self.visual_model.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                        text_embeds=seg_embedding_projected,
                    )
                    
                    # SAMのマスクデコーダを使用してマスクを生成
                    low_res_masks, _ = self.visual_model.mask_decoder(
                        image_embeddings=image_embeddings[batch_idx].unsqueeze(0),
                        image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
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
        
        return self.visual_model.image_encoder(image)
    
    @property
    def prompt_encoder(self):
        """SAMのプロンプトエンコーダへのアクセスを提供"""
        if not hasattr(self, "visual_model") or self.visual_model is None:
            raise ValueError("SAMモデルが初期化されていません。initialize_lisa_modulesを呼び出してください。")
        
        return self.visual_model.prompt_encoder
    
    @property
    def mask_decoder(self):
        """SAMのマスクデコーダへのアクセスを提供"""
        if not hasattr(self, "visual_model") or self.visual_model is None:
            raise ValueError("SAMモデルが初期化されていません。initialize_lisa_modulesを呼び出してください。")
        
        return self.visual_model.mask_decoder
    
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

"""
GemmaLISA モデルの基本クラス定義
"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel, GenerationMixin, AutoConfig
from model.segment_anything.modeling import Sam
from model.segment_anything import build_sam_vit_h

from typing import List, Optional, Tuple, Union, Dict, Any

# 設定クラスをインポート
from .configuration import GemmaLISAConfig


class GemmaLISAMetaModel(nn.Module):
    """
    GemmaとSAMを組み合わせたメタモデル。
    このクラスはGemmaベースのエンコーダデコーダとSAMの統合を管理します。
    """
    
    def __init__(
        self,
        vision_model=None,
        lisa_model=None,
        sam_model=None
    ):
        """
        GemmaLISAメタモデルを初期化します。
        
        Args:
            vision_model: 視覚エンコーダモデル（Gemmaのビジョンエンコーダ部分）
            lisa_model: LISAモデル（言語処理部分）
            sam_model: セグメントエニシングモデル（SAM）
        """
        super().__init__()
        
        self.vision_model = vision_model
        self.lisa_model = lisa_model
        self.sam_model = sam_model
        self.text_hidden_fcs = None  # テキスト埋め込みからSAMプロンプト埋め込みへの変換モジュール
    
    def get_vision_tower(self):
        """ビジョンタワーモデルを取得します"""
        vision_model = getattr(self, 'vision_model', None)
        if vision_model is None:
            raise ValueError("vision_modelが初期化されていません")
        return vision_model
    
    def get_vision_embeddings(self, images, device=None):
        """
        画像から視覚的埋め込みを取得します。
        
        Args:
            images: 入力画像のバッチ
            device: 使用するデバイス（指定がなければ現在のデバイスを使用）
            
        Returns:
            画像の埋め込み表現
        """
        vision_tower = self.get_vision_tower()
        if not hasattr(vision_tower, "ignored_prefix") or vision_tower.ignored_prefix != "vision_model.":
            vision_tower.ignored_prefix = "vision_model."
        
        # 視覚モデルのプロセッサがあれば使用
        processor = getattr(vision_tower, "image_processor", None)
        if processor is None:
            processor = getattr(vision_tower, "processor", None)
        
        # 画像の埋め込みを取得
        if hasattr(vision_tower, "get_image_embeddings"):
            image_embeddings = vision_tower.get_image_embeddings(
                images,
                processor=processor,
                device=device
            )
        else:
            # デフォルトの処理（forward直接呼び出し）
            image_embeddings = vision_tower(images).last_hidden_state
        
        return image_embeddings
    
    def initialize_lisa_modules(self, model_args, sam_checkpoint=None, freeze_sam=True):
        """
        LISAモジュールを初期化します。
        
        Args:
            model_args: モデル初期化のための引数
            sam_checkpoint: SAMモデルのチェックポイントパス
            freeze_sam: SAMモデルをフリーズするかどうか
        """
        # SAMモデルがまだ初期化されていない場合はここで初期化
        if self.sam_model is None and sam_checkpoint is not None:
            try:
                # build_sam_vit_h関数を使ってSAMモデルを構築
                self.sam_model = build_sam_vit_h(checkpoint=sam_checkpoint)
            except Exception as e:
                print(f"SAMモデルの初期化中にエラーが発生しました: {e}")
        
        # SAMモデルを凍結するかどうかの設定
        if self.sam_model is not None:
            if freeze_sam:
                # SAMモデルのパラメータを凍結
                for param in self.sam_model.parameters():
                    param.requires_grad = False
            elif hasattr(model_args, "train_mask_decoder") and model_args.train_mask_decoder:
                # イメージエンコーダとプロンプトエンコーダを凍結し、マスクデコーダのみ訓練
                for param in self.sam_model.image_encoder.parameters():
                    param.requires_grad = False
                for param in self.sam_model.prompt_encoder.parameters():
                    param.requires_grad = False
                for param in self.sam_model.mask_decoder.parameters():
                    param.requires_grad = True
        
        # テキスト埋め込みからSAMプロンプト埋め込みへの変換モジュールを初期化
        if hasattr(model_args, "out_dim") and model_args.out_dim is not None:
            out_dim = model_args.out_dim
        else:
            out_dim = 256  # SAMのデフォルト次元
        
        # モデルの隠れ層のサイズを取得
        if self.lisa_model is not None:
            config = self.lisa_model.config
            config_type = type(config).__name__
            print(f"設定オブジェクトの型: {config_type}")
            print(f"設定属性一覧: {dir(config)}")
            
            if hasattr(config, "text_config"):
                # Gemma3Configの場合（マルチモーダル設定）
                print(f"text_config属性一覧: {dir(config.text_config)}")
                hidden_size = getattr(config.text_config, "hidden_size", 4096)
                print(f"Gemma3Configからtext_config.hidden_size={hidden_size}を取得")
            else:
                # すでにGemma3TextConfigの場合（テキスト設定のみ）または他の構成
                # model_dimを試す (Gemma3で使用される可能性のある名前)
                if hasattr(config, "model_dim"):
                    hidden_size = config.model_dim
                    print(f"config.model_dim={hidden_size}を取得")
                # hidden_sizeを試す
                elif hasattr(config, "hidden_size"):
                    hidden_size = config.hidden_size
                    print(f"config.hidden_size={hidden_size}を取得")
                # Gemma3固有の可能性がある他の属性を試す
                elif hasattr(config, "hidden_dim"):
                    hidden_size = config.hidden_dim
                    print(f"config.hidden_dim={hidden_size}を取得")
                # デフォルト値を使用
                else:
                    hidden_size = 4096
                    print(f"属性が見つからないためデフォルト値({hidden_size})を使用")
        else:
            # lisa_modelがない場合はデフォルト値を使用
            hidden_size = 4096  # Gemma3のデフォルト隠れ層サイズ
            print(f"警告: lisa_modelが初期化されていません。デフォルトの隠れ層サイズ({hidden_size})を使用します。")
        
        # 変換モジュールの初期化
        self.text_hidden_fcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.GELU(),
                nn.Linear(hidden_size // 2, out_dim),
                nn.LayerNorm(out_dim)
            )
        ])
    
    def forward(self, *args, **kwargs):
        """
        モデルのフォワードパス。
        具体的な実装はサブクラスで行う。
        """
        raise NotImplementedError("このメソッドはサブクラスで実装する必要があります")


class LISAPreTrainedModel(PreTrainedModel, GenerationMixin):
    """
    事前学習済みLISAモデル用の抽象クラス。
    このクラスは、TransformersのPreTrainedModelクラスとGenerationMixinクラスを継承し、
    LISAモデルに必要な共通メソッドを提供します。
    """
    
    config_class = AutoConfig  # 設定クラスをAutoConfigに設定
    base_model_prefix = "lisa"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LISAAttention"]
    _skip_keys_device_placement = "past_key_values"
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version", r"encoder\.version"]
    
    def _init_weights(self, module):
        """
        モジュールの重みを初期化します。
        """
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    def _set_gradient_checkpointing(self, module, value=False):
        """
        グラディエントチェックポイントの設定を行います。
        """
        if isinstance(module, GemmaLISAMetaModel):
            module.gradient_checkpointing = value 
    
    def get_input_embeddings(self):
        """
        入力埋め込み層を取得します。
        
        このデフォルト実装はNotImplementedErrorを発生させます。
        サブクラスでこのメソッドをオーバーライドする必要があります。
        """
        raise NotImplementedError
    
    def set_input_embeddings(self, value):
        """
        入力埋め込み層を設定します。
        
        このデフォルト実装はNotImplementedErrorを発生させます。
        サブクラスでこのメソッドをオーバーライドする必要があります。
        
        Args:
            value: 新しい埋め込み層
        """
        raise NotImplementedError 
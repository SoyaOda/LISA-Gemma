"""
GemmaLISA モデルの主要な実装
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Dict, Any
import math
import warnings
from dataclasses import dataclass
import transformers

from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig, PretrainedConfig
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers.utils import logging
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration

from model.segment_anything import sam_model_registry
from model.segment_anything.modeling import Sam
from model.segment_anything import build_sam_vit_h
from model.GemmaLISA.modeling_base import GemmaLISAMetaModel, LISAPreTrainedModel
from model.GemmaLISA.losses import dice_loss, sigmoid_ce_loss
from model.GemmaLISA.utils import prepare_images_for_sam, prepare_images_for_gemma
from model.GemmaLISA.configuration import GemmaLISAConfig

logger = logging.get_logger(__name__)


class LISAModel(LISAPreTrainedModel):
    """
    GemmaLISAの基本モデルクラス。
    このクラスは、Gemma言語モデルとSAMセグメンテーションモデルを統合します。
    """
    
    def __init__(self, config, sam_model=None, lisa_vision_tower=None, **kwargs):
        """
        LISAモデルの初期化
        
        Args:
            config: モデルの設定
            sam_model: SAMモデル（指定された場合）
            lisa_vision_tower: ビジョンエンコーダ（指定された場合）
        """
        super().__init__(config)
        
        # 最初の初期化
        self.model = GemmaLISAMetaModel(
            vision_model=lisa_vision_tower,
            lisa_model=None,
            sam_model=sam_model
        )
        
        # ビジョンタワーの指定がなく、configに設定がある場合は初期化
        if lisa_vision_tower is None and hasattr(config, "mm_vision_tower") and config.mm_vision_tower is not None:
            from transformers import AutoModel
            logger.info(f"ビジョンタワーをロード中: {config.mm_vision_tower}")
            
            # ビジョンモデルのロード
            self.model.vision_model = AutoModel.from_pretrained(
                config.mm_vision_tower,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
            # プロジェクションレイヤーの初期化
            if not hasattr(self.model.vision_model, "get_image_embeddings"):
                hidden_size = self.model.vision_model.config.hidden_size
                self.model.vision_model.get_image_embeddings = lambda images, **kwargs: self.model.vision_model(images).last_hidden_state
        
        self.post_init()
    
    def get_model(self):
        """モデルのインスタンスを返します"""
        return self.model
    
    def get_vision_tower(self):
        """ビジョンタワーを返します"""
        return self.model.get_vision_tower()
    
    def initialize_lisa_modules(self, model_args, fsdp=None):
        """
        LISAモジュールを初期化します
        
        Args:
            model_args: モデル初期化のための引数
            fsdp: 完全分散データ並列処理の設定（存在する場合）
        """
        self.model.initialize_lisa_modules(model_args)
    
    def get_input_embeddings(self):
        """
        入力埋め込み層を取得します。
        モデルが言語モデルを持っている場合は、その埋め込み層を返します。
        """
        if hasattr(self.model, 'lisa_model') and self.model.lisa_model is not None:
            return self.model.lisa_model.get_input_embeddings()
        raise NotImplementedError("言語モデルが初期化されていません")
    
    def set_input_embeddings(self, value):
        """
        入力埋め込み層を設定します。
        
        Args:
            value: 新しい埋め込み層
        """
        if hasattr(self.model, 'lisa_model') and self.model.lisa_model is not None:
            self.model.lisa_model.set_input_embeddings(value)
        else:
            raise NotImplementedError("言語モデルが初期化されていません")
    
    def forward(self, **kwargs):
        """
        モデルのフォワードパス
        
        具体的な実装はLISAForCausalLMクラスで行います
        """
        raise NotImplementedError("このメソッドはLISAForCausalLMクラスで実装されています")


class LISAForCausalLM(LISAPreTrainedModel):
    """
    因果言語モデリングのためのLISAモデル。
    GemmaベースのLLMとSAMを組み合わせて、テキストと視覚的セグメンテーションを処理します。
    """
    
    def __init__(self, config, sam_checkpoint=None, **kwargs):
        """
        LISAForCausalLMモデルの初期化
        
        Args:
            config: モデルの設定
            sam_checkpoint: SAMモデルのチェックポイントパス（指定された場合）
        """
        super().__init__(config)
        
        # SAMモデルを初期化
        sam_model = None
        if sam_checkpoint is not None:
            logger.info(f"SAMモデルをロード中: {sam_checkpoint}")
            try:
                # build_sam_vit_h関数を使ってSAMモデルを構築
                sam_model = build_sam_vit_h(checkpoint=sam_checkpoint)
                logger.info("SAMモデルのロードに成功しました")
            except Exception as e:
                logger.warning(f"SAMモデルのロード中にエラーが発生しました: {e}")
        
        # LISAモデルを初期化
        self.lisa = LISAModel(config, sam_model=sam_model, **kwargs)
        
        # 言語モデルの設定を確認し初期化
        if hasattr(config, "gemma_config") and config.gemma_config is not None:
            logger.info("Gemma構成を使用してLLMをロード中")
            
            if hasattr(config, "mm_gemma_path") and config.mm_gemma_path is not None:
                # Gemma3モデルを直接ロード
                self.language_model = Gemma3ForConditionalGeneration.from_pretrained(
                    config.mm_gemma_path,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            else:
                # 設定のみから初期化（通常は推奨されない）
                gemma_config = config.gemma_config
                if isinstance(gemma_config, dict):
                    gemma_config = PretrainedConfig.from_dict(gemma_config)
                self.language_model = Gemma3ForConditionalGeneration(gemma_config)
                logger.warning("Gemma3モデルを設定のみから初期化しました。事前学習済み重みは読み込まれていません。")
                
        elif hasattr(config, "mm_gemma_path") and config.mm_gemma_path is not None:
            logger.info(f"Gemmaモデルをロード中: {config.mm_gemma_path}")
            self.language_model = Gemma3ForConditionalGeneration.from_pretrained(
                config.mm_gemma_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        else:
            raise ValueError("GemmaモデルのconfigまたはパスをLISA configに指定する必要があります")
        
        # SAMセグメントデコーダの初期化
        self.seg_token_idx = config.seg_token_idx if hasattr(config, "seg_token_idx") else None
        logger.info(f"セグメントトークンID: {self.seg_token_idx}")
        
        # メタモデルに言語モデルを接続
        self.lisa.model.lisa_model = self.language_model
        
        self.post_init()
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        事前学習済みモデルをロードします。
        
        Args:
            pretrained_model_name_or_path: 事前学習済みモデルのパスまたは名前
            
        Returns:
            LISAForCausalLM: ロードされたモデル
        """
        logger.info(f"LISAForCausalLM.from_pretrainedを呼び出し: {pretrained_model_name_or_path}")
        
        # 重要なパラメータを抽出
        seg_token_idx = kwargs.pop("seg_token_idx", None)
        vision_pretrained = kwargs.pop("vision_pretrained", None)
        out_dim = kwargs.pop("out_dim", 256)
        train_mask_decoder = kwargs.pop("train_mask_decoder", False)
        
        # LISA特有のパラメータを処理
        ce_loss_weight = kwargs.pop("ce_loss_weight", 1.0)
        dice_loss_weight = kwargs.pop("dice_loss_weight", 1.0)
        bce_loss_weight = kwargs.pop("bce_loss_weight", 1.0)
        use_mm_start_end = kwargs.pop("use_mm_start_end", True)
        
        # まずGemma3モデルをロード
        gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # 新しいインスタンスを作成
        config = GemmaLISAConfig()
        config.mm_gemma_path = pretrained_model_name_or_path
        # Gemma3の設定をコピー
        config.gemma_config = gemma_model.config
            
        # 重要：セグメントトークンIDの設定
        if seg_token_idx is not None:
            config.seg_token_idx = seg_token_idx
            logger.info(f"設定にセグメントトークンIDを設定: {seg_token_idx}")
                
        # SAMチェックポイントパス
        sam_checkpoint = None
        if vision_pretrained is not None:
            logger.info(f"SAMチェックポイントパス: {vision_pretrained}")
            sam_checkpoint = vision_pretrained
        
        # モデルパラメータの保存
        config.out_dim = out_dim
        config.train_mask_decoder = train_mask_decoder
        
        # モデルの初期化
        model = cls(config, sam_checkpoint=sam_checkpoint)
        
        # カスタム属性の設定
        model.ce_loss_weight = ce_loss_weight
        model.dice_loss_weight = dice_loss_weight
        model.bce_loss_weight = bce_loss_weight
        
        # 言語モデルにGemma3モデルを設定
        model.language_model = gemma_model
        model.lisa.model.lisa_model = gemma_model
        
        return model
    
    def get_model(self):
        """モデルのインスタンスを返します"""
        return self.lisa.model
    
    def get_vision_tower(self):
        """ビジョンタワーを返します"""
        return self.lisa.model.get_vision_tower()
    
    def get_language_model(self):
        """言語モデルを返します"""
        return self.language_model
    
    def get_input_embeddings(self):
        """
        入力埋め込み層を取得します。
        
        この実装は、言語モデルの入力埋め込み層を返します。
        トークン埋め込みのサイズ変更に必要です。
        """
        return self.language_model.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        """
        入力埋め込み層を設定します。
        
        Args:
            value: 新しい埋め込み層
        """
        self.language_model.set_input_embeddings(value)
    
    def initialize_lisa_modules(self, model_args, fsdp=None):
        """
        LISAモジュールを初期化します
        
        Args:
            model_args: モデル初期化のための引数
            fsdp: 完全分散データ並列処理の設定（存在する場合）
        """
        # SAMのチェックポイントを確認
        sam_checkpoint = None
        if hasattr(model_args, "sam_checkpoint") and model_args.sam_checkpoint is not None:
            sam_checkpoint = model_args.sam_checkpoint
        
        # SAMのフリーズ設定
        freeze_sam = True
        if hasattr(model_args, "freeze_sam") and not model_args.freeze_sam:
            freeze_sam = False
        
        # LISAモジュールを初期化
        self.lisa.model.initialize_lisa_modules(model_args, sam_checkpoint, freeze_sam)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        masks: Optional[torch.FloatTensor] = None,
        offset: Optional[int] = 0,
        last_hidden_state_for_masks = None,
        return_dict: Optional[bool] = None,
        image_paths: Optional[List[str]] = None,  # 画像パスを引数として受け入れる（実際には使用しない）
        # データセットから渡される追加引数（実際には使用しない）
        masks_list: Optional[List] = None,
        label_list: Optional[List] = None,
        resize_list: Optional[List] = None,
        questions_list: Optional[List] = None,
        sampled_classes_list: Optional[List] = None,
        inference: Optional[bool] = False,
        conversation_list: Optional[List] = None,
        pixel_values: Optional[torch.FloatTensor] = None,  # Gemma3用の画像テンソル（実際にはimagesを使用）
        **kwargs,  # その他の不明な引数を受け取るためのキャッチオール
    ):
        """
        モデルのフォワードパス
        
        Args:
            input_ids: 入力ID
            attention_mask: アテンションマスク
            past_key_values: 過去のキーと値のキャッシュ
            inputs_embeds: 入力の埋め込み
            labels: 学習ラベル
            use_cache: キャッシュを使用するかどうか
            output_attentions: アテンション出力を返すかどうか
            output_hidden_states: 隠れ状態を返すかどうか
            images: 入力画像
            image_sizes: 画像サイズ
            masks: セグメンテーションマスク
            offset: オフセット
            last_hidden_state_for_masks: マスク生成のための最後の隠れ状態
            return_dict: 辞書形式で結果を返すかどうか
            image_paths: 画像パス（実際には使用しない）
            masks_list: データセットから渡される追加引数（実際には使用しない）
            label_list: データセットから渡される追加引数（実際には使用しない）
            resize_list: データセットから渡される追加引数（実際には使用しない）
            questions_list: データセットから渡される追加引数（実際には使用しない）
            sampled_classes_list: データセットから渡される追加引数（実際には使用しない）
            inference: データセットから渡される追加引数（実際には使用しない）
            conversation_list: データセットから渡される追加引数（実際には使用しない）
            pixel_values: Gemma3用の画像テンソル（実際にはimagesを使用）
            
        Returns:
            モデルの出力
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # LLMモデルの出力とSAMの出力を格納する変数
        lisa_outputs = None
        sam_outputs = None
        
        # masks_listが提供されており、masksがNoneの場合、masks_listをmasksとして使用
        if masks is None and masks_list is not None:
            masks = masks_list
        
        # 画像の処理とLLMの入力準備
        if inputs_embeds is None and images is not None and input_ids is not None:
            # 画像の埋め込みを準備
            # logger.info(f"画像の処理中: {images.shape}")
            device = input_ids.device
            
            # 画像をSAM用に変換
            images_sam = prepare_images_for_sam(images, device=device)
            
            # 画像埋め込みを取得
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            
            # [SEG]トークンの位置を特定
            seg_token_mask = input_ids == self.seg_token_idx
            seg_token_positions = seg_token_mask.nonzero()
            
            # SAMモデルでの処理
            if self.lisa.model.sam_model is not None and seg_token_mask.any():
                # SAMのエンコーダで画像特徴量を抽出
                with torch.no_grad():
                    # SAMのイメージエンコーダで画像特徴を取得
                    sam_image_features = self.lisa.model.sam_model.image_encoder(images_sam)
            
            # フォワードパス実行
            lisa_outputs = self.language_model(
                input_ids=None,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=input_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
            )
            
            # [SEG]トークンからのマスク生成
            if seg_token_mask.any() and self.lisa.model.sam_model is not None:
                last_hidden_state = lisa_outputs.hidden_states[-1]
                
                # [SEG]トークンの位置での特徴量を取得
                seg_hidden_states = []
                for b, s in seg_token_positions:
                    seg_hidden_states.append(last_hidden_state[b, s])
                
                # SEGトークンの埋め込みをスタックして変換用の入力形式に整形
                seg_hidden_states = torch.stack(seg_hidden_states)
                
                # text_hidden_fcsが初期化されているか確認し、なければエラー
                if self.lisa.model.text_hidden_fcs is None:
                    raise ValueError("text_hidden_fcsが初期化されていません。initialize_lisa_modulesを呼び出してください。")
                
                # テキスト埋め込みをSAMプロンプト埋め込みに変換
                seg_hidden_projected = self.lisa.model.text_hidden_fcs[0](seg_hidden_states)
                # プロンプトエンコーダの入力形式に整形
                text_embeds = seg_hidden_projected.unsqueeze(1)  # [N, 1, 256]
                
                # SAMのプロンプトエンコーダで処理
                sparse_embeddings, dense_embeddings = self.lisa.model.sam_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=text_embeds,
                )
                
                # SAMのマスクデコーダで予測
                masks, iou_predictions = self.lisa.model.sam_model.mask_decoder(
                    image_embeddings=sam_image_features,
                    image_pe=self.lisa.model.sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                
                sam_outputs = {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                }
        else:
            # 通常の言語モデル処理
            lisa_outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        
        # 損失計算
        loss = lisa_outputs.loss if hasattr(lisa_outputs, "loss") else None
        
        # マスク学習を行う場合の損失計算
        if masks is not None and sam_outputs is not None:
            # masksがリスト型の場合はテンソルに変換
            if isinstance(masks, list):
                # リスト内の要素がテンソルかどうかをチェック
                if all(isinstance(m, torch.Tensor) for m in masks):
                    # すべてがテンソルの場合はスタック
                    masks = torch.stack(masks)
                else:
                    # リスト内の要素にテンソル以外がある場合
                    # 各要素をテンソルに変換してからスタック
                    masks = torch.stack([m if isinstance(m, torch.Tensor) else torch.tensor(m, device=sam_outputs["masks"].device) for m in masks])
            
            # マスクの形状を調整（必要な場合）
            if masks.shape != sam_outputs["masks"].shape:
                # 予測と同じ形状になるように調整
                if len(masks.shape) < len(sam_outputs["masks"].shape):
                    # 次元を追加
                    masks = masks.unsqueeze(1)
                # デバイスを一致させる
                masks = masks.to(sam_outputs["masks"].device)
            
            num_masks = masks.shape[1] if len(masks.shape) > 1 else 1
            mask_bce_loss = sigmoid_ce_loss(sam_outputs["masks"], masks, num_masks=num_masks)
            mask_dice_loss = dice_loss(sam_outputs["masks"], masks, num_masks=num_masks)
            mask_loss = mask_bce_loss + mask_dice_loss
            
            # 言語モデルの損失とマスク損失を組み合わせる
            if loss is not None:
                loss = loss + 20.0 * mask_loss
            else:
                loss = 20.0 * mask_loss
        
        # 結果の返却
        return transformers.modeling_outputs.CausalLMOutputWithPast(
            loss=loss,
            logits=lisa_outputs.logits if hasattr(lisa_outputs, "logits") else None,
            past_key_values=lisa_outputs.past_key_values if hasattr(lisa_outputs, "past_key_values") else None,
            hidden_states=lisa_outputs.hidden_states if hasattr(lisa_outputs, "hidden_states") else None,
            attentions=lisa_outputs.attentions if hasattr(lisa_outputs, "attentions") else None,
        )
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """
        生成のための入力を準備します
        
        Args:
            input_ids: 入力ID
            past_key_values: 過去のキーと値
            attention_mask: アテンションマスク
            inputs_embeds: 入力埋め込み
            
        Returns:
            生成のための入力辞書
        """
        # キャッシュがある場合は最後のトークンIDのみを保持
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        
        # imagesパラメータを引き継ぐ
        images = kwargs.get("images", None)
        image_sizes = kwargs.get("image_sizes", None)
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
            "images": images,
            "image_sizes": image_sizes,
        }
    
    def save_pretrained(self, *args, **kwargs):
        """
        モデルの保存
        
        SAMモデルの部分は保存しないようにする
        """
        # 一時的にSAMモデルを取り外して保存
        tmp_sam_model = self.lisa.model.sam_model
        self.lisa.model.sam_model = None
        
        # 元のメソッドで保存
        super().save_pretrained(*args, **kwargs)
        
        # SAMモデルを戻す
        self.lisa.model.sam_model = tmp_sam_model 
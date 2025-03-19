from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Gemma3ForConditionalGeneration, AutoConfig, AutoTokenizer, BitsAndBytesConfig

from model.gemma3.constants import (DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN,
                                 DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN,
                                 IMAGE_TOKEN_INDEX, IGNORE_INDEX)
from model.segment_anything import build_sam_vit_h


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


class GemmaLISAMetaModel:
    """Gemma3とSAMを統合するためのメタモデル"""
    
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(GemmaLISAMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):
        """LISAモジュールを初期化します"""
        # SAMモデルをロード
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # テキスト特徴量をSAMプロンプト用に変換する射影層
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class GemmaLISAForCausalLM(Gemma3ForConditionalGeneration, GemmaLISAMetaModel):
    """Gemma3とSAMを統合したLISAモデル"""
    
    def __init__(
        self,
        config,
        **kwargs,
    ):
        if not hasattr(config, "train_mask_decoder"):
            # Gemma3固有の設定
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
            
        self.seg_token_idx = kwargs.pop("seg_token_idx")
        
        # 親クラスの初期化
        Gemma3ForConditionalGeneration.__init__(self, config)
        GemmaLISAMetaModel.__init__(self, config, **kwargs)
        
    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        """SAMの視覚エンコーダで画像埋め込みを取得"""
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings
    
    def forward(
        self,
        images: torch.FloatTensor,  # SAM用の高解像度画像
        pixel_values: torch.FloatTensor,  # Gemma3用の画像
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        inference: bool = False,
        **kwargs,
    ):
        """フォワードパス"""
        # SAMの視覚エンコーダで画像埋め込みを取得
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1

        # [SEG]トークンの位置を特定
        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().to(input_ids.device),
            ],
            dim=1,
        )
        # モデル入力用の画像トークン位置調整（画像は先頭にあると仮定）
        seg_token_mask = torch.cat(
            [torch.zeros((seg_token_mask.shape[0], 255)).bool().to(input_ids.device), seg_token_mask],
            dim=1,
        )

        if inference:
            # 推論モード
            n_batch = 1
            length = input_ids.shape[0]
            assert pixel_values.shape[0] == 1
            pixel_values_extend = pixel_values.expand(length, -1, -1, -1).contiguous()

            output_hidden_states = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                # Gemma3モデルのフォワードパス
                output_i = super().forward(
                    pixel_values=pixel_values_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                )
                output_hidden_states.append(output_i.hidden_states)
                torch.cuda.empty_cache()

            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = None

        else:
            # 訓練モード
            pixel_values_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                pixel_values_i = (
                    pixel_values[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                pixel_values_list.append(pixel_values_i)
            pixel_values = torch.cat(pixel_values_list, dim=0)

            # Gemma3モデルのフォワードパス
            output = super().forward(
                pixel_values=pixel_values,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states

        # テキスト特徴量をSAMプロンプト用に変換
        hidden_states = []
        assert len(self.text_hidden_fcs) == 1
        hidden_states.append(self.text_hidden_fcs[0](output_hidden_states[-1]))

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]

        # セグメントトークンのオフセットを計算
        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().to(input_ids.device), seg_token_offset], dim=0
        )

        seg_token_offset = seg_token_offset[offset]

        # 予測埋め込みを整理
        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_

        # SAMでマスク予測
        multimask_output = False
        pred_masks = []
        for i in range(len(pred_embeddings)):
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            low_res_masks, iou_predictions = self.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            pred_mask = self.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            pred_masks.append(pred_mask[:, 0])

        model_output = output
        gt_masks = masks_list

        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
            }

        # 損失計算
        output = model_output.logits
        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]

            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        loss = ce_loss + mask_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }

    def evaluate(
        self,
        pixel_values,
        images,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        tokenizer=None,
    ):
        """評価用メソッド"""
        with torch.no_grad():
            outputs = self.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences

            # [SEG]トークンの位置を特定
            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
            seg_token_mask = torch.cat(
                [
                    torch.zeros((seg_token_mask.shape[0], 255)).bool().to(seg_token_mask.device),
                    seg_token_mask,
                ],
                dim=1,
            )

            hidden_states = []
            hidden_states.append(self.text_hidden_fcs[0](output_hidden_states))
            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[seg_token_mask]

            batch_size = len(resize_list)

            image_embeddings = self.get_visual_embs(images)

            multimask_output = True
            pred_masks = []
            for i in range(batch_size):
                if i < len(pred_embeddings):
                    (
                        sparse_embeddings,
                        dense_embeddings,
                    ) = self.visual_model.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                        text_embeds=pred_embeddings[i].unsqueeze(0).unsqueeze(1),
                    )
                    low_res_masks, iou_predictions = self.visual_model.mask_decoder(
                        image_embeddings=image_embeddings[i].unsqueeze(0),
                        image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=multimask_output,
                    )

                    masks = self.visual_model.postprocess_masks(
                        low_res_masks,
                        input_size=resize_list[i],
                        original_size=original_size_list[i],
                    )
                    masks = masks.squeeze(1)

                    # 最良のマスクを返す
                    max_iou_idx = iou_predictions.argmax(dim=1)
                    mask = []
                    for j, max_idx in enumerate(max_iou_idx):
                        mask.append(masks[j][max_idx])
                    mask = torch.stack(mask, dim=0)
                    pred_masks.append(mask.cpu())
                else:
                    # セグメントトークンがない場合
                    mask_shape = (1, *original_size_list[i])
                    pred_masks.append(torch.zeros(mask_shape))

            texts = tokenizer.batch_decode(output_ids, skip_special_tokens=False)

            return output_ids, texts, pred_masks


# ラッパークラス（互換性のため）
class LISAForCausalLM(GemmaLISAForCausalLM):
    """GemmaLISAForCausalLMの別名（互換性のため）"""
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name,
        revision=None,
        sam_checkpoint=None,
        seg_token_idx=None,
        torch_dtype=None,
        low_cpu_mem_usage=False,
        vision_tower=None,
        vision_pretrained=None,
        out_dim=256,
        ce_loss_weight=1.0,
        dice_loss_weight=0.5,
        bce_loss_weight=2.0,
        train_mask_decoder=True,
        load_in_8bit=False,
        load_in_4bit=False,
        quantization_config=None,
        device_map=None,
        **kwargs,
    ):
        """
        プリトレーニング済みモデルからインスタンスを作成
        
        Args:
            pretrained_model_name: Gemmaモデル名
            revision: モデルのリビジョン
            sam_checkpoint: SAMチェックポイントパス
            seg_token_idx: セグメンテーショントークンのインデックス
            torch_dtype: Torchのデータ型
            low_cpu_mem_usage: 低CPUメモリ使用フラグ
            vision_tower: ビジョンタワー（Gemmaでは不要）
            vision_pretrained: SAM視覚エンコーダの重み
            out_dim: 出力次元
            ce_loss_weight: クロスエントロピーロスの重み
            dice_loss_weight: Diceロスの重み
            bce_loss_weight: BCEロスの重み
            train_mask_decoder: マスクデコーダを学習するかどうか
            load_in_8bit: 8bitロードフラグ
            load_in_4bit: 4bitロードフラグ
            quantization_config: 量子化設定
            device_map: デバイスマップ
            
        Returns:
            LISAForCausalLM: モデルインスタンス
        """
        # 量子化設定の構成
        if quantization_config is None:
            if load_in_8bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            elif load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype or torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
        
        # Gemma3の設定を構成
        try:
            # モデル引数の構成
            model_args = {
                "torch_dtype": torch_dtype,
                "low_cpu_mem_usage": low_cpu_mem_usage,
                "seg_token_idx": seg_token_idx,
                "vision_pretrained": vision_pretrained,
                "out_dim": out_dim,
                "ce_loss_weight": ce_loss_weight,
                "dice_loss_weight": dice_loss_weight,
                "bce_loss_weight": bce_loss_weight,
                "train_mask_decoder": train_mask_decoder,
                "quantization_config": quantization_config,
                "device_map": device_map,
                **kwargs,
            }
            
            # テキスト設定を準備
            config = AutoConfig.from_pretrained(pretrained_model_name, trust_remote_code=True)
            
            # Gemma3のテキスト設定が必要な場合の対応
            if hasattr(config, "text_config"):
                text_config = config.text_config
                # 必要な属性をトップレベルにコピー
                config.vocab_size = getattr(text_config, "vocab_size", None)
                config.hidden_size = getattr(text_config, "hidden_size", None)
                config.num_hidden_layers = getattr(text_config, "num_hidden_layers", None)
                config.sliding_window_pattern = getattr(text_config, "sliding_window_pattern", 6)
            
            # ベースモデルとしてロード
            from transformers import AutoModelForCausalLM
            base_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name,
                trust_remote_code=True,
                config=config,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=low_cpu_mem_usage,
                quantization_config=quantization_config,
                device_map=device_map,
                revision=revision,
            )
            
            # 新しいモデルインスタンスを作成
            model = cls(config, **model_args)
            
            # ベースモデルの状態辞書をロード
            model.load_state_dict(base_model.state_dict(), strict=False)
            
            # LISAモジュールを初期化
            model.initialize_lisa_modules(config)
            
            return model
            
        except Exception as e:
            print(f"Error loading Gemma model: {e}")
            import traceback
            traceback.print_exc()
            raise e

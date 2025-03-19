from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)

from .gemma3.model.gemma3_model import GemmaLISAForCausalLM
from .segment_anything import build_sam_vit_h


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
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
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


class LISAForCausalLM(GemmaLISAForCausalLM):
    """
    LISA + Gemma + SAM統合モデル
    オリジナルLISAコードの互換性を維持するためのラッパークラス
    """
    
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
        
        # モデル引数の構成
        model_args = {
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": low_cpu_mem_usage,
            "seg_token_idx": seg_token_idx,
            "sam_checkpoint": sam_checkpoint,
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
        
        # Gemmaモデルとしてロード
        from transformers import AutoConfig, AutoModelForCausalLM
        
        try:
            config = AutoConfig.from_pretrained(pretrained_model_name, revision=revision)
            
            # ベースモデルをロード
            base_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=low_cpu_mem_usage,
                quantization_config=quantization_config,
                device_map=device_map,
                revision=revision,
            )
            
            # クラスのインスタンス化（configからhidden_sizeなどを取得）
            model = cls(config, **model_args)
            
            # モデルの状態辞書をロード
            model.load_state_dict(base_model.state_dict(), strict=False)
            
            # プロセッサをロード
            from model.gemma3.mm_utils import DummyProcessor
            from transformers import AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
            model.processor = DummyProcessor(tokenizer)
            
            return model
            
        except Exception as e:
            print(f"Error loading Gemma model: {e}")
            import traceback
            traceback.print_exc()
            raise e 
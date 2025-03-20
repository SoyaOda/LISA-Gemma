# Gemma3対応のLISAモデル
from .GemmaLISA.modeling import LISAForCausalLM, LISAModel
from .GemmaLISA.modeling_base import LISAPreTrainedModel
from .GemmaLISA.losses import dice_loss, sigmoid_ce_loss
from .GemmaLISA.utils import prepare_images_for_sam

# インポートエラーを回避するため設定クラスをインラインで定義
from transformers import AutoConfig, PretrainedConfig
from typing import Optional, Dict, Any, Union

class GemmaLISAConfig(PretrainedConfig):
    """
    GemmaLISAモデルの設定クラス。
    
    GemmaとSAMの統合モデルのための設定を管理します。
    """
    
    model_type = "gemma_lisa"
    
    def __init__(
        self,
        # 言語モデル関連
        mm_gemma_path: str = "google/gemma-3-4b-it",
        gemma_config: Optional[Dict[str, Any]] = None,
        
        # ビジョン関連
        mm_vision_tower: Optional[str] = None,
        
        # トークン関連
        seg_token_idx: int = None,
        
        # その他のパラメータ
        padding_idx: int = 0,
        initializer_range: float = 0.02,
        
        # 変換・微調整関連
        mm_projector_type: str = "linear",
        mm_projector_hidden_size: int = 4096,
        mm_projector_depth: int = 1,
        mm_hidden_size: int = 4096,
        freeze_gemma: bool = True,
        
        # ビジョンエンコーダ設定
        vision_hidden_size: int = 1408,
        
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # 言語モデル設定
        self.mm_gemma_path = mm_gemma_path
        
        # もしgemma_configが指定されていなければ、パスからロード
        if gemma_config is None and mm_gemma_path is not None:
            self.gemma_config = AutoConfig.from_pretrained(mm_gemma_path).to_dict()
        else:
            self.gemma_config = gemma_config
        
        # ビジョン関連設定
        self.mm_vision_tower = mm_vision_tower
        
        # トークン関連
        self.seg_token_idx = seg_token_idx
        self.padding_idx = padding_idx
        
        # 変換・微調整関連
        self.mm_projector_type = mm_projector_type
        self.mm_hidden_size = mm_hidden_size
        self.mm_projector_hidden_size = mm_projector_hidden_size
        self.mm_projector_depth = mm_projector_depth
        self.initializer_range = initializer_range
        self.vision_hidden_size = vision_hidden_size
        self.freeze_gemma = freeze_gemma

__all__ = [
    'LISAForCausalLM',
    'LISAModel', 
    'LISAPreTrainedModel',
    'GemmaLISAConfig',
    'dice_loss',
    'sigmoid_ce_loss',
    'prepare_images_for_sam'
] 
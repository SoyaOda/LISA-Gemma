"""
GemmaLISA モデルの設定クラス
"""

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
        gemma_config: Optional[Union[Dict[str, Any], PretrainedConfig]] = None,
        
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
        """
        初期化関数
        
        Args:
            mm_gemma_path: Gemmaモデルのパス
            gemma_config: Gemma設定の辞書またはPretrainedConfigオブジェクト
            mm_vision_tower: ビジョンタワーモデルのパス
            seg_token_idx: セグメントトークンのインデックス
            padding_idx: パディングトークンのインデックス
            initializer_range: 初期化の範囲
            mm_projector_type: マルチモーダルプロジェクターのタイプ
            mm_projector_hidden_size: プロジェクターの隠れ層のサイズ
            mm_projector_depth: プロジェクターの深さ
            mm_hidden_size: マルチモーダルな隠れ層のサイズ
            freeze_gemma: Gemmaモデルをフリーズするかどうか
            vision_hidden_size: ビジョンエンコーダの隠れ層のサイズ
        """
        super().__init__(**kwargs)
        
        # 言語モデル設定
        self.mm_gemma_path = mm_gemma_path
        
        # もしgemma_configが指定されていなければ、パスからロード
        if gemma_config is None and mm_gemma_path is not None:
            self.gemma_config = AutoConfig.from_pretrained(mm_gemma_path)
        else:
            # 辞書の場合はPretrainedConfigオブジェクトに変換
            if isinstance(gemma_config, dict):
                self.gemma_config = PretrainedConfig.from_dict(gemma_config)
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

    @property
    def decoder_config(self):
        """
        GenerationConfigのためのデコーダ設定を提供します。
        これはgemma_configと同一です。
        
        Returns:
            PretrainedConfig: デコーダ設定
        """
        # 必ずPretrainedConfigオブジェクトを返す
        if not isinstance(self.gemma_config, PretrainedConfig) and isinstance(self.gemma_config, dict):
            return PretrainedConfig.from_dict(self.gemma_config)
        return self.gemma_config
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs) -> "GemmaLISAConfig":
        """
        事前学習済みのモデル名またはパスから設定をロードします。
        
        Args:
            pretrained_model_name_or_path: 事前学習済みモデル名またはパス
            **kwargs: 追加の引数
            
        Returns:
            GemmaLISAConfig: 設定インスタンス
        """
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        
        # Gemma設定をロード
        if "mm_gemma_path" in config_dict and config_dict.get("gemma_config") is None:
            config_dict["gemma_config"] = AutoConfig.from_pretrained(config_dict["mm_gemma_path"])
        
        # 設定インスタンスを生成
        return cls.from_dict(config_dict, **kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        設定を辞書形式で返します。
        
        Returns:
            Dict[str, Any]: 設定の辞書
        """
        output = super().to_dict()
        
        # gemma_configが辞書でない場合は変換
        if hasattr(self, "gemma_config") and self.gemma_config is not None:
            if hasattr(self.gemma_config, "to_dict"):
                output["gemma_config"] = self.gemma_config.to_dict()
            elif isinstance(self.gemma_config, dict):
                output["gemma_config"] = self.gemma_config
        
        return output 
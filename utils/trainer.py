import os
import gc
import math
import torch
import torch.nn as nn
import transformers
import logging
from transformers import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.trainer_pt_utils import get_parameter_names
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Union, Any, Tuple

from utils.utils import AverageMeter, dict_to_cuda


# ロガーの初期化
logger = logging.getLogger(__name__)


class GemmaLISATrainer(Trainer):
    """
    Gemma3とSAMを統合したLISAモデルのカスタムTrainer
    
    言語モデルの損失とセグメンテーション損失の両方を考慮して学習を行います。
    """
    
    def __init__(self, **kwargs):
        # 追加ハイパーパラメータを先に取り出す
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", 1.0)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", 2.0)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", 0.5)
        
        # 親クラスの初期化
        super().__init__(**kwargs)
        
        # 混合精度トレーニング用のスケーラーを初期化
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() and (self.args.fp16 or self.args.bf16) else None
        
        # GPUデバイス情報のログ
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"GemmaLISATrainer: {device_count}台のGPUを使用してトレーニングを開始します")
            for i in range(device_count):
                logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            logger.info("GemmaLISATrainer: CPU環境でトレーニングを開始します（警告: GPUを使用すると大幅に高速化されます）")
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        言語モデル損失とセグメンテーション損失を計算する
        
        Args:
            model: 学習するモデル
            inputs: 入力データ (ディクショナリ)
            return_outputs: 出力も返すかどうか
            
        Returns:
            損失値または (損失値, 出力) のタプル
        """
        # inputs内のテンソルをCUDAに転送
        inputs = dict_to_cuda(inputs)
        
        # モデルの出力を取得
        outputs = model(**inputs)
        
        # 総合損失
        loss = outputs["loss"]
        
        # 損失の内訳を記録
        if hasattr(outputs, "lm_loss") and outputs["lm_loss"] is not None:
            self.log({"lm_loss": outputs["lm_loss"].detach().cpu().item()})
        
        if hasattr(outputs, "mask_loss") and outputs["mask_loss"] is not None:
            self.log({"mask_loss": outputs["mask_loss"].detach().cpu().item()})
        
        if return_outputs:
            return loss, outputs
        return loss
    
    def _save_checkpoint(self, model, trial):
        """
        チェックポイントを保存する
        
        Args:
            model: 保存するモデル
            trial: Trialオブジェクト (HPO用)
        
        Returns:
            保存先のパス
        """
        # Trainer標準の保存処理を行う
        output_dir = super()._save_checkpoint(model, trial)
        
        # ここでSAM関連の追加コンポーネントのセーブやその他のカスタム処理を行うことも可能
        # 例: text_hidden_fcs など追加モジュールの保存
        
        return output_dir
    
    def _save(self, output_dir, state_dict=None, save_model=True, safe_serialization=True):
        """
        モデルを保存する
        
        Args:
            output_dir: 保存先ディレクトリ
            state_dict: 保存する状態辞書
            save_model: モデルを保存するかどうか
            safe_serialization: safetensorsを使用するかどうか
        """
        # safetensorsの保存エラーを回避するため、safe_serializationを無効化
        safe_serialization = False
        
        # ロギングレベルを一時的に変更して冗長な警告を抑制
        import logging
        import warnings
        
        # 元のロギングレベルを保存
        original_level = logging.getLogger().level
        original_transformers_level = logging.getLogger("transformers").level
        
        try:
            # ロギングレベルをERRORに設定して警告を抑制
            if safe_serialization:
                logging.getLogger().setLevel(logging.ERROR)
                logging.getLogger("transformers").setLevel(logging.ERROR)
                
                # safetensorsの警告を抑制
                warnings.filterwarnings("ignore", message="Some tensors share memory")
            
            logger.info(f"モデルを保存します: {output_dir} (safe_serialization={safe_serialization})")
            
            # Transformersの新しいバージョンでは_saveメソッドの引数が異なる
            if hasattr(super(), "_save") and callable(getattr(super(), "_save")):
                # 親クラスの_saveメソッドのシグネチャを確認
                import inspect
                signature = inspect.signature(super()._save)
                params = list(signature.parameters.keys())
                
                if len(params) >= 3 and "safe_serialization" in params:
                    # 新しいバージョン: 4引数
                    super()._save(output_dir, state_dict, save_model, safe_serialization)
                elif len(params) >= 3 and "save_model" in params:
                    # 中間バージョン: 3引数
                    super()._save(output_dir, state_dict, save_model)
                else:
                    # 古いバージョン: 2引数
                    super()._save(output_dir, state_dict)
            else:
                # 直接モデルを保存
                if state_dict is None:
                    state_dict = self.model.state_dict()
                
                if save_model:
                    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
                    self.model.config.save_pretrained(output_dir)
        except RuntimeError as e:
            # safetensorsエラーの場合、PyTorchの標準保存方法を使用（ログは最小限に）
            if "Some tensors share memory" in str(e):
                logger.info("safetensors形式での保存に失敗したため、PyTorch形式で保存します")
                
                if state_dict is None:
                    state_dict = self.model.state_dict()
                
                torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
                self.model.config.save_pretrained(output_dir)
            else:
                # その他のエラーは再発生
                raise e
        finally:
            # 元のロギングレベルを復元
            logging.getLogger().setLevel(original_level)
            logging.getLogger("transformers").setLevel(original_transformers_level)
            warnings.resetwarnings()
    
    def create_optimizer(self):
        """
        最適化アルゴリズムを作成する
        
        Returns:
            optimizer: 最適化アルゴリズム
        """
        if self.optimizer is None:
            # 重み減衰を適用するパラメータ名の集合を取得
            # LayerNormやバイアスには重み減衰を適用しない
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = {name for name in decay_parameters if "bias" not in name}
            
            # パラメータをグループ化
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]
            
            # オプティマイザの初期化
            optimizer_cls = (
                torch.optim.AdamW
                if self.args.optim == "adamw_torch"
                else transformers.optimization.AdamW
            )
            
            # オプティマイザを作成
            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
            
            # モデルのパラメータ名からパラメータ本体への辞書を作成（デバッグ・ログ用）
            param_name_map = {param: name for name, param in self.model.named_parameters()}
            
            # パラメータグループの情報を出力
            print("最適化アルゴリズムのパラメータグループ:")
            for i, pg in enumerate(self.optimizer.param_groups):
                print(f"  パラメータグループ {i}:")
                print(f"    学習率: {pg['lr']}")
                print(f"    Weight Decay: {pg.get('weight_decay', 0.0)}")
                print(f"    パラメータ数: {len(pg['params'])}")
                
                # 詳細なパラメータ名をログに残す場合はコメントを外す
                # param_names = [param_name_map.get(p, "unknown") for p in pg['params']]
                # print(f"    パラメータ: {param_names[:5]}... (合計 {len(param_names)} 個)")
        
        return self.optimizer
    
    def get_train_dataloader(self) -> DataLoader:
        """
        学習用のDataLoaderを取得する
        
        Returns:
            DataLoader: 学習用のDataLoader
        """
        # collate_fnが指定されていない場合のみDefaultDataCollatorを使用
        if self.data_collator is None:
            from utils.utils import collate_fn
            self.data_collator = collate_fn
        
        return super().get_train_dataloader()
    
    def log(self, logs: Dict[str, float], start_time=None) -> None:
        """
        ログを記録する
        
        Args:
            logs: ログデータ (辞書形式)
            start_time: 開始時間（オプション）、親クラスとの互換性のため
        """
        # GCを実行してメモリリークを防止
        if self.args.local_rank <= 0 and self.state.global_step % 100 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 親クラスのログ処理
        super().log(logs, start_time)
    
    def _get_checkpoint_path(self):
        """
        最新のチェックポイントパスを取得する
        
        Returns:
            str: 最新のチェックポイントパス、存在しない場合はNone
        """
        output_dir = self.args.output_dir
        if not os.path.exists(output_dir):
            return None
            
        # チェックポイントディレクトリが存在するか確認
        checkpoint_dirs = [
            d for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d)) and d.startswith(PREFIX_CHECKPOINT_DIR)
        ]
        
        if len(checkpoint_dirs) == 0:
            return None
            
        # チェックポイントディレクトリを番号順にソート
        checkpoint_dirs = sorted(
            checkpoint_dirs,
            key=lambda x: int(x.replace(PREFIX_CHECKPOINT_DIR + "-", ""))
        )
        
        # 最新のチェックポイントディレクトリを返す
        return os.path.join(output_dir, checkpoint_dirs[-1])
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        学習ステップを実行する
        
        Args:
            model: 学習するモデル
            inputs: 入力データ
            num_items_in_batch: バッチ内のアイテム数（Transformers 4.x API互換用）
            
        Returns:
            損失値
        """
        model.train()
        
        # エポック開始時またはステップが100の倍数の時にGPUメモリ情報を表示
        if self.state.global_step == 0 or self.state.global_step % 100 == 0:
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                logger.info(f"ステップ {self.state.global_step} - GPU メモリ使用状況: "
                           f"割り当て済み: {allocated:.2f}GB / {total_memory:.2f}GB "
                           f"(予約済み: {reserved:.2f}GB)")
        
        # 入力をCUDAに転送（CUDAが利用可能な場合のみ）
        inputs = dict_to_cuda(inputs)
        
        # 勾配をゼロにリセット
        self.optimizer.zero_grad()
        
        # 順伝播
        outputs = model(**inputs)
        loss = outputs["loss"]
        
        # 損失のスケーリング（混合精度学習時）
        if (self.args.fp16 or self.args.bf16) and self.scaler is not None:
            self.scaler.scale(loss).backward()
            
            # 勾配クリッピング（CUDAが利用可能な場合のみ）
            if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                # CUDAが利用可能な場合のみunscaleとclip_grad_normを実行
                if torch.cuda.is_available():
                    self.scaler.unscale_(self.optimizer)
                    self.accelerator.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # 通常の学習
            loss.backward()
            
            # 勾配クリッピング（CUDAが利用可能な場合のみ）
            if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0 and torch.cuda.is_available():
                self.accelerator.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                
            self.optimizer.step()
        
        # スケジューラのステップ
        self.lr_scheduler.step()
        
        # ログに損失を記録
        logs = {"loss": loss.detach().cpu().item()}
        
        # 個別の損失をログに記録
        if "lm_loss" in outputs and outputs["lm_loss"] is not None:
            logs["lm_loss"] = outputs["lm_loss"].detach().cpu().item()
        if "mask_loss" in outputs and outputs["mask_loss"] is not None:
            logs["mask_loss"] = outputs["mask_loss"].detach().cpu().item()
        
        # ログをTrainerクラスのログ機構に追加
        self.log(logs)
        
        return loss.detach() 
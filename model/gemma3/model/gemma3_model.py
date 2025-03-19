from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoProcessor, AutoTokenizer, 
                         AutoModelForCausalLM, AutoConfig,
                         Gemma3ForConditionalGeneration)

from model.segment_anything import build_sam_vit_h
from ..mm_utils import get_gemma_processor, GemmaImageProcessor
from ..utils import build_gemma_inputs, init_gemma_tokenizer


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


class GemmaLISAMetaModel(nn.Module):
    """Gemmaモデルとサムを統合したメタモデルクラス"""
    
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__()

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):
        # SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer - Gemma3の隠れ状態からSAMのプロンプト次元へ
        if hasattr(config, 'hidden_size'):
            in_dim = config.hidden_size
        elif hasattr(config, 'text_config'):
            in_dim = config.text_config.hidden_size
        else:
            # デフォルト値（Gemma3-4bの場合）
            in_dim = 2048
        
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


class GemmaLISAModel(GemmaLISAMetaModel):
    """GemmaとSAMを統合したモデルクラス"""
    
    def __init__(
        self,
        config,
        **kwargs,
    ):
        # 親クラスの初期化
        super(GemmaLISAModel, self).__init__(config, **kwargs)
        
        # Configure model parameters
        self.config.use_cache = False
        
        # SAM関連の初期化
        self.sam_checkpoint = kwargs.get("sam_checkpoint", None)
        self.seg_token_idx = kwargs.get("seg_token_idx", None)
        
        # Gemma3モデル
        self.gemma_model = None
        
        if self.sam_checkpoint is not None:
            self.initialize_lisa_modules(self.config)
    
    def forward(self, *args, **kwargs):
        """
        フォワードメソッド - Gemma3モデルに処理を委譲
        """
        if self.gemma_model is not None:
            return self.gemma_model(*args, **kwargs)
        else:
            raise ValueError("gemma_modelが初期化されていません")

    def encode_images(self, pixel_values):
        """
        Gemma3のビジョンエンコーダで画像をエンコードする
        """
        if self.gemma_model is not None and hasattr(self.gemma_model, "vision_model"):
            # Gemma3モデルの内部エンコーダを使用
            vision_outputs = self.gemma_model.vision_model(pixel_values)
            return vision_outputs.last_hidden_state
        else:
            raise ValueError("Gemma3モデルが初期化されていないか、vision_modelが見つかりません")
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, sam_checkpoint=None, **kwargs):
        """
        GemmaLISAModelのプリトレーニング済みモデルをロードするメソッド
        sam_checkpointパラメータを受け取り、SAMをロードします
        """
        # LISAに特化したパラメータを抽出
        seg_token_idx = kwargs.pop("seg_token_idx", None)
        train_mask_decoder = kwargs.pop("train_mask_decoder", True)
        out_dim = kwargs.pop("out_dim", 256)
        
        # Gemma3ForConditionalGenerationの設定を取得
        print(f"Gemma3モデルをロード中: {pretrained_model_name_or_path}")
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=True
        )
        
        # text_configからhidden_sizeなどの属性をトップレベルにコピー
        if hasattr(config, 'text_config'):
            config.hidden_size = getattr(config.text_config, 'hidden_size', 2048)
            config.vocab_size = getattr(config.text_config, 'vocab_size', 262144)
            config.num_hidden_layers = getattr(config.text_config, 'num_hidden_layers', 18)
            config.num_attention_heads = getattr(config.text_config, 'num_attention_heads', 8)
            print(f"text_configからhidden_size={config.hidden_size}を設定しました")
        
        # LISA用のカスタム設定を追加
        config.seg_token_idx = seg_token_idx
        config.train_mask_decoder = train_mask_decoder
        config.out_dim = out_dim
        
        # GemmaLISAModelインスタンスを作成するためのパラメータ
        lisa_model_args = {
            "sam_checkpoint": sam_checkpoint,
            "seg_token_idx": seg_token_idx,
            "train_mask_decoder": train_mask_decoder,
            "out_dim": out_dim,
        }
        
        # LISAモデルのインスタンス化
        lisa_model = cls(config, **lisa_model_args)
        
        # Gemma3モデルをロード
        # SAM関連のパラメータをkwargsから削除してGemma3モデルに渡さないようにする
        gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            trust_remote_code=True,
            **kwargs  # SAM関連パラメータは既に取り除いたkwargsのみを渡す
        )
        
        # Gemma3モデルのattributesをコピー
        lisa_model.gemma_model = gemma_model
        
        return lisa_model


class GemmaLISAForCausalLM(nn.Module):
    """
    Gemmaをベースとしたモデル
    SAMを統合してセグメンテーション機能を持たせる
    """
    
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__()
        
        # 引数からSAM関連の設定を取得
        if not hasattr(config, "train_mask_decoder"):
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", 1.0)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", 0.5)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", 2.0)
            
        # セグメンテーショントークンのインデックスを取得
        self.seg_token_idx = kwargs.pop("seg_token_idx", None)
        
        # SAMのチェックポイントパス
        self.sam_checkpoint = kwargs.pop("sam_checkpoint", None)
        
        # モデルの初期化
        self.model = GemmaLISAModel(config, **kwargs)
        
        # プロセッサの初期化
        self.processor = None
        
        # SAMモジュールの初期化
        self.initialize_lisa_modules(config, **kwargs)
    
    def initialize_lisa_modules(self, config, **kwargs):
        """
        LISAモジュール（SAM統合）を初期化する
        """
        # SAM
        self.visual_model = build_sam_vit_h(self.sam_checkpoint)
        for param in self.visual_model.parameters():
            param.requires_grad = False
            
        # マスクデコーダを学習する場合
        train_mask_decoder = kwargs.get("train_mask_decoder", True)
        if train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # テキスト特徴からセグメンテーション用埋め込みへの変換層
        if hasattr(config, 'hidden_size'):
            in_dim = config.hidden_size
        elif hasattr(config, 'text_config'):
            in_dim = config.text_config.hidden_size
        else:
            # デフォルト値（Gemma3-4bの場合）
            in_dim = 2048
        
        out_dim = kwargs.get("out_dim", 256)  # SAMのプロンプト次元
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True
    
    def load_processor(self, model_name):
        """
        モデル名からプロセッサをロードする
        
        Args:
            model_name: Gemmaモデル名
        """
        self.processor = get_gemma_processor(model_name)
        # トークナイザの初期化
        if hasattr(self.processor, "tokenizer"):
            init_gemma_tokenizer(self.processor.tokenizer)
        
        # 画像プロセッサの設定
        self.image_processor = GemmaImageProcessor(self.processor)
    
    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        """
        画像からSAMの視覚エンコーダで特徴量を抽出する
        
        Args:
            pixel_values: 画像ピクセル値 (B, C, H, W)
            
        Returns:
            torch.Tensor: SAMの視覚特徴量
        """
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
    
    def forward(self, **kwargs):
        """
        モデルの順伝播処理
        通常の推論かカスタムフォワードかを判断
        """
        if "past_key_values" in kwargs:
            return self.model.gemma_model(**kwargs)
        
        if "images" in kwargs:  # LISAの形式でimages引数がある場合
            return self.model_forward(**kwargs)
        
        return self.model.gemma_model(**kwargs)
    
    def model_forward(
        self,
        images: torch.FloatTensor,
        pixel_values: torch.FloatTensor,
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
        """
        カスタムフォワード処理
        言語モデルとセグメンテーションモデルを組み合わせる
        
        Args:
            images: SAM用の高解像度画像
            pixel_values: Gemma3用の224x224にリサイズされた画像
            input_ids: 入力テキストのトークンID
            labels: 言語モデル用のラベル
            attention_masks: アテンションマスク
            offset: バッチ内の各サンプルの開始インデックス
            masks_list: セグメンテーションマスクのリスト
            label_list: ラベルリスト
            resize_list: リサイズ情報のリスト
            inference: 推論モードかどうか
            
        Returns:
            出力結果（テキスト生成＋セグメンテーション）
        """
        # SAM用の画像エンコーディング
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1
        
        # セグメントトークンのマスク作成
        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )
        # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
        seg_token_mask = torch.cat(
            [torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(), seg_token_mask],
            dim=1,
        )
        
        # 推論モードと学習モードで処理を分岐
        if inference:
            n_batch = 1
            length = input_ids.shape[0]
            assert pixel_values.shape[0] == 1
            pixel_values_extend = pixel_values.expand(length, -1, -1, -1).contiguous()

            output_hidden_states = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                
                # Gemma3モデルの入力を構築
                model_inputs = build_gemma_inputs(
                    input_ids=input_ids[start_i:end_i],
                    pixel_values=pixel_values_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    output_hidden_states=True,
                )
                
                # Gemma3モデルの順伝播
                output_i = self.model.gemma_model(**model_inputs)
                output_hidden_states.append(output_i.hidden_states)
                torch.cuda.empty_cache()

            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = None

        else:
            # 学習モード
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
            
            # Gemma3モデルの入力を構築
            model_inputs = build_gemma_inputs(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_masks,
                labels=labels,
                output_hidden_states=True,
            )
            
            # Gemma3モデルの順伝播
            output = self.model.gemma_model(**model_inputs)
            output_hidden_states = output.hidden_states

        # テキスト特徴から埋め込みベクトルを抽出
        hidden_states = []
        assert len(self.text_hidden_fcs) == 1
        hidden_states.append(self.text_hidden_fcs[0](output_hidden_states[-1]))
        
        # セグメントトークン位置の特徴を抽出
        pred_embeddings = []
        for i, hs in enumerate(hidden_states):
            pred_embeddings.append(hs[seg_token_mask])
        
        # 予測埋め込みがなければテキスト生成のみを返す
        num_preds = len(pred_embeddings[0])
        if num_preds == 0:
            ret = {"loss": output.loss} if output is not None else {}
            return ret
        
        # セグメンテーションマスクを予測
        mask_preds = []
        for i, embeddings in enumerate(pred_embeddings):
            mask_pred = self.visual_model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=embeddings.unsqueeze(1),
                dense_prompt_embeddings=None,
                multimask_output=False,
            )["low_res_logits"]
            
            mask_preds.append(mask_pred)
        
        # マスク予測結果を処理
        mask_preds = torch.cat(mask_preds)
        
        # 学習時のロス計算
        if not inference:
            # 言語生成ロス
            loss = output.loss * self.ce_loss_weight
            
            # マスク予測のロス計算
            if masks_list[0] is not None:
                mask_for_pred = []
                num_mask_per_batch = []
                
                for idx in range(len(masks_list)):
                    gt_masks = masks_list[idx]
                    num_mask = gt_masks.shape[0]
                    num_mask_per_batch.append(num_mask)
                    
                    if len(mask_for_pred) == 0:
                        mask_for_pred = gt_masks
                    else:
                        mask_for_pred = torch.cat([mask_for_pred, gt_masks], dim=0)
                
                total_num_masks = sum(num_mask_per_batch)
                
                # セグメンテーションロス（BCE + Dice）
                loss_bce = sigmoid_ce_loss(
                    mask_preds,
                    mask_for_pred,
                    num_masks=total_num_masks,
                )
                
                loss_dice = dice_loss(
                    mask_preds,
                    mask_for_pred,
                    num_masks=total_num_masks,
                )
                
                loss_mask = loss_bce * self.bce_loss_weight + loss_dice * self.dice_loss_weight
                
                # 合計ロス
                loss = loss + loss_mask
            
            return {"loss": loss}
        
        else:
            # 推論時の処理
            return {"pred_masks": mask_preds, "hidden_states": output_hidden_states}
    
    def generate(self, *args, **kwargs):
        """
        Gemma3モデルのgenerate関数を呼び出すラッパー
        """
        return self.model.gemma_model.generate(*args, **kwargs)
    
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
        """
        モデルによる評価処理（推論）
        
        Args:
            pixel_values: Gemma3用の画像テンソル
            images: SAM用の高解像度画像テンソル
            input_ids: 入力テキストのトークンID
            resize_list: リサイズ情報のリスト
            original_size_list: 元画像サイズのリスト
            max_new_tokens: 生成する最大トークン数
            tokenizer: トークナイザ
            
        Returns:
            dict: 生成テキストとセグメンテーションマスク
        """
        # 入力準備
        attention_mask = torch.ones_like(input_ids)
        offset = torch.Tensor([0, 1]).long().cuda()
        label_dummy = None
        masks_dummy = [None]
        label_list_dummy = [None]
        
        # 自己回帰生成
        with torch.no_grad():
            outputs = self.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
            )
        
        # 生成されたトークンをデコード
        outputs = outputs[:, input_ids.shape[1]:] # 入力を除いた生成部分だけを抽出
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # セグメンテーション実行
        with torch.no_grad():
            # セグメンテーション予測
            seg_results = self.model_forward(
                images=images,
                pixel_values=pixel_values,
                input_ids=outputs,
                labels=label_dummy,
                attention_masks=torch.ones_like(outputs),
                offset=offset,
                masks_list=masks_dummy,
                label_list=label_list_dummy,
                resize_list=resize_list,
                inference=True,
            )
        
        # セグメンテーション結果の後処理
        if "pred_masks" in seg_results:
            mask_pred = seg_results["pred_masks"]
            # マスクのリサイズ、後処理など
            masks = []
            for i, (pred, (h, w), original_size) in enumerate(zip(mask_pred, resize_list, original_size_list)):
                pred = pred.unsqueeze(0)
                pred = F.interpolate(
                    pred, (h, w), mode="bilinear", align_corners=False
                )
                
                # マスクを元のサイズにリサイズ
                pred = F.interpolate(
                    pred, original_size, mode="bilinear", align_corners=False
                )
                pred = pred.sigmoid().gt(0.5)
                masks.append(pred.squeeze())
        else:
            masks = None
        
        # 結果を返す
        return {"response": response, "masks": masks} 
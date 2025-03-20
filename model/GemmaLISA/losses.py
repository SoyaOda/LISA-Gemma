"""
GemmaLISA モデルの損失関数モジュール
"""

import torch
import torch.nn.functional as F


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


class DiceLoss(torch.nn.Module):
    """
    DICE損失のモジュール実装
    """
    def __init__(self, scale=1000, eps=1e-6):
        super().__init__()
        self.scale = scale
        self.eps = eps
        
    def forward(self, inputs, targets, num_masks=1, reduction="mean"):
        """
        DICE損失を計算する
        
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
        denominator = torch.sum(inputs, dim=-1) + torch.sum(targets, dim=-1) + self.eps
        
        loss = 1 - (numerator / denominator)
        
        if reduction == "none":
            return loss
        
        # バッチとマスクの平均を取る
        if num_masks == 0:
            return torch.tensor(0.0, device=inputs.device)
        
        return loss.sum() / (loss.shape[0] * num_masks) if reduction == "mean" else loss.sum()


class SigmoidCELoss(torch.nn.Module):
    """
    シグモイドクロスエントロピー損失のモジュール実装
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs, targets, num_masks=1, reduction="mean"):
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
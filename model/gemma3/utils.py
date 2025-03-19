import torch
import torch.nn as nn

from .constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, 
                      DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN)


def build_gemma_inputs(
    input_ids,
    pixel_values=None,
    attention_mask=None,
    labels=None,
    past_key_values=None,
    return_dict=None,
    **kwargs
):
    """
    Gemma3モデルへの入力を構築する関数

    Args:
        input_ids: トークナイズされた入力ID
        pixel_values: 画像ピクセル値
        attention_mask: アテンションマスク
        labels: 言語モデル学習用のラベル
        past_key_values: 過去の計算結果
        return_dict: 辞書形式で返すかどうか

    Returns:
        dict: Gemma3モデルに渡す引数の辞書
    """
    model_inputs = {}
    model_inputs["input_ids"] = input_ids
    if pixel_values is not None:
        model_inputs["pixel_values"] = pixel_values
    if attention_mask is not None:
        model_inputs["attention_mask"] = attention_mask
    if labels is not None:
        model_inputs["labels"] = labels
    if past_key_values is not None:
        model_inputs["past_key_values"] = past_key_values
    if return_dict is not None:
        model_inputs["return_dict"] = return_dict
    
    # その他のカスタム引数があれば追加
    for k, v in kwargs.items():
        model_inputs[k] = v
        
    return model_inputs


def expand_image_token(text, num_patch=256):
    """
    テキスト中の画像トークンを画像パッチトークンに展開する

    Args:
        text: 入力テキスト
        num_patch: 画像パッチの数

    Returns:
        str: 展開されたテキスト
    """
    if DEFAULT_IMAGE_TOKEN in text:
        text = text.replace(
            DEFAULT_IMAGE_TOKEN, 
            DEFAULT_IM_START_TOKEN 
            + DEFAULT_IMAGE_PATCH_TOKEN * num_patch 
            + DEFAULT_IM_END_TOKEN
        )
    elif DEFAULT_IM_START_TOKEN in text and DEFAULT_IM_END_TOKEN in text:
        start_idx = text.find(DEFAULT_IM_START_TOKEN)
        end_idx = text.find(DEFAULT_IM_END_TOKEN) + len(DEFAULT_IM_END_TOKEN)
        # start_tokenとend_tokenの間にパッチトークンを挿入
        text = (
            text[:start_idx + len(DEFAULT_IM_START_TOKEN)]
            + DEFAULT_IMAGE_PATCH_TOKEN * num_patch
            + text[end_idx - len(DEFAULT_IM_END_TOKEN):]
        )
    return text


def init_gemma_tokenizer(tokenizer):
    """
    Gemma3トークナイザを初期化する関数

    Args:
        tokenizer: Gemma3トークナイザ

    Returns:
        tokenizer: 初期化されたトークナイザ
    """
    # Gemma3のパディングトークンが定義されていない場合は設定
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    
    # 特殊トークンを追加
    added_tokens = []
    for token in [DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, 
                  DEFAULT_IMAGE_PATCH_TOKEN, "[SEG]"]:
        if token not in tokenizer.get_vocab():
            added_tokens.append(token)
    
    if added_tokens:
        tokenizer.add_tokens(added_tokens)
    
    return tokenizer 
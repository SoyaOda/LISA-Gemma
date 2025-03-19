#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gemma3+SAMモデル用の定数定義
"""

# デフォルトの画像トークン
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IM_START_TOKEN = "<im_start>"  
DEFAULT_IM_END_TOKEN = "<im_end>"

# トークナイザで無視するインデックス
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200

# システムプロンプト
SYSTEM_PROMPT = "You are a helpful visual assistant that can segment objects in images."

# 回答リスト
ANSWER_LIST = ["Yes.", "No.", "Maybe.", "I don't know."]

# 短い質問リスト
SHORT_QUESTION_LIST = [
    "What is this?", 
    "What do you see?", 
    "Can you describe this image?", 
    "What's in this picture?"
] 
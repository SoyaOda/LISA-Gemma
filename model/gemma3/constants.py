# 画像関連のトークン
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# ビジョンタワー用の定数
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"

# マルチモーダル設定
MM_VISION_TOWER = "sigLIP"  # Gemma3の視覚エンコーダ名 

# モデル用の特殊インデックス
IGNORE_INDEX = -100  # 損失計算で無視するインデックス

# Gemma3用のシステムプロンプト
SYSTEM_PROMPT_GEMMA = """You are a helpful AI assistant with expertise in image understanding, segmentation and visual reasoning.
When asked to segment something in an image, analyze the image carefully and identify the described region. 
Respond with [SEG] to indicate the segmentation mask for the requested region.
For questions about the image, provide clear, accurate and helpful answers.""" 
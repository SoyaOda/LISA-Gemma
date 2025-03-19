import torch
import torch.nn as nn

from .constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from transformers import AutoProcessor


def get_gemma_processor(model_name):
    """
    Gemma3のプロセッサを取得する関数
    
    Args:
        model_name: Gemma3のモデル名（huggingface hub上の名前）
        
    Returns:
        AutoProcessor: Gemma3のプロセッサインスタンス
    """
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        return processor
    except Exception as e:
        print(f"Error loading Gemma3 processor: {e}")
        # フォールバック：画像はSigLIPエンコーダに合わせて処理、トークナイザはそのまま使用
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return DummyProcessor(tokenizer)


class DummyProcessor:
    """
    Gemma3のAutoProcessorが使えない場合のフォールバック用クラス
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def apply_chat_template(self, messages, add_generation_prompt=True, return_tensors="pt", **kwargs):
        # 単純にトークナイザを使用（画像情報は別途処理）
        if isinstance(messages[0]["content"], list):
            # マルチモーダルメッセージの場合、テキスト部分だけ抽出
            text_content = []
            for message in messages:
                message_text = ""
                if isinstance(message["content"], list):
                    for content in message["content"]:
                        if content["type"] == "text":
                            message_text += content["text"]
                else:
                    message_text = message["content"]
                
                if message["role"] == "user":
                    text_content.append(f"USER: {message_text}")
                elif message["role"] == "assistant":
                    text_content.append(f"ASSISTANT: {message_text}")
            
            text = "\n".join(text_content)
            if add_generation_prompt:
                text += "\nASSISTANT: "
        else:
            # 通常のテキストメッセージの場合
            text = ""
            for message in messages:
                if message["role"] == "user":
                    text += f"USER: {message['content']}\n"
                elif message["role"] == "assistant":
                    text += f"ASSISTANT: {message['content']}\n"
            
            if add_generation_prompt:
                text += "ASSISTANT: "
        
        return self.tokenizer(text, return_tensors=return_tensors, **kwargs)


class GemmaImageProcessor:
    """
    Gemma3用の画像処理クラス
    SigLIPエンコーダに合わせた前処理を行う
    """
    def __init__(self, processor=None):
        self.processor = processor
        
    def preprocess_images(self, images):
        """
        画像をGemma3のSigLIPエンコーダに合わせて前処理する
        
        Args:
            images: PIL画像のリスト
            
        Returns:
            torch.Tensor: 前処理された画像テンソル(B, C, H, W)
        """
        if self.processor is not None:
            # プロセッサが利用可能な場合はそれを使用
            pixel_values = self.processor.image_processor(images, return_tensors="pt").pixel_values
            return pixel_values
        else:
            # フォールバック：基本的な前処理
            # SigLIPが224x224の入力サイズを想定
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])
            
            processed_images = []
            for image in images:
                processed_images.append(transform(image))
            
            return torch.stack(processed_images) 
import dataclasses
from typing import List, Tuple, Any

import transformers


@dataclasses.dataclass
class Conversation:
    """
    A class representing a conversation between a user and an assistant.
    """

    system: str
    roles: Tuple[str, str]
    messages: List[Tuple[str, str]]
    offset: int
    sep_style: str
    sep: str
    sep2: str = None
    
    def get_prompt(self):
        """Get the prompt for generation."""
        
        ret = self.system
        if self.sep_style == "gemma":
            for role, message in self.messages:
                if message:
                    ret += f"{role}\n{message}\n"
                else:
                    ret += f"{role}\n"
        elif self.sep_style == "llava_llama3":
            for role, message in self.messages:
                if message:
                    ret += f"{role}: {message}{self.sep}"
                else:
                    ret += f"{role}:"
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")
            
        return ret
    
    def append_message(self, role, message):
        """Append a new message."""
        self.messages.append([role, message])
    
    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=self.messages.copy(),
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
        )
    
    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep_style": self.sep_style,
            "sep": self.sep,
            "sep2": self.sep2,
        }


class ConversationLibrary:
    def __init__(self):
        self.default_conversation = None
        self.conv_templates = {}

        # Gemma3用テンプレート
        self.register_conv_template(
            Conversation(
                system="<|system|>\nYou are a helpful assistant that can understand images. Reply step by step.\n",
                roles=("<|user|>", "<|assistant|>"),
                messages=[],
                offset=2,
                sep_style="gemma",
                sep="",
            ),
            name="gemma_v1",
        )
        
        # LLaVA-LLaMA3用テンプレート (互換性のため維持)
        self.register_conv_template(
            Conversation(
                system="",
                roles=("USER", "ASSISTANT"),
                messages=[],
                offset=2,
                sep_style="llava_llama3",
                sep="\n",
            ),
            name="llava_llama3",
        )

    def register_conv_template(self, template, name):
        self.conv_templates[name] = template

    def apply_chat_template(
        self, messages, tokenizer, add_generation_prompt=True, return_tensors="pt", **kwargs
    ):
        """
        Gemma3のapply_chat_template互換メソッド
        """
        messages_processed = []
        for message in messages:
            if isinstance(message, dict):
                if message["role"] == "system":
                    continue  # システムメッセージは別途処理
                
                content = message["content"]
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if part["type"] == "text":
                            text_parts.append(part["text"])
                    
                    if message["role"] == "user":
                        messages_processed.append((self.default_conversation.roles[0], " ".join(text_parts)))
                    else:
                        messages_processed.append((self.default_conversation.roles[1], " ".join(text_parts)))
                else:
                    if message["role"] == "user":
                        messages_processed.append((self.default_conversation.roles[0], content))
                    else:
                        messages_processed.append((self.default_conversation.roles[1], content))
                        
        conv = self.default_conversation.copy()
        conv.messages = messages_processed
        
        prompt = conv.get_prompt()
        if add_generation_prompt:
            prompt += f"{conv.roles[1]}\n"
        
        inputs = tokenizer(prompt, return_tensors=return_tensors, **kwargs)
        return inputs


# シングルトンインスタンス
conv_lib = ConversationLibrary()
conv_lib.default_conversation = conv_lib.conv_templates["gemma_v1"]


def get_default_conv_template(name=None):
    if name is None:
        return conv_lib.default_conversation.copy()
    return conv_lib.conv_templates[name].copy() 
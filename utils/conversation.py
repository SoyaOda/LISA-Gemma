"""
Conversation prompt templates.
"""

import dataclasses
from enum import Enum, auto
from typing import Any, List, Optional, Dict, Sequence, Union


class SeparatorStyle(Enum):
    """区切り文字のスタイル"""
    SINGLE = auto()
    TWO = auto()
    GEMMA = auto()


@dataclasses.dataclass
class Conversation:
    """対話を表現するクラス"""
    
    # 発話ロールのリスト
    roles: List[str]
    
    # メッセージのリスト
    messages: List[List[str]]
    
    # 対応するオフセット
    offsets: List[List[int]]
    
    # システムプロンプト
    system: str
    
    # システムプロンプトトークン
    system_token: str = ""
    
    # 区切り記号
    sep: str = ""
    
    # 区切り記号2（TWOスタイルの場合）
    sep2: Optional[str] = None
    
    # 区切り文字のスタイル
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    
    # ファイル名（モデルディレクトリとしてハイパーパラメータの一部）
    version: str = "unknown"
    
    def __post_init__(self):
        if len(self.roles) <= 0:
            raise ValueError("少なくとも1つのロールが必要です")
        if not self.offsets:
            self.offsets = [[] for _ in range(len(self.messages))]
        
    def copy(self):
        return Conversation(
            roles=self.roles.copy(),
            messages=[[x for x in y] for y in self.messages],
            offsets=[[x for x in y] for y in self.offsets],
            system=self.system,
            system_token=self.system_token,
            sep=self.sep,
            sep2=self.sep2,
            sep_style=self.sep_style,
            version=self.version,
        )
    
    def append_message(self, role: str, message: str, offset: Optional[int] = None):
        """メッセージを追加"""
        if role not in self.roles:
            raise ValueError(f"未知のロール: {role}")
        
        self.messages.append([message])
        
        if offset is not None:
            self.offsets.append([offset])
        else:
            self.offsets.append([0])
    
    def get_prompt(self) -> str:
        """会話からプロンプトを生成"""
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system
            for i, (role, message) in enumerate(self.get_turns()):
                if i == 0 and message:
                    ret += message
                else:
                    ret += self.sep + role + ": " + message
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(self.get_turns()):
                if i == 0 and message:
                    ret += message
                else:
                    ret += role + ": " + message + seps[i % 2]
            return ret
        elif self.sep_style == SeparatorStyle.GEMMA:
            # Gemma3モデルの会話形式
            from .constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
            
            # システムプロンプトからスタート
            ret = "System: " + self.system
            
            # 各ターンの処理
            for i, (role, message) in enumerate(self.get_turns()):
                # Gemma3互換の画像トークン処理
                # DEFAULT_IMAGE_TOKENを<start_of_image>に置き換え
                if DEFAULT_IMAGE_TOKEN in message:
                    message = message.replace(DEFAULT_IMAGE_TOKEN, "<start_of_image>")
                elif DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN in message:
                    message = message.replace(
                        DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN, 
                        "<start_of_image>"
                    )
                
                # ターンを追加
                ret += "\n\n" + role + ": " + message
            
            return ret
        else:
            raise ValueError(f"未知の区切りスタイル: {self.sep_style}")
    
    def get_turns(self):
        """発話ターンの取得"""
        ret = []
        for i, role in enumerate(self.roles):
            for j, message in enumerate(self.messages):
                if message:
                    ret.append((role, message[0]))
        return ret


# Gemma3のモデル形式
gemma_v1 = Conversation(
    system="You are a helpful AI assistant.",
    roles=["User", "Assistant"],
    messages=[],
    offsets=[],
    sep_style=SeparatorStyle.GEMMA,
    version="gemma-v1",
)


# LLAVA形式をLISAで拡張したテンプレート（参考用）
llava_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=["Human", "Assistant"],
    messages=[], 
    offsets=[],
    sep_style=SeparatorStyle.TWO, 
    sep=" ", 
    sep2="</s>",
    version="llava-v1"
)


# 利用可能な会話テンプレート
conv_templates = {
    "gemma_v1": gemma_v1,
    "llava_v1": llava_v1,
}


def get_default_conv_template(template_name="gemma_v1") -> Conversation:
    """名前に基づいてデフォルトの会話テンプレートを取得
    
    Args:
        template_name: テンプレート名 ("gemma_v1" または "llava_v1")
        
    Returns:
        対応する会話テンプレートのコピー
    """
    if template_name not in conv_templates:
        raise ValueError(f"利用できるテンプレートは {conv_templates.keys()} ですが、リクエストされたのは {template_name} です。")
    return conv_templates[template_name].copy()


if __name__ == "__main__":
    conv = conv_templates["gemma_v1"].copy()
    conv.append_message(conv.roles[0], "Hello!")
    conv.append_message(conv.roles[1], "Hi!")
    conv.append_message(conv.roles[0], "How are you?")
    conv.append_message(conv.roles[1], None)
    print(conv.get_prompt())

"""文件内容：本文件包含 twin_align 包的基础信息。
主要职责：负责声明包版本，方便外部代码识别当前基线版本。
前置文件：twin_align/constants.py。
后置文件：被所有需要包级版本信息的外部代码调用。
"""

from .constants import MODEL_VERSION

__all__ = ["MODEL_VERSION"]

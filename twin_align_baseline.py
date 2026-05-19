"""文件内容：本文件包含命令行兼容入口。
主要职责：负责调用 twin_align.cli.main()，保持原运行命令不变。
前置文件：无。
后置文件：twin_align/cli.py。
"""

from twin_align.cli import main


if __name__ == "__main__":
    main()

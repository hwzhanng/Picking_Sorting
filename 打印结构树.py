#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os


def generate_tree(dir_path, indent=""):
    """
    递归地打印出目录结构树。

    参数:
    dir_path (str): 要遍历的目录路径。
    indent (str): 用于排版的缩进前缀。
    """

    # 尝试列出目录中的所有条目
    try:
        # 过滤掉隐藏文件/目录（以.开头）
        entries = [e for e in os.listdir(dir_path) if not e.startswith('.')]
        entries.sort()  # 排序以保证输出一致
    except PermissionError:
        print(f"{indent}└── [权限不足，无法访问]")
        return
    except FileNotFoundError:
        print(f"错误: 目录 '{dir_path}' 未找到。")
        return

    # 为条目生成连接线
    pointers = ['├── '] * (len(entries) - 1) + ['└── ']

    for pointer, entry in zip(pointers, entries):
        full_path = os.path.join(dir_path, entry)

        # 打印当前条目（文件或目录）
        print(f"{indent}{pointer}{entry}")

        # 如果是目录，则递归进入
        if os.path.isdir(full_path):
            # 计算下一层的缩进
            # 如果当前是最后一个条目 (└──)，则下一层缩进用空格
            # 否则 (├──)，下一层缩进用竖线 (│)
            next_indent = indent + ('    ' if pointer == '└── ' else '│   ')
            generate_tree(full_path, next_indent)


# --- 程序主入口 ---
if __name__ == "__main__":
    # 目标目录
    start_directory = "/home/cle/catch_it"

    if os.path.isdir(start_directory):
        # 首先打印根目录
        # os.path.basename(start_directory) 会获取 "catch_it"
        print(f"{os.path.basename(start_directory)}/")

        # 开始递归遍历
        generate_tree(start_directory)
    else:
        print(f"错误: 路径 '{start_directory}' 不是一个有效的目录。")
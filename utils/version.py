"""
统一版本识别工具

提供 git tag / commit / dirty 状态的运行时查询，
用于 train.py 的 run_name、run_config.json、启动 banner。
"""

import subprocess


def get_git_version() -> str:
    """优先使用 git describe --tags --dirty --always"""
    try:
        out = subprocess.check_output(
            ["git", "describe", "--tags", "--dirty", "--always"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return "unknown"


def get_git_commit() -> str:
    """git rev-parse --short HEAD"""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return "unknown"


def is_git_dirty() -> bool:
    """工作区是否有未提交的修改"""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return len(out) > 0
    except Exception:
        return False


def get_run_version_label() -> str:
    """
    生成运行时版本标签，用于 run_name 和 banner。

    优先返回最近的 git tag（如 v0.5.16），
    dirty 时追加 -dirty 后缀。
    """
    ver = get_git_version()
    if is_git_dirty() and not ver.endswith("-dirty"):
        ver += "-dirty"
    return ver

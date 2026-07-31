"""Shared label normalization rules for AIGC evaluation scripts."""

from __future__ import annotations

AI_LABELS: frozenset[str] = frozenset(
    {
        "1",
        "ai",
        "aigc",
        "generated",
        "machine",
        "ai生成",
        "ai生成文本",
    }
)

HUMAN_LABELS: frozenset[str] = frozenset(
    {
        "0",
        "human",
        "manual",
        "人工",
        "人类",
        "人类文本",
        "人工文本",
    }
)


def normalize_label(value: object) -> int:
    """Normalize a supported AI or human label to ``1`` or ``0``.

    Args:
        value: Raw label value read from a CSV row or supplied by a caller.

    Returns:
        ``1`` for AI-generated text and ``0`` for human-written text.

    Raises:
        ValueError: If the value is empty or not part of the supported set.
    """
    normalized = "".join(str(value).strip().lower().split())

    if normalized in AI_LABELS:
        return 1
    if normalized in HUMAN_LABELS:
        return 0

    raise ValueError(f"无法识别的标签：{value!r}")

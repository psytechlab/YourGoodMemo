"""
preprocessors.py
================
Утилиты и типы данных для pipeline обработки личных переписок.
"""

import re
from datetime import timedelta
from typing import Dict, NamedTuple, Optional

import pandas as pd


# ══════════════════════════════════════════════════════════════════════
# Структура данных одного чата
# ══════════════════════════════════════════════════════════════════════

class ChatData(NamedTuple):
    chat_id:     str            # имя файла или строковый ID
    chat_name:   str            # имя контакта (data["name"])
    chat_type:   str            # personal_chat
    target_id:   str            # sender_id таргет-пользователя
    target_name: str            # имя таргета
    df:          pd.DataFrame   # messages_df из TelegramChatParser


# ══════════════════════════════════════════════════════════════════════
# Утилиты
# ══════════════════════════════════════════════════════════════════════

def _window(df: pd.DataFrame, window_days: Optional[int]) -> pd.DataFrame:
    """Обрезает DataFrame по временному окну (последние window_days дней)"""
    if window_days is None:
        return df
    cutoff = df["date"].max() - timedelta(days=window_days)
    return df[df["date"] >= cutoff]


def _history_months(df: pd.DataFrame) -> float:
    """Длина истории чата в месяцах"""
    if df.empty:
        return 0.0
    delta = df["date"].max() - df["date"].min()
    return round(delta.days / 30.44, 1)


def _days_active(df: pd.DataFrame) -> int:
    """Количество уникальных дней с активностью"""
    return df["date"].dt.date.nunique()


def _voice_share(df: pd.DataFrame, target_id: str) -> float:
    """Доля голосовых сообщений среди всех сообщений таргета"""
    t = df[df["sender_id"] == target_id]
    if t.empty:
        return 0.0
    voice = (t["content_type"] == "voice_message").sum()
    return round(voice / len(t), 3)


def _initiation_balance(df: pd.DataFrame, target_id: str) -> float:
    """
    Доля дней, когда первое сообщение в чате написал таргет
    0.5 = симметрия, > 0.5 = таргет чаще инициирует
    """
    df = df[df["msg_type"] == "message"].copy()
    if df.empty:
        return 0.5
    df["date_only"] = df["date"].dt.date
    first_per_day = df.sort_values("date").groupby("date_only").first()
    target_initiations = (first_per_day["sender_id"] == target_id).sum()
    return round(target_initiations / len(first_per_day), 3)


_LEXICON_CACHE: Dict[str, "re.Pattern[str]"] = {}


def _make_pattern(word: str) -> "re.Pattern[str]":
    """
    Компилирует regex для слова/фразы из лексикона.
    Для слов на кириллице/латинице добавляет word-boundary \\b.
    Для символов (₽, !, …) — точное вхождение.
    """
    if word not in _LEXICON_CACHE:
        escaped = re.escape(word)
        prefix = r"\b" if re.match(r"^\w", word, re.UNICODE) else ""
        suffix = r"\b" if re.search(r"\w$", word, re.UNICODE) else ""
        _LEXICON_CACHE[word] = re.compile(
            prefix + escaped + suffix,
            re.IGNORECASE | re.UNICODE,
        )
    return _LEXICON_CACHE[word]

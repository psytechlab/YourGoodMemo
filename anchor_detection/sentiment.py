"""
sentiment.py
============
BERT-based sentiment / tone classifier for Russian text.

Model: blanchefort/rubert-tiny2-russian-sentiment
  - 3 classes: POSITIVE / NEUTRAL / NEGATIVE
  - Based on rubert-tiny2 (~80MB, fast on CPU)
  - Trained on Russian social media / chat data

Запускается на сырых сообщениях ДО лемматизации:
  - BERT использует свой токенизатор (WordPiece, subword)
  - Лемматизация убирает пунктуацию и строчные буквы, важные для тональности
  - Модели независимы: каждая получает оригинальный текст

add_sentiment_to_df() вызывается в orchestrator.py сразу после load_chats(),
результат (столбцы 'sentiment', 'sentiment_score') копируется в df_work
внутри Stage 1 через _lemmatize_df(df.copy()).
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd

_PIPELINE = None
_MODEL_NAME = "seara/rubert-tiny2-russian-sentiment"


def _get_pipeline():
    """Lazy-load — модель загружается один раз при первом вызове."""
    global _PIPELINE
    if _PIPELINE is None:
        from transformers import pipeline as hf_pipeline

        print(f"[sentiment] Загружаю {_MODEL_NAME}...")
        _PIPELINE = hf_pipeline(
            "text-classification",
            model=_MODEL_NAME,
            device=-1,   # CPU
            top_k=1,
        )
        print("[sentiment] Модель загружена.")
    return _PIPELINE


def _classify_batch(texts: List[str], batch_size: int = 64) -> List[Tuple[str, float]]:
    """
    Классифицирует список текстов батчами.
    Возвращает List[(label, confidence)], label в нижнем регистре:
    'positive' | 'neutral' | 'negative'.
    """
    pipe = _get_pipeline()
    # Обрезаем текст чтобы не выходить за 512 токенов; пустые тексты → одиночный пробел
    safe = [t[:512] if (t and t.strip()) else " " for t in texts]

    results: List[Tuple[str, float]] = []
    for i in range(0, len(safe), batch_size):
        batch = safe[i : i + batch_size]
        preds = pipe(batch, truncation=True, max_length=512)
        for pred in preds:
            top = pred[0] if isinstance(pred, list) else pred
            results.append((top["label"].lower(), float(top["score"])))
    return results


def add_sentiment_to_df(
    df: pd.DataFrame,
    text_col: str = "text",
    batch_size: int = 64,
) -> pd.DataFrame:
    """
    Добавляет столбцы 'sentiment' (str) и 'sentiment_score' (float) к DataFrame.

    Обрабатывает только строки, где msg_type == 'message' и has_text == True.
    Остальные строки получают значение по умолчанию: 'neutral', 0.5.

    Аргументы
    ---------
    df          : DataFrame из TelegramChatParser
    text_col    : имя столбца с текстом (обычно 'text')
    batch_size  : размер батча для BERT-инференса

    Возвращает копию df с новыми столбцами.
    """
    df = df.copy()
    df["sentiment"] = "neutral"
    df["sentiment_score"] = 0.5

    mask = (
        (df.get("msg_type", pd.Series("message", index=df.index)) == "message")
        & df.get("has_text", pd.Series(True, index=df.index))
        & df[text_col].notna()
        & (df[text_col].astype(str).str.strip() != "")
    )

    texts = df.loc[mask, text_col].astype(str).tolist()
    if not texts:
        return df

    print(f"[sentiment] Классифицирую тональность {len(texts)} сообщений...", flush=True)
    labels_scores = _classify_batch(texts, batch_size=batch_size)

    labels = [ls[0] for ls in labels_scores]
    scores = [ls[1] for ls in labels_scores]

    df.loc[mask, "sentiment"] = labels
    df.loc[mask, "sentiment_score"] = scores

    pos = labels.count("positive")
    neu = labels.count("neutral")
    neg = labels.count("negative")
    print(
        f"[sentiment] Готово: {pos} позитивных, {neu} нейтральных, {neg} негативных",
        flush=True,
    )
    return df

"""
stage1_labeler.py
=================
Ступень 1: скользящие окна + LLM-валидация кандидатов.

Алгоритм для каждого anchor:
  1. Regex pre-filter по лексиконам (мгновенно).
  2. LLM-валидация с контекстом ±2 сообщения.
  3. Стоп, если >= min_candidates_threshold И >= min_time_windows уникальных недель.
     Иначе — следующее окно (90 дней назад).
  4. Fallback: LLM full scan (субвыборка scan_max_messages сообщений).
"""

import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def _fmt_t(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def _temporal_diversity(candidates: List[Dict]) -> int:
    """Считает количество уникальных ISO-недель среди кандидатов."""
    weeks: set = set()
    for c in candidates:
        ts = c.get("ts", "")
        try:
            dt = datetime.fromisoformat(ts[:16])
            weeks.add(dt.isocalendar()[:2])  # (year, week)
        except Exception:
            pass
    return len(weeks)


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from multi_pipeline.config import PipelineConfig
from anchor_detection.llm_client import OllamaClient
from anchor_detection.lemmatizer import lemmatize_lexicon, lemmatize_text
from anchor_detection.lexicons import (
    EMOTIONAL_DISCLOSURE, EMPATHY_RESPONSE,
    WE_PRONOUNS,
    HELP_REQUEST_MARKERS,
    DEBT_ANXIETY, FINANCIAL_POSITIVE, MONEY_KEYWORDS,
)
from anchor_detection.preprocessors import _make_pattern
from anchor_detection.lexicons_p2 import (
    PRAISE_WORDS, FUTURE_PLAN_WORDS, POSITIVE_FUTURE_MARKERS,
    HOBBY_ACTIVITY_WORDS, HOBBY_REGULARITY_MARKERS,
    STRESSOR_WORDS,
)

# ══════════════════════════════════════════════════════════════════════
# Гибридный скрининг: константы и промты
# ══════════════════════════════════════════════════════════════════════

# Сколько сообщений брать в стратифицированную выборку (fallback full_scan=False)
LLM_SCREEN_SAMPLE = 45

# Описания якорей для LLM-скрининга — намеренно широкие, чтобы поймать косвенные сигналы
ANCHOR_SCREEN_DESCRIPTIONS: Dict[str, str] = {
    "S1": (
        "человек делится своими чувствами, переживаниями, тревогами — "
        "даже без «эмоциональных» слов; или собеседник реагирует с пониманием "
        "и вниманием на то, что говорит другой"
    ),
    "S2": (
        "принадлежность к группе, компании, команде, тусовке — "
        "может быть без слова «мы»: упоминание имён группы, регулярных встреч, "
        "общих событий, «как обычно», названий их компании"
    ),
    "S5": (
        "кто-то обращается к пользователю за помощью, советом, объяснением "
        "или просит что-то сделать — не риторически, а реально"
    ),
    "D2": (
        "упоминание денег, зарплаты, трат, покупок или финансовых планов "
        "без явной тревоги — как часть обычной жизни"
    ),
    "S3": (
        "собеседник хвалит, благодарит, отмечает заслуги или компетентность "
        "пользователя — прямо или косвенно (пересылает, упоминает третьим)"
    ),
    "C1": (
        "конкретные планы на будущее с временным горизонтом: поездка, "
        "курс, событие, запись — не просто «хочу», а уже что-то предпринято"
    ),
    "D3": (
        "регулярное хобби или физическая активность: спорт, игры, "
        "творчество, учёба — с признаками прогресса или регулярности"
    ),
    "E2": (
        "негативное событие в жизни пользователя: потеря, расставание, "
        "провал, конфликт — может быть очень косвенным («ну вот и всё», "
        "«не получилось», «сложный период»)"
    ),
}

_SCREEN_SYSTEM = """Ты — быстрый скрининг потенциальных сигналов в переписке.

Тебе переданы сообщения и описание паттерна. Найди ID сообщений,
которые ПОТЕНЦИАЛЬНО содержат этот сигнал — даже очень косвенно.

ПРАВИЛА:
1. Лучше включить лишнее, чем пропустить важное.
2. Пропускай только очевидно нерелевантное (новости без контекста, технические ссылки,
   стикеры, короткие «ок»/«да»/«нет» без смысла).
3. НЕ нужно подтверждать сигнал — только отметить потенциальных кандидатов.
4. Ответ строго в JSON на русском языке.

Формат: {"relevant_ids": ["msg_123", "msg_456"]}
Если ничего нет: {"relevant_ids": []}"""


def _llm_full_scan(
    df: pd.DataFrame,
    anchor_code: str,
    client: OllamaClient,
    target_id: str,
    role_filter: Optional[str] = None,
    batch_size: int = 12,
    overlap: int = 0,
    timeout_per_msg: int = 15,
    max_messages: int = 0,
) -> List[Dict]:
    """
    Полный LLM-скан всех сообщений батчами.

    batch_size      — сообщений за один вызов (меньше = надёжнее, меньше шанс таймаута)
    overlap         — сколько сообщений из конца предыдущего батча включать в начало следующего
                      (сохраняет контекст диалога на границе батчей)
    timeout_per_msg — секунд на одно сообщение; итоговый таймаут = batch_size × timeout_per_msg
    max_messages    — лимит сообщений; если > 0 и df больше — берём стратифицированную выборку
    """
    description = ANCHOR_SCREEN_DESCRIPTIONS.get(anchor_code, anchor_code)

    df_work = df[df["has_text"] & (df["msg_type"] == "message")].copy()
    if role_filter == "target":
        df_work = df_work[df_work["sender_id"] == target_id]
    elif role_filter == "contact":
        df_work = df_work[df_work["sender_id"] != target_id]

    if df_work.empty:
        return []

    df_work = df_work.sort_values("date").reset_index(drop=True)
    if max_messages > 0 and len(df_work) > max_messages:
        n_orig = len(df_work)
        step = max(1, n_orig // max_messages)
        df_work = df_work.iloc[::step].head(max_messages).reset_index(drop=True)
        print(f"  [Stage 1 / full-scan / {anchor_code}] субвыборка: {len(df_work)}/{n_orig} сообщений")
    all_candidates: List[Dict] = []
    seen_ids: set = set()

    step = max(1, batch_size - overlap)
    call_timeout = batch_size * timeout_per_msg

    for batch_start in range(0, len(df_work), step):
        batch = df_work.iloc[batch_start : batch_start + batch_size]
        lines = [f"Паттерн для поиска: {description}\n\nСообщения:"]
        for _, row in batch.iterrows():
            lines.append(
                f"[msg_{row['msg_id']}] {str(row['date'])[:10]} {row['sender']}: "
                f"{str(row['text'])[:150]!r}"
            )
        try:
            result = client.chat(
                system=_SCREEN_SYSTEM,
                user="\n".join(lines),
                temperature=0.0,
                json_mode=True,
                timeout=call_timeout,
            )
            batch_idx = {row["msg_id"]: row for _, row in batch.iterrows()}
            for mid_str in result.get("relevant_ids", []):
                try:
                    mid = int(str(mid_str).replace("msg_", ""))
                except ValueError:
                    continue
                if mid in batch_idx and mid not in seen_ids:
                    seen_ids.add(mid)
                    row  = batch_idx[mid]
                    role = _infer_role(anchor_code, row.get("sender_id", ""), target_id)
                    all_candidates.append(_row_to_dict(row, role))
        except Exception as exc:
            print(f"  [Stage 1 / full-scan / {anchor_code}] батч {batch_start}: {exc}")

    return all_candidates


def _stratified_sample(
    df: pd.DataFrame,
    n: int = LLM_SCREEN_SAMPLE,
) -> pd.DataFrame:
    """
    Стратифицированная выборка: делит df на n временных корзин
    и берёт по одному сообщению из каждой.
    Это обеспечивает покрытие всего временного диапазона.
    """
    if df.empty or len(df) <= n:
        return df
    df_sorted = df.sort_values("date").reset_index(drop=True)
    bucket_size = len(df_sorted) // n
    indices = [i * bucket_size for i in range(n)]
    return df_sorted.iloc[indices]


def _llm_screen(
    df: pd.DataFrame,
    anchor_code: str,
    client: OllamaClient,
    target_id: str,
    role_filter: Optional[str] = None,  # "target" | "contact" | None (все)
) -> List[Dict]:
    """
    LLM-скрининг стратифицированной выборки.
    Возвращает список dict кандидатов (тот же формат, что _row_to_dict).
    """
    description = ANCHOR_SCREEN_DESCRIPTIONS.get(anchor_code, anchor_code)

    # Фильтр по роли отправителя
    df_work = df[df["has_text"] & (df["msg_type"] == "message")].copy()
    if role_filter == "target":
        df_work = df_work[df_work["sender_id"] == target_id]
    elif role_filter == "contact":
        df_work = df_work[df_work["sender_id"] != target_id]

    sample = _stratified_sample(df_work, LLM_SCREEN_SAMPLE)
    if sample.empty:
        return []

    # Форматируем сообщения для LLM
    lines = [f"Паттерн для поиска: {description}\n\nСообщения:"]
    for _, row in sample.iterrows():
        lines.append(
            f"[msg_{row['msg_id']}] {str(row['date'])[:10]} {row['sender']}: "
            f"{str(row['text'])[:150]!r}"
        )

    user_prompt = "\n".join(lines)

    try:
        result = client.chat(
            system    = _SCREEN_SYSTEM,
            user      = user_prompt,
            temperature = 0.0,
            json_mode = True,
        )
        relevant_ids = result.get("relevant_ids", [])

        # Строим словарь sample по msg_id для быстрого поиска
        sample_idx = {row["msg_id"]: row for _, row in sample.iterrows()}

        candidates = []
        for mid_str in relevant_ids:
            try:
                mid = int(str(mid_str).replace("msg_", ""))
            except ValueError:
                continue
            if mid in sample_idx:
                row  = sample_idx[mid]
                role = _infer_role(anchor_code, row.get("sender_id", ""), target_id)
                candidates.append(_row_to_dict(row, role))

        return candidates
    except Exception as exc:
        print(f"  [Stage 1 / LLM-screen / {anchor_code}] Ошибка: {exc}")
        return []


# ══════════════════════════════════════════════════════════════════════
# Промт валидатора кандидатов
# ══════════════════════════════════════════════════════════════════════

_VALIDATOR_SYSTEM = """Ты — фильтр кандидатов на точку опоры.

Тебе переданы сообщения, отобранные как потенциально релевантные
для точки опоры «{anchor_name}».

ОСТАВЛЯЙ сообщения, где:
  - Есть личный смысл: конкретное переживание, действие, признание, план
  - Сигнал прямой и искренний (не ирония)
  - Содержание напрямую связано с темой точки опоры

ОТСЕИВАЙ сообщения, где:
  - Мем, интернет-шаблон, шутка, очевидная ирония
  - Слово встретилось в совершенно другом контексте (случайное совпадение)
  - Риторический вопрос без ожидания ответа
  - Чисто информационное или техническое сообщение без личного смысла

ВАЖНО: если сообщение ДЕЙСТВИТЕЛЬНО содержит сигнал — оставляй его.
Не отклоняй релевантные сообщения только потому, что они «слишком очевидно»
относятся к теме — именно такие сообщения нам и нужны.

ПРАВИЛА:
1. Оставляй только сообщения с реальным личным смыслом.
2. Все текстовые поля (причины отклонения) пиши ТОЛЬКО на русском языке.
3. Используй только message_ids из переданного списка.

Формат ответа:
{{
  "relevant": ["msg_123", "msg_456"],
  "rejected": {{"msg_789": "причина отклонения на русском языке"}}
}}"""


# ══════════════════════════════════════════════════════════════════════
# Regex pre-filter: находит кандидатов для каждого anchor
# ══════════════════════════════════════════════════════════════════════

def _contains_any(text: str, lexicon) -> bool:
    t = text.lower()
    return any(_make_pattern(w).search(t) for w in lexicon)


def _build_lexicon_pattern(lexicon) -> "re.Pattern":
    """Строит объединённый regex из лексикона для векторизованной фильтрации DataFrame."""
    parts = []
    for w in lexicon:
        escaped = re.escape(w)
        prefix  = r"\b" if re.match(r"^\w", w, re.UNICODE) else ""
        suffix  = r"\b" if re.search(r"\w$", w, re.UNICODE) else ""
        parts.append(prefix + escaped + suffix)
    combined = "|".join(f"(?:{p})" for p in parts)
    return re.compile(combined, re.IGNORECASE | re.UNICODE)


def _lemmatize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет колонку text_lemma — лемматизированный текст каждого сообщения."""
    df = df.copy()
    df["text_lemma"] = df["text"].fillna("").apply(lemmatize_text)
    return df


def _vec_filter(df: pd.DataFrame, *lexicons) -> pd.DataFrame:
    """
    Векторизованный фильтр: возвращает строки df, где текст содержит
    хотя бы одно слово из любого переданного лексикона.

    Если в df есть колонка text_lemma (добавляется через _lemmatize_df),
    дополнительно проверяет лемматизированный текст против лемматизированных
    версий лексиконов. Это позволяет находить словоформы, не прописанные явно:
    «помогала», «помогаешь» → лемма «помочь» совпадёт с леммой слова «помочь».
    """
    if df.empty:
        return df

    texts = df["text"].str.lower().fillna("")
    has_lemma = "text_lemma" in df.columns
    lemma_col = df["text_lemma"].fillna("") if has_lemma else None

    mask = pd.Series(False, index=df.index)
    for lex in lexicons:
        if not lex:
            continue
        pat = _build_lexicon_pattern(lex)
        mask |= texts.str.contains(pat, regex=True, na=False)
        if has_lemma:
            lem_lex = lemmatize_lexicon(lex)
            if lem_lex:
                lem_pat = _build_lexicon_pattern(lem_lex)
                mask |= lemma_col.str.contains(lem_pat, regex=True, na=False)
    return df[mask]


def _prefilter_candidates(
    df: pd.DataFrame,
    anchor_code: str,
    target_id: str,
    max_candidates: int = 80,
) -> List[Dict]:
    """
    Векторизованный regex-поиск кандидатов для конкретного anchor.
    Использует _vec_filter вместо iterrows() — быстрее в 10-50x на больших df.
    """
    df_text = df[(df["msg_type"] == "message") & df["has_text"]].copy()
    t_df = df_text[df_text["sender_id"] == target_id]
    c_df = df_text[df_text["sender_id"] != target_id]

    if anchor_code == "S1":
        emo  = _vec_filter(t_df, EMOTIONAL_DISCLOSURE)
        emp  = _vec_filter(c_df, EMPATHY_RESPONSE)
        rows = (
            [(r, "target_emotional") for _, r in emo.iterrows()] +
            [(r, "contact_empathy")  for _, r in emp.iterrows()]
        )

    elif anchor_code == "S2":
        we_all = list(WE_PRONOUNS)
        hits = _vec_filter(t_df, frozenset(we_all))
        rows = [(r, "we_pronoun") for _, r in hits.iterrows()]

    elif anchor_code == "S5":
        hits = _vec_filter(c_df, HELP_REQUEST_MARKERS)
        rows = [(r, "help_request") for _, r in hits.iterrows()]

    elif anchor_code == "D2":
        hits = _vec_filter(t_df, MONEY_KEYWORDS, DEBT_ANXIETY, FINANCIAL_POSITIVE)
        rows = [(r, "financial") for _, r in hits.iterrows()]

    else:
        rows = []

    candidates = [_row_to_dict(r, role) for r, role in rows]
    return candidates[-max_candidates:]


def _infer_role(anchor_code: str, sender_id: str, target_id: str) -> str:
    """Возвращает семантическую роль кандидата на основе якоря и отправителя."""
    if anchor_code == "S1":
        return "target_emotional" if str(sender_id) == str(target_id) else "contact_empathy"
    roles = {
        "S2": "we_pronoun",
        "S5": "help_request",
        "D2": "financial",
        "S3": "praise_received",
        "C1": "future_plan",
        "D3": "hobby",
        "E2": "stressor_llm",
    }
    return roles.get(anchor_code, f"llm_{anchor_code.lower()}")


def _row_to_dict(row, role: str) -> Dict:
    return {
        "msg_id":  f"msg_{row['msg_id']}",
        "ts":      str(row["date"])[:16],
        "sender":  row.get("sender", ""),
        "text":    str(row.get("text", ""))[:250],
        "role":    role,
    }


def _ctx_msg(row) -> Dict:
    return {
        "msg_id": f"msg_{row['msg_id']}",
        "ts":     str(row["date"])[:16],
        "sender": row.get("sender", ""),
        "text":   str(row.get("text", ""))[:120],
    }


def _enrich_with_context(
    candidates: List[Dict],
    df: pd.DataFrame,
    time_window_minutes: int = 5,
    n_positional: int = 3,
    min_context_msgs: int = 2,
) -> List[Dict]:
    """
    Добавляет context_before/context_after к каждому кандидату.

    Стратегия:
    1. Берёт сообщения в временном окне ±time_window_minutes (захватывает диалог целиком).
    2. Если контекстных сообщений < min_context_msgs → fallback на позиционный ±n_positional
       (случай когда ответили позже или пауза в диалоге).
    """
    if not candidates:
        return candidates

    df_msgs = df[
        (df["msg_type"] == "message") & df["has_text"]
    ].sort_values("date").reset_index(drop=True)

    id_to_pos: Dict[int, int] = {
        int(row["msg_id"]): idx for idx, row in df_msgs.iterrows()
    }
    window_delta = timedelta(minutes=time_window_minutes)

    enriched = []
    for cand in candidates:
        try:
            mid = int(cand["msg_id"].replace("msg_", ""))
        except (ValueError, AttributeError):
            enriched.append(cand)
            continue

        pos = id_to_pos.get(mid)
        if pos is None:
            enriched.append(cand)
            continue

        cand_ts = df_msgs.iloc[pos]["date"]

        # Стратегия 1: временное окно
        time_before = df_msgs[
            (df_msgs["date"] >= cand_ts - window_delta) &
            (df_msgs["date"] <  cand_ts)
        ]
        time_after = df_msgs[
            (df_msgs["date"] >  cand_ts) &
            (df_msgs["date"] <= cand_ts + window_delta)
        ]

        # Fallback: позиционный, если в окне слишком мало соседей
        if len(time_before) + len(time_after) < min_context_msgs:
            before = df_msgs.iloc[max(0, pos - n_positional): pos]
            after  = df_msgs.iloc[pos + 1: pos + 1 + n_positional]
        else:
            before = time_before
            after  = time_after

        enriched.append({
            **cand,
            "context_before": [_ctx_msg(r) for _, r in before.iterrows()],
            "context_after":  [_ctx_msg(r) for _, r in after.iterrows()],
        })

    return enriched


# ══════════════════════════════════════════════════════════════════════
# LLM-валидация кандидатов
# ══════════════════════════════════════════════════════════════════════

_ANCHOR_NAMES = {
    # P1
    "S1": "Эмоциональная поддержка и близкие связи",
    "S2": "Принадлежность к сообществу",
    "S5": "«Я нужен» — просьбы о помощи к пользователю",
    "D2": "Финансовая безопасность",
    # P2
    "S3": "Признание и самооценочная поддержка",
    "C1": "Надежда и оптимизм — конкретные планы на будущее",
    "D3": "Хобби, увлечения и поток",
    "E2": "Копинг-репертуар — преодоление трудностей",
}


def _fmt_candidate_with_ctx(c: Dict, df_msgs: "Optional[pd.DataFrame]",
                            id_to_pos: "Optional[Dict]") -> str:
    """Форматирует кандидата с ±2 соседними сообщениями для контекста."""
    base = f"[{c['msg_id']}] ({c['role']}) {c['sender']}: {c['text']!r}"
    if df_msgs is None or id_to_pos is None:
        return base
    try:
        mid = int(str(c["msg_id"]).replace("msg_", ""))
        pos = id_to_pos.get(mid)
        if pos is None:
            return base
        lines = []
        for i in range(max(0, pos - 2), pos):
            r = df_msgs.iloc[i]
            lines.append(f"  ← {r['sender']}: {str(r['text'])[:100]!r}")
        lines.append(base)
        for i in range(pos + 1, min(len(df_msgs), pos + 3)):
            r = df_msgs.iloc[i]
            lines.append(f"  → {r['sender']}: {str(r['text'])[:100]!r}")
        return "\n".join(lines)
    except Exception:
        return base


_VALIDATE_BATCH = 25  # максимум кандидатов за один вызов валидатора


def _validate_candidates(
    candidates: List[Dict],
    anchor_code: str,
    client: OllamaClient,
    df: Optional[pd.DataFrame] = None,
) -> List[Dict]:
    """
    Отправляет кандидатов в LLM для фильтрации мемов и ложных срабатываний.
    df — если передан, валидатор видит ±2 соседних сообщения для контекста.
    Если кандидатов > _VALIDATE_BATCH — разбивает на батчи и мержит результаты.
    Возвращает только подтверждённые релевантные сообщения.
    """
    if not candidates:
        return []

    # Разбивка на батчи если кандидатов слишком много для контекстного окна
    if len(candidates) > _VALIDATE_BATCH:
        all_kept: List[Dict] = []
        for i in range(0, len(candidates), _VALIDATE_BATCH):
            batch = candidates[i : i + _VALIDATE_BATCH]
            all_kept.extend(_validate_candidates(batch, anchor_code, client, df=df))
        return all_kept

    anchor_name = _ANCHOR_NAMES.get(anchor_code, anchor_code)
    system = _VALIDATOR_SYSTEM.format(anchor_name=anchor_name)

    # Строим индекс позиций для контекстного показа
    df_msgs: Optional[pd.DataFrame] = None
    id_to_pos: Optional[Dict] = None
    if df is not None:
        df_msgs = df[(df["msg_type"] == "message") & df["has_text"]].sort_values("date").reset_index(drop=True)
        id_to_pos = {int(row["msg_id"]): idx for idx, row in df_msgs.iterrows()}

    user_prompt = (
        f"Кандидаты для точки опоры «{anchor_name}» ({len(candidates)} шт.):\n\n"
        + "\n\n".join(
            _fmt_candidate_with_ctx(c, df_msgs, id_to_pos)
            for c in candidates
        )
    )

    try:
        call_timeout = max(180, len(candidates) * 6)
        result = client.chat(
            system      = system,
            user        = user_prompt,
            temperature = 0.0,
            json_mode   = True,
            timeout     = call_timeout,
        )
        relevant_ids = set(result.get("relevant", []))
        rejected     = result.get("rejected", {})

        kept = [c for c in candidates if c["msg_id"] in relevant_ids]
        print(f"    Кандидатов: {len(candidates)} → "
              f"принято: {len(kept)}, отклонено: {len(rejected)}")

        for mid, reason in list(rejected.items())[:5]:
            print(f"    ✗ {mid}: {reason}")

        return kept

    except Exception as exc:
        print(f"  [Stage 1 / {anchor_code}] LLM-валидация упала: {exc}")
        return candidates


# ══════════════════════════════════════════════════════════════════════
# P2 предфильтры
# ══════════════════════════════════════════════════════════════════════

def _build_e2_episodes_from_screen(
    screen_candidates: List[Dict],
    df: pd.DataFrame,
    target_id: str,
    max_episodes: int = 5,
) -> List[Dict]:
    """
    Строит E2-эпизоды из кандидатов, найденных LLM-скринингом.
    Для каждого потенциального стрессора берёт 3 недели последующей переписки.
    """
    if not screen_candidates:
        return []

    df_msgs = df[(df["msg_type"] == "message") & df["has_text"]].sort_values("date")
    msg_idx = {row["msg_id"]: row for _, row in df_msgs.iterrows()}
    episodes = []

    for cand in screen_candidates[:max_episodes]:
        try:
            mid = int(cand["msg_id"].replace("msg_", ""))
        except ValueError:
            continue
        row = msg_idx.get(mid)
        if row is None:
            continue
        stressor_ts = row["date"]
        aftermath_end = stressor_ts + timedelta(days=21)
        aftermath = df_msgs[
            (df_msgs["date"] > stressor_ts) &
            (df_msgs["date"] <= aftermath_end)
        ].head(30)
        episodes.append({
            "stressor_msg": _row_to_dict(row, "stressor_llm"),
            "aftermath_msgs": [_row_to_dict(r, "aftermath") for _, r in aftermath.iterrows()],
        })
    return episodes


def _prefilter_p2_candidates(
    df: pd.DataFrame, anchor_code: str, target_id: str, max_candidates: int = 80
) -> List[Dict]:
    """Векторизованный regex предфильтр для S3, C1, D3."""
    df_text    = df[(df["msg_type"] == "message") & df["has_text"]].copy()
    target_df  = df_text[df_text["sender_id"] == target_id]
    contact_df = df_text[df_text["sender_id"] != target_id]

    if anchor_code == "S3":
        hits = _vec_filter(contact_df, PRAISE_WORDS)
        rows = [(r, "praise_received") for _, r in hits.iterrows()]

    elif anchor_code == "C1":
        # FUTURE_PLAN_WORDS через _vec_filter + фразовые маркеры отдельно
        hits_words   = _vec_filter(target_df, FUTURE_PLAN_WORDS)
        marker_pat   = re.compile(
            "|".join(re.escape(p) for p in POSITIVE_FUTURE_MARKERS),
            re.IGNORECASE | re.UNICODE,
        )
        hits_markers = target_df[
            target_df["text"].str.lower().str.contains(marker_pat, regex=True, na=False)
        ]
        combined = pd.concat([hits_words, hits_markers]).drop_duplicates(subset="msg_id")
        rows = [(r, "future_plan") for _, r in combined.iterrows()]

    elif anchor_code == "D3":
        hits_words = _vec_filter(target_df, HOBBY_ACTIVITY_WORDS)
        reg_pat    = re.compile(
            "|".join(re.escape(p) for p in HOBBY_REGULARITY_MARKERS),
            re.IGNORECASE | re.UNICODE,
        )
        hits_reg   = target_df[
            target_df["text"].str.lower().str.contains(reg_pat, regex=True, na=False)
        ]
        combined = pd.concat([hits_words, hits_reg]).drop_duplicates(subset="msg_id")
        rows = [(r, "hobby") for _, r in combined.iterrows()]

    else:
        rows = []

    candidates = [_row_to_dict(r, role) for r, role in rows]
    return candidates[-max_candidates:]


def _prefilter_e2_episodes(
    df: pd.DataFrame, target_id: str, max_episodes: int = 5
) -> List[Dict]:
    """
    Ищет ЭПИЗОДЫ: стрессор + поведение таргета в следующие 3 недели.
    Использует всю историю (не только окно) для улавливания паттерна.
    """
    df_msgs = df[
        (df["msg_type"] == "message") & df["has_text"]
    ].sort_values("date").copy()

    target_df = df_msgs[df_msgs["sender_id"] == target_id]
    episodes  = []

    for _, row in target_df.iterrows():
        if not _contains_any(row["text"], STRESSOR_WORDS):
            continue

        stressor_ts = row["date"]
        aftermath_end = stressor_ts + timedelta(days=21)

        # Все сообщения в период +21 день
        aftermath = df_msgs[
            (df_msgs["date"] > stressor_ts) &
            (df_msgs["date"] <= aftermath_end)
        ].head(30)

        episodes.append({
            "stressor_msg": _row_to_dict(row, "stressor"),
            "aftermath_msgs": [_row_to_dict(r, "aftermath") for _, r in aftermath.iterrows()],
        })

        if len(episodes) >= max_episodes:
            break

    return episodes


# ══════════════════════════════════════════════════════════════════════
# Основной класс
# ══════════════════════════════════════════════════════════════════════

class MessageLabeler:
    """
    Anchor-centric Stage 1:
      1. Regex pre-filter → кандидаты для каждого anchor (~50-80 шт.)
      2. LLM валидирует кандидатов (5 вызовов вместо 91)

    Возвращает: Dict[anchor_code → List[Dict]] — готовые сообщения для Stage 2.
    """

    def __init__(self, config: PipelineConfig, verbose: bool = True):
        self.config  = config
        self.verbose = verbose
        self.client  = OllamaClient(
            model    = config.labeler_model,
            base_url = config.ollama_url,
            num_ctx  = config.num_ctx_labeler,
            timeout  = 180,
        )

    def label_for_anchors(
        self,
        df: pd.DataFrame,
        target_id: str,
        window_days: Optional[int] = None,
        anchors: Optional[List[str]] = None,
        full_scan: bool = True,
    ) -> Dict[str, List[Dict]]:
        """
        Для каждого anchor возвращает список релевантных сообщений.

        Алгоритм (скользящие окна, приоритет у свежих событий):
          1. Лемматизация всего df (один раз).
          2. W1 (0–window_days) → regex → LLM-валидация.
             Если >= min_candidates_threshold — стоп.
             Иначе W2 (window_days–2*window_days) → ... до конца истории.
          3. Если после всех окон < порога — LLM full scan (fallback).
        """
        anchors   = anchors or ["S1", "S2", "S5", "D2"]
        threshold = self.config.min_candidates_threshold

        if self.verbose:
            print("  [Stage 1] Лемматизация сообщений...", flush=True)
        df_work  = _lemmatize_df(df.copy())
        role_map = {"S1": None, "S2": "target", "S5": "contact", "D2": "target"}
        result: Dict[str, List[Dict]] = {}

        if df_work.empty:
            return {code: [] for code in anchors}

        max_date = df_work["date"].max()
        min_date = df_work["date"].min()

        for code in anchors:
            t0        = time.time()
            validated: List[Dict] = []
            seen_ids:  set        = set()

            if window_days is not None:
                chunk = 0
                while (len(validated) < threshold or
                       _temporal_diversity(validated) < self.config.min_time_windows):
                    end    = max_date - timedelta(days=chunk * window_days)
                    start  = max_date - timedelta(days=(chunk + 1) * window_days)
                    win_df = df_work[(df_work["date"] > start) & (df_work["date"] <= end)]

                    if not win_df.empty:
                        cands = _prefilter_candidates(win_df, code, target_id)
                        new_c = [c for c in cands if c["msg_id"] not in seen_ids]
                        if self.verbose:
                            print(f"  [Stage 1 / {code}] W{chunk+1}: {len(new_c)} regex",
                                  end="", flush=True)
                        if new_c:
                            if self.verbose:
                                print(" → валидация...", end=" ", flush=True)
                            new_v = _validate_candidates(new_c, code, self.client, df=df_work)
                            for c in new_v:
                                if c["msg_id"] not in seen_ids:
                                    seen_ids.add(c["msg_id"])
                                    validated.append(c)
                            if self.verbose:
                                div = _temporal_diversity(validated)
                                done = len(validated) >= threshold and div >= self.config.min_time_windows
                                mark = f" ✓ ({div}w)" if done else f" ({div}w)"
                                print(f"{len(new_v)} → итого {len(validated)}{mark}")
                        else:
                            if self.verbose:
                                print()

                    chunk += 1
                    if start <= min_date:
                        break
            else:
                cands = _prefilter_candidates(df_work, code, target_id)
                if cands:
                    validated = _validate_candidates(cands, code, self.client, df=df_work)
                    seen_ids  = {c["msg_id"] for c in validated}

            # Fallback: LLM full scan если всё ещё < порога
            if len(validated) < threshold:
                role = role_map.get(code)
                mode = "полный скан" if full_scan else "выборка"
                if self.verbose:
                    print(f"  [Stage 1 / {code}] {len(validated)}/{threshold} → LLM ({mode})...",
                          end=" ", flush=True)
                if full_scan:
                    screen = _llm_full_scan(
                        df_work, code, self.client, target_id, role,
                        batch_size      = self.config.scan_batch_size,
                        overlap         = self.config.scan_batch_overlap,
                        timeout_per_msg = self.config.scan_timeout_per_msg,
                        max_messages    = self.config.scan_max_messages,
                    )
                else:
                    screen = _llm_screen(df_work, code, self.client, target_id, role)
                new_s    = [c for c in screen if c["msg_id"] not in seen_ids]
                screen_v = _validate_candidates(new_s, code, self.client, df=df_work) if new_s else []
                for c in screen_v:
                    if c["msg_id"] not in seen_ids:
                        seen_ids.add(c["msg_id"])
                        validated.append(c)
                if self.verbose:
                    print(f"+{len(screen_v)} → итого {len(validated)}")

            result[code] = _enrich_with_context(
                validated, df_work,
                time_window_minutes = self.config.context_window_minutes,
                n_positional        = self.config.context_n_positional,
                min_context_msgs    = self.config.context_min_msgs,
            )
            if self.verbose:
                print(f"  [Stage 1 / {code}] итого: {len(result[code])} кандидатов за {_fmt_t(time.time() - t0)}")

        return result

    def label_for_p2_anchors(
        self,
        df:          pd.DataFrame,
        target_id:   str,
        window_days: Optional[int] = None,
        anchors:     Optional[List[str]] = None,
        full_scan:   bool = True,
    ) -> Dict[str, List[Dict]]:
        """
        Предфильтр для P2 якорей: S3, C1, D3, E2.
        S3/C1/D3: скользящие окна + валидация (та же логика, что label_for_anchors).
        E2: вся история — ищет эпизоды стрессор+копинг.
        """
        anchors   = anchors or ["S3", "C1", "D3", "E2"]
        threshold = self.config.min_candidates_threshold

        if self.verbose:
            print("  [Stage 1 / P2] Лемматизация сообщений...", flush=True)
        df_work  = _lemmatize_df(df.copy())
        role_map = {"S3": "contact", "C1": "target", "D3": "target", "E2": "target"}
        result: Dict[str, List[Dict]] = {}

        if df_work.empty:
            return {code: [] for code in anchors}

        max_date = df_work["date"].max()
        min_date = df_work["date"].min()

        for code in anchors:
            t0 = time.time()
            if self.verbose:
                print(f"  [Stage 1 / {code}] Regex...", end=" ", flush=True)

            if code == "E2":
                # E2: ищем эпизоды по всей истории (не по окну)
                items = _prefilter_e2_episodes(df, target_id)
                if len(items) == 0:
                    if self.verbose:
                        print("0 эпизодов → LLM-скрининг стрессоров...", end=" ", flush=True)
                    screen = _llm_full_scan(
                        df, "E2", self.client, target_id, "target",
                        batch_size      = self.config.scan_batch_size,
                        overlap         = self.config.scan_batch_overlap,
                        timeout_per_msg = self.config.scan_timeout_per_msg,
                        max_messages    = self.config.scan_max_messages,
                    ) if full_scan else _llm_screen(df, "E2", self.client, target_id, "target")
                    items = _build_e2_episodes_from_screen(screen, df, target_id)
                if self.verbose:
                    print(f"{len(items)} эпизодов за {_fmt_t(time.time() - t0)}")
                result[code] = items
                continue

            validated: List[Dict] = []
            seen_ids:  set        = set()

            if window_days is not None:
                chunk = 0
                while (len(validated) < threshold or
                       _temporal_diversity(validated) < self.config.min_time_windows):
                    end    = max_date - timedelta(days=chunk * window_days)
                    start  = max_date - timedelta(days=(chunk + 1) * window_days)
                    win_df = df_work[(df_work["date"] > start) & (df_work["date"] <= end)]

                    if not win_df.empty:
                        cands = _prefilter_p2_candidates(win_df, code, target_id)
                        new_c = [c for c in cands if c["msg_id"] not in seen_ids]
                        if self.verbose:
                            print(f"  [Stage 1 / {code}] W{chunk+1}: {len(new_c)} regex",
                                  end="", flush=True)
                        if new_c:
                            if self.verbose:
                                print(" → валидация...", end=" ", flush=True)
                            new_v = _validate_candidates(new_c, code, self.client, df=df_work)
                            for c in new_v:
                                if c["msg_id"] not in seen_ids:
                                    seen_ids.add(c["msg_id"])
                                    validated.append(c)
                            if self.verbose:
                                div = _temporal_diversity(validated)
                                done = len(validated) >= threshold and div >= self.config.min_time_windows
                                mark = f" ✓ ({div}w)" if done else f" ({div}w)"
                                print(f"{len(new_v)} → итого {len(validated)}{mark}")
                        else:
                            if self.verbose:
                                print()

                    chunk += 1
                    if start <= min_date:
                        break
            else:
                cands = _prefilter_p2_candidates(df_work, code, target_id)
                if cands:
                    validated = _validate_candidates(cands, code, self.client, df=df_work)
                    seen_ids  = {c["msg_id"] for c in validated}

            # Fallback: LLM full scan если всё ещё < порога
            if len(validated) < threshold:
                role = role_map.get(code)
                mode = "полный скан" if full_scan else "выборка"
                if self.verbose:
                    print(f"  [Stage 1 / {code}] {len(validated)}/{threshold} → LLM ({mode})...",
                          end=" ", flush=True)
                if full_scan:
                    screen = _llm_full_scan(
                        df_work, code, self.client, target_id, role,
                        batch_size      = self.config.scan_batch_size,
                        overlap         = self.config.scan_batch_overlap,
                        timeout_per_msg = self.config.scan_timeout_per_msg,
                        max_messages    = self.config.scan_max_messages,
                    )
                else:
                    screen = _llm_screen(df_work, code, self.client, target_id, role)
                new_s    = [c for c in screen if c["msg_id"] not in seen_ids]
                screen_v = _validate_candidates(new_s, code, self.client, df=df_work) if new_s else []
                for c in screen_v:
                    if c["msg_id"] not in seen_ids:
                        seen_ids.add(c["msg_id"])
                        validated.append(c)
                if self.verbose:
                    print(f"+{len(screen_v)} → итого {len(validated)}")

            result[code] = _enrich_with_context(
                validated, df_work,
                time_window_minutes = self.config.context_window_minutes,
                n_positional        = self.config.context_n_positional,
                min_context_msgs    = self.config.context_min_msgs,
            )
            if self.verbose:
                print(f"  [Stage 1 / {code}] итого: {len(result[code])} кандидатов за {_fmt_t(time.time() - t0)}")

        return result


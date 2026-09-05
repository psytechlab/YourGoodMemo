"""
stage2_p2_detector.py
=====================
Детекторы точек опоры 2-го приоритета.

S3 — Признание и самооценочная поддержка
C1 — Надежда и оптимизм
D3 — Хобби, увлечения и поток
E2 — Копинг-репертуар и стратегии регуляции (длинный контекст, mistral-nemo)
"""

import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anchor_detection.llm_client import OllamaClient


# ══════════════════════════════════════════════════════════════════════
# Общие утилиты
# ══════════════════════════════════════════════════════════════════════

_COMMON_RULES = """
ОБЩИЕ ПРАВИЛА:
1. Используй ТОЛЬКО предоставленный контекст. Не выдумывай message_ids.
2. При недостатке данных — strength <= 0.4, data_sufficiency="low".
3. Ответ строго в JSON на русском языке. Все текстовые поля только на русском.
4. Интернет-мемы, шаблонные фразы и шутки НЕ являются доказательствами.
"""

_OUTPUT_SCHEMA = """
Формат ответа:
{
  "anchor_code": "<код>",
  "evidence_found": true | false,
  "candidate_evidence": [
    {
      "message_ids": ["msg_123"],
      "why": "1-2 предложения — что конкретно здесь видно"
    }
  ],
  "subscores": { ... },
  "data_sufficiency": "low" | "medium" | "high"
}
"""


def _fmt_with_context(c: Dict) -> str:
    """Форматирует кандидата с соседними сообщениями (context_before/context_after)."""
    lines = []
    for m in c.get("context_before", []):
        lines.append(f"  ↑ [{m['msg_id']}] {m['sender']}: {m['text']!r}")
    role = c.get("role", "")
    sent_str = f" [{c['sentiment']}]" if c.get("sentiment") else ""
    lines.append(
        f"[{c['msg_id']}] role={role}{sent_str} {c.get('ts','')[:10]} {c['sender']}: {c['text']!r}"
    )
    for m in c.get("context_after", []):
        lines.append(f"  ↓ [{m['msg_id']}] {m['sender']}: {m['text']!r}")
    return "\n".join(lines)


def _sentiment_summary(msgs: List[Dict]) -> str:
    """Распределение тональностей для добавления в контекст промпта."""
    if not msgs:
        return ""
    counts: Dict[str, int] = {}
    for m in msgs:
        s = m.get("sentiment", "")
        if s:
            counts[s] = counts.get(s, 0) + 1
    if not counts:
        return ""
    total = len(msgs)
    return ", ".join(
        f"{s}: {counts.get(s, 0)}/{total}"
        for s in ["positive", "neutral", "negative"]
        if counts.get(s, 0) > 0
    )


def _default(code: str) -> Dict:
    return {
        "anchor_code": code,
        "evidence_found": False,
        "candidate_evidence": [],
        "subscores": {},
        "data_sufficiency": "low",
        "caveat": f"Stage 2 не нашёл кандидатов для {code}.",
    }


def _sanitize(result: Dict, code: str) -> Dict:
    """Принудительно устанавливает корректные значения после ответа LLM."""
    result["anchor_code"] = code
    result.setdefault("evidence_found",     False)
    result.setdefault("candidate_evidence", [])
    result.setdefault("subscores",          {})
    result.setdefault("data_sufficiency",   "low")
    # evidence_found не может быть True при пустом candidate_evidence
    if not result["candidate_evidence"]:
        result["evidence_found"] = False
    return result


# ══════════════════════════════════════════════════════════════════════
# S3 — Признание и самооценочная поддержка
# ══════════════════════════════════════════════════════════════════════

_S3_SYSTEM = """Ты — детектор точки опоры S3 «Признание и самооценочная поддержка».

Задача: определить, получает ли ПОЛЬЗОВАТЕЛЬ (таргет) от ДРУГИХ ЛЮДЕЙ позитивную обратную связь.

КРИТИЧЕСКИ ВАЖНО — НАПРАВЛЕНИЕ ПОХВАЛЫ:
Тебе важно найти сообщения, где СОБЕСЕДНИК хвалит ПОЛЬЗОВАТЕЛЯ.
Похвала ОТ пользователя В АДРЕС собеседника — НЕ засчитывается.
В каждом кандидате указан sender (отправитель). Роль кандидата "praise_received"
означает что это сообщение ОТ СОБЕСЕДНИКА К ПОЛЬЗОВАТЕЛЮ.

ПОЛОЖИТЕЛЬНЫЕ СИГНАЛЫ:
- Собеседник хвалит пользователя: «молодец», «умница», «отлично сделал», «ты крутой»
- Благодарность пользователю за конкретное действие: «спасибо, ты выручил»
- Признание компетентности: «ты хорошо разбираешься», «без тебя бы не справился»
- Реакция собеседника на достижение пользователя

ОПОРА ЕСТЬ: >= 2 разных эпизодов похвалы ОТ других ИЛИ 1 глубокое признание.

ANTI-PATTERNS (не засчитывать):
- Пользователь хвалит собеседника (неверное направление)
- Шаблонное «спасибо» без контекста значимости
- Ирония или сарказм

""" + _COMMON_RULES + _OUTPUT_SCHEMA + """

subscores для S3:
{
  "explicit_praise": 0.0-1.0,
  "gratitude_received": 0.0-1.0,
  "competence_recognition": 0.0-1.0
}"""


def detect_S3_stage2(candidates: List[Dict], agg: Dict, client: OllamaClient) -> Dict:
    if not candidates:
        return _default("S3")
    payload: Dict[str, Any] = {"aggregates": agg, "candidates": candidates}
    sent = _sentiment_summary(candidates)
    if sent:
        payload["sentiment_distribution"] = sent
    user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)
    try:
        result = client.chat(system=_S3_SYSTEM, user=user_prompt, temperature=0.1,
                             timeout=max(360,len(candidates) * 10))
        return _sanitize(result, "S3")
    except Exception as exc:
        print(f"  [S3] Ошибка: {exc}")
        return _default("S3")


# ══════════════════════════════════════════════════════════════════════
# C1 — Надежда и оптимизм
# ══════════════════════════════════════════════════════════════════════

_C1_SYSTEM = """Ты — детектор точки опоры C1 «Надежда и оптимизм».

Задача: определить, видит ли пользователь своё «завтра» — строит ли конкретные планы,
смотрит ли в будущее с позитивным ожиданием.

ПОЛОЖИТЕЛЬНЫЕ СИГНАЛЫ:
- Конкретные планы с временным горизонтом: «в следующем месяце поеду», «записался на»
- Позитивные ожидания: «жду не дождусь», «будет классно»
- Упоминание будущих событий (поездки, встречи, цели)
- «Завтра» как место, где что-то хорошее — не просто завтрашний день

ОПОРА ЕСТЬ: >= 3 разных конкретных плана ИЛИ устойчивый паттерн позитивного будущего.

ANTI-PATTERNS:
- «Буду дома» или «завтра работа» — нейтральное расписание, не опора
- Тревожные ожидания: «не знаю что будет», «боюсь»
- Единичная фраза без контекста

""" + _COMMON_RULES + _OUTPUT_SCHEMA + """

subscores для C1:
{
  "concrete_plans": 0.0-1.0,
  "positive_future_sentiment": 0.0-1.0,
  "temporal_specificity": 0.0-1.0
}"""


def detect_C1_stage2(candidates: List[Dict], agg: Dict, client: OllamaClient) -> Dict:
    if not candidates:
        return _default("C1")
    payload: Dict[str, Any] = {"aggregates": agg, "candidates": candidates}
    sent = _sentiment_summary(candidates)
    if sent:
        payload["sentiment_distribution"] = sent
    user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)
    try:
        result = client.chat(system=_C1_SYSTEM, user=user_prompt, temperature=0.1,
                             timeout=max(360,len(candidates) * 10))
        return _sanitize(result, "C1")
    except Exception as exc:
        print(f"  [C1] Ошибка: {exc}")
        return _default("C1")


# ══════════════════════════════════════════════════════════════════════
# D3 — Хобби, увлечения и поток
# ══════════════════════════════════════════════════════════════════════

_D3_SYSTEM = """Ты — детектор точки опоры D3 «Хобби, увлечения и поток».

Задача: определить, есть ли у пользователя регулярная деятельность, которой он занимается
ради самого процесса — спорт, игры, творчество, учёба, любое хобби.

ПОЛОЖИТЕЛЬНЫЕ СИГНАЛЫ:
- Регулярные упоминания одной активности (не разовые)
- Прогресс: «побил рекорд», «прошёл уровень», «дочитал», «дорисовал»
- Эмоциональная вовлечённость: «классно получилось», «кайфую от»
- Медиа по теме хобби (фото забега, скриншоты игры)
- Приглашение других разделить активность

ОПОРА ЕСТЬ: одна активность упоминается >= 5 раз за период ИЛИ 3 разные активности.

ANTI-PATTERNS:
- Разовое упоминание без продолжения
- Работа как «хобби» (если это основная деятельность)
- «Смотрел телевизор» без вовлечённости

""" + _COMMON_RULES + _OUTPUT_SCHEMA + """

subscores для D3:
{
  "regularity": 0.0-1.0,
  "emotional_engagement": 0.0-1.0,
  "progression": 0.0-1.0,
  "variety": 0.0-1.0
}"""


def detect_D3_stage2(candidates: List[Dict], agg: Dict, client: OllamaClient) -> Dict:
    if not candidates:
        return _default("D3")
    payload: Dict[str, Any] = {"aggregates": agg, "candidates": candidates}
    sent = _sentiment_summary(candidates)
    if sent:
        payload["sentiment_distribution"] = sent
    user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)
    try:
        result = client.chat(system=_D3_SYSTEM, user=user_prompt, temperature=0.1,
                             timeout=max(360,len(candidates) * 10))
        return _sanitize(result, "D3")
    except Exception as exc:
        print(f"  [D3] Ошибка: {exc}")
        return _default("D3")


# ══════════════════════════════════════════════════════════════════════
# E2 — Копинг-репертуар и стратегии регуляции
# ══════════════════════════════════════════════════════════════════════

_E2_SYSTEM = """Ты — детектор точки опоры E2 «Копинг-репертуар и стратегии регуляции».

Задача: найти эпизоды, когда пользователь столкнулся с трудностью (стрессором) и
показал активный, здоровый способ справляться — не застрял, не ушёл в изоляцию,
а продолжил жить, общаться, действовать.

ФОРМАТ ВХОДНЫХ ДАННЫХ:
Тебе переданы эпизоды. Каждый эпизод — это стрессор + то, что происходило после.

ПОЛОЖИТЕЛЬНЫЕ СИГНАЛЫ (активный копинг ПОСЛЕ стрессора):
- Встречался с друзьями, звонил, выходил из дома
- Продолжал работать, учиться, заниматься хобби
- Искал помощь или совет
- Использовал юмор по отношению к ситуации
- Формулировал «сделаю вот так» вместо «всё пропало»
- Через 1-4 недели после стрессора писал нормально, без признаков застревания

ANTI-PATTERNS (НЕ опора, риск):
- Изоляция: перестал отвечать, пропал из переписки на недели
- «Хочу бухать», «ничего не хочу», апатия без выхода
- Признаки застревания: одна тема несколько недель без движения
- Единственный эпизод трудности — нужен паттерн (>= 2 эпизодов)

ВАЖНО: не путай активный копинг с отрицанием. Человек может грустить И продолжать жить.
Если он грустил, но через неделю снова смеялся и встречался с людьми — это ОПОРА.

""" + _COMMON_RULES + _OUTPUT_SCHEMA + """

subscores для E2:
{
  "active_coping_episodes": 0.0-1.0,
  "avoidance_absence": 0.0-1.0,
  "strategy_diversity": 0.0-1.0,
  "recovery_speed": 0.0-1.0
}"""


def detect_E2_stage2(
    episodes: List[Dict],   # [{"stressor_msg": ..., "aftermath_msgs": [...]}]
    agg:      Dict,
    client:   OllamaClient,
) -> Dict:
    """
    E2 требует передачи ЭПИЗОДОВ (стрессор + поведение после), а не отдельных сообщений.
    Использует модель с длинным контекстом (mistral-nemo).
    """
    if not episodes:
        return _default("E2")

    # Форматируем эпизоды для LLM; тональность [positive/neutral/negative] показывает окрас
    lines = []
    for i, ep in enumerate(episodes[:5], 1):
        stressor = ep.get("stressor_msg", {})
        aftermath = ep.get("aftermath_msgs", [])
        lines.append(f"\n--- Эпизод {i} ---")
        st_sent = stressor.get("sentiment", "")
        st_sent_s = f" [{st_sent}]" if st_sent else ""
        lines.append(
            f"СТРЕССОР [{stressor.get('msg_id','?')}]{st_sent_s} "
            f"{stressor.get('ts','')[:10]} {stressor.get('sender','')}: "
            f"{stressor.get('text','')!r}"
        )
        lines.append("ПОСЛЕ (следующие 2-3 недели):")
        for m in aftermath[:20]:
            m_sent = m.get("sentiment", "")
            m_sent_s = f" [{m_sent}]" if m_sent else ""
            lines.append(
                f"  [{m.get('msg_id','?')}]{m_sent_s} {m.get('ts','')[:10]} "
                f"{m.get('sender','')}: {m.get('text','')!r}"
            )

    user_prompt = "\n".join(lines)

    try:
        n_msgs = sum(1 + len(ep.get("aftermath_msgs", [])) for ep in episodes[:5])
        result = client.chat(system=_E2_SYSTEM, user=user_prompt, temperature=0.1,
                             timeout=max(360, n_msgs * 10))
        return _sanitize(result, "E2")
    except Exception as exc:
        print(f"  [E2] Ошибка: {exc}")
        return _default("E2")


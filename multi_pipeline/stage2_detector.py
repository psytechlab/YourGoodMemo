"""
stage2_detector.py
==================
Ступень 2: детекция точек опоры по предварительно размеченным сообщениям.

Ключевое отличие от монолитного подхода:
  - Детектор получает ТОЛЬКО сообщения с релевантными метками Stage 1.
  - Контекст в 5-10 раз меньше → меньше шума → меньше галлюцинаций.
  - Задача каждого детектора проще и точнее.

Выход каждого детектора — dict с полями candidate_evidence,
который передаётся в Stage 3 для валидации.
"""

import json
from typing import Any, Dict, List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from multi_pipeline.config import PipelineConfig
from anchor_detection.llm_client import OllamaClient


# ══════════════════════════════════════════════════════════════════════
# Общие части промтов
# ══════════════════════════════════════════════════════════════════════

_RULES = """
ПРАВИЛА:
1. Используй ТОЛЬКО сообщения из предоставленного контекста.
   Не выдумывай message_ids, цитаты и имена.
2. Каждый элемент evidence обязан ссылаться на реальные msg_id из контекста.
3. Если данных мало — укажи data_sufficiency="low" и не завышай strength.
4. Мемы, шаблонные фразы, риторика — НЕ являются доказательствами.
5. Ответ строго в JSON. Все текстовые поля — только на русском языке.
6. Каждое сообщение содержит поля context_before и context_after — соседние сообщения в диалоге.
   Используй их для понимания тона и смысла.
"""


def _fmt_with_context(c: Dict) -> str:
    """Форматирует кандидата с соседними сообщениями для LLM."""
    lines = []
    for m in c.get("context_before", []):
        lines.append(f"  ↑ [{m['msg_id']}] {m['sender']}: {m['text']!r}")
    role = c.get("role", "")
    # Добавляем тональность в строку если она есть (помогает LLM понять окрас)
    sent = c.get("sentiment", "")
    sent_str = f" [{sent}]" if sent else ""
    lines.append(
        f"[{c['msg_id']}] role={role}{sent_str} {c.get('ts', '')[:10]} {c['sender']}: {c['text']!r}"
    )
    for m in c.get("context_after", []):
        lines.append(f"  ↓ [{m['msg_id']}] {m['sender']}: {m['text']!r}")
    return "\n".join(lines)


def _sentiment_summary(msgs: List[Dict]) -> str:
    """
    Возвращает строку с распределением тональностей для заголовка промпта.
    Пример: 'positive: 2/7, neutral: 4/7, negative: 1/7'
    Если данные о тональности отсутствуют — возвращает пустую строку.
    """
    if not msgs:
        return ""
    counts: Dict[str, int] = {}
    has_sent = False
    for m in msgs:
        s = m.get("sentiment", "")
        if s:
            has_sent = True
            counts[s] = counts.get(s, 0) + 1
    if not has_sent:
        return ""
    total = len(msgs)
    order = ["positive", "neutral", "negative"]
    parts = [f"{s}: {counts.get(s, 0)}/{total}" for s in order if counts.get(s, 0) > 0]
    return ", ".join(parts)

_SCHEMA = """
Формат ответа:
{
  "anchor_code": "<код>",
  "evidence_found": true | false,
  "candidate_evidence": [
    {
      "message_ids": ["msg_123", "msg_456"],
      "why": "1-2 предложения: что именно здесь указывает на опору"
    }
  ],
  "subscores": { ... },
  "data_sufficiency": "low" | "medium" | "high"
}
"""


def _validate_candidate(result: Dict, code: str) -> Dict:
    """Дополняет обязательные поля и применяет санти-чеки."""
    result["anchor_code"] = code          # принудительно — LLM иногда пишет название вместо кода
    result.setdefault("evidence_found",      False)
    result.setdefault("candidate_evidence",  [])
    result.setdefault("subscores",           {})
    result.setdefault("data_sufficiency",    "low")
    # Санти-чек: evidence_found не может быть True при пустом candidate_evidence
    if not result["candidate_evidence"]:
        result["evidence_found"] = False
    return result


# ══════════════════════════════════════════════════════════════════════
# S1 — Эмоциональная поддержка и близкие связи
# ══════════════════════════════════════════════════════════════════════

_S1_SYSTEM = """Ты — детектор точки опоры S1 «Эмоциональная поддержка и близкие связи».

Тебе переданы два набора:
  target_messages  — сообщения ПОЛЬЗОВАТЕЛЯ, помеченные как "emotional"
                     (он делится чувствами, переживаниями)
  contact_messages — сообщения КОНТАКТА, помеченные как "empathy"
                     (он сочувствует, поддерживает)

Задача: определить, есть ли между ними устойчивая связь эмоциональной поддержки.

СИГНАЛЫ ОПОРЫ:
  - Пользователь несколько раз за период открывается эмоционально (не единичный всплеск).
  - Контакт отвечает поддержкой, а не обесцениванием.
  - История диалога длится несколько месяцев.

АНТИ-ПАТТЕРНЫ:
  - Единственный эмоциональный всплеск без истории.
  - Контакт только получает поддержку, но не даёт (асимметрия).
  - Шаблонные «держись» без реального отклика.
""" + _RULES + _SCHEMA + """
subscores для S1:
{
  "self_disclosure_depth":  0.0-1.0,
  "empathic_response_quality": 0.0-1.0,
  "pattern_regularity":    0.0-1.0,
  "relationship_longevity": 0.0-1.0
}"""


def detect_S1_stage2(
    target_msgs:  List[Dict],
    contact_msgs: List[Dict],
    aggregates:   Dict,
    client:       OllamaClient,
) -> Dict:
    payload: Dict[str, Any] = {
        "aggregates":       aggregates,
        "target_messages":  target_msgs,
        "contact_messages": contact_msgs,
    }
    sent_t = _sentiment_summary(target_msgs)
    sent_c = _sentiment_summary(contact_msgs)
    if sent_t or sent_c:
        payload["sentiment_target_msgs"]  = sent_t or "нет данных"
        payload["sentiment_contact_msgs"] = sent_c or "нет данных"
    user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)

    try:
        n = len(target_msgs) + len(contact_msgs)
        result = client.chat(system=_S1_SYSTEM, user=user_prompt, temperature=0.1,
                             timeout=max(360,n * 10))
        return _validate_candidate(result, "S1")
    except Exception as exc:
        print(f"  [Stage 2 / S1] Ошибка: {exc}")
        return _validate_candidate({"data_sufficiency": "low"}, "S1")


# ══════════════════════════════════════════════════════════════════════
# S2 — Принадлежность к сообществу
# ══════════════════════════════════════════════════════════════════════

_S2_SYSTEM = """Ты — детектор точки опоры S2 «Принадлежность к сообществу».

Тебе переданы сообщения из личных переписок, содержащие местоимения
«мы / наш / у нас» или упоминания совместных активностей.

Задача: определить, есть ли у пользователя ощущение принадлежности к живому сообществу.

СИГНАЛЫ ОПОРЫ:
  - Пользователь регулярно (>= 1 раза в неделю) участвует в группе.
  - Использует «мы» применительно к группе (не риторически).
  - Есть регулярные совместные события (один день недели, общие темы).

АНТИ-ПАТТЕРНЫ:
  - Рабочие чаты с формальными сообщениями.
  - «Мы» в общем смысле («мы все», «мы как общество»).
  - Единственная группа без реального участия.
""" + _RULES + _SCHEMA + """
subscores для S2:
{
  "activity_regularity": 0.0-1.0,
  "we_pronoun_authenticity": 0.0-1.0,
  "group_events_regularity": 0.0-1.0,
  "mutual_recognition": 0.0-1.0
}"""


def detect_S2_stage2(
    group_msgs: List[Dict],
    aggregates: Dict,
    client:     OllamaClient,
) -> Dict:
    if not group_msgs:
        return _validate_candidate({
            "data_sufficiency": "low",
            "candidate_evidence": [],
        }, "S2")

    payload: Dict[str, Any] = {"aggregates": aggregates, "group_messages": group_msgs}
    sent = _sentiment_summary(group_msgs)
    if sent:
        payload["sentiment_distribution"] = sent
    user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)

    try:
        result = client.chat(system=_S2_SYSTEM, user=user_prompt, temperature=0.1,
                             timeout=max(360,len(group_msgs) * 10))
        return _validate_candidate(result, "S2")
    except Exception as exc:
        print(f"  [Stage 2 / S2] Ошибка: {exc}")
        return _validate_candidate({}, "S2")


# ══════════════════════════════════════════════════════════════════════
# S5 — «Я нужен» / ответственность за других
# ══════════════════════════════════════════════════════════════════════

_S5_SYSTEM = """Ты — детектор точки опоры S5 «Я нужен / ответственность за других».

Тебе переданы сообщения от собеседников пользователя, помеченные как "help_req"
(содержат просьбу о помощи, действии или совете).

Задача: определить, есть ли регулярный паттерн — люди обращаются к пользователю
за помощью, он нужен им, несёт реальную ответственность.

СИГНАЛЫ ОПОРЫ:етные (сделай X, подскажи Y, помоги с Z).
  - Паттерн повторяется: не единичная просьба, а регулярность.

АНТИ-ПАТТЕРНЫ:
  - Один и тот же
  - Несколько разных людей обращаются к пользователю за помощью.
  - Просьбы конкр человек с мелкими бытовыми вопросами (не помощь, а разговор).
  - Риторические вопросы без ожидания действия.
  - Манипулятивные «ты мне нужен» в конфликте — это не опора.
  - Мемы и шаблонные фразы.
""" + _RULES + _SCHEMA + """
subscores для S5:
{
  "help_request_regularity": 0.0-1.0,
  "request_specificity":     0.0-1.0,
  "requester_diversity":     0.0-1.0,
  "exploitation_risk":       0.0-1.0
}"""


def detect_S5_stage2(
    help_msgs:  List[Dict],
    ping_msgs:  List[Dict],
    aggregates: Dict,
    client:     OllamaClient,
) -> Dict:
    payload: Dict[str, Any] = {
        "aggregates":            aggregates,
        "help_request_messages": help_msgs,
        "post_silence_pings":    ping_msgs,
    }
    sent = _sentiment_summary(help_msgs)
    if sent:
        payload["sentiment_help_msgs"] = sent
    user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)

    try:
        n = len(help_msgs) + len(ping_msgs)
        result = client.chat(system=_S5_SYSTEM, user=user_prompt, temperature=0.1,
                             timeout=max(360,n * 10))
        return _validate_candidate(result, "S5")
    except Exception as exc:
        print(f"  [Stage 2 / S5] Ошибка: {exc}")
        return _validate_candidate({}, "S5")


# ══════════════════════════════════════════════════════════════════════
# D2 — Финансовая безопасность
# ══════════════════════════════════════════════════════════════════════

_D2_SYSTEM = """Ты — детектор точки опоры D2 «Финансовая безопасность».

Тебе переданы сообщения пользователя, помеченные как "financial"
(содержат упоминания денег, трат, доходов, долгов).

Задача: определить ощущение материальной устойчивости — отсутствие острых угроз.

СИГНАЛЫ ОПОРЫ:
  - Плановые траты без тревоги (отпуск, подарки, ремонт).
  - Снижение упоминаний долгов в динамике.
  - Готовность тратить на других (щедрость).
  - «Закрыл кредит», «накопил», «откладываю».

АНТИ-ПАТТЕРНЫ:
  - Демонстративное потребление при тревожном фоне («купил машину» + «нечем платить»).
  - Слово «должен/должна» в смысле «обязан» (не финансовый долг).
  - Единственная крупная покупка без контекста.
  - Жалобы «всё дорого» рядом с тратами.

ВАЖНО: Различай «должен» как финансовый долг и «должен» как обязанность/ожидание.
""" + _RULES + _SCHEMA + """
subscores для D2:
{
  "planned_spending_score":   0.0-1.0,
  "debt_anxiety_absence":     0.0-1.0,
  "generosity_score":         0.0-1.0,
  "positive_events_score":    0.0-1.0
}"""


def detect_D2_stage2(
    financial_msgs: List[Dict],
    aggregates:     Dict,
    client:         OllamaClient,
) -> Dict:
    payload: Dict[str, Any] = {"aggregates": aggregates, "financial_messages": financial_msgs}
    sent = _sentiment_summary(financial_msgs)
    if sent:
        payload["sentiment_distribution"] = sent
    user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)

    try:
        result = client.chat(system=_D2_SYSTEM, user=user_prompt, temperature=0.1,
                             timeout=max(360,len(financial_msgs) * 10))
        return _validate_candidate(result, "D2")
    except Exception as exc:
        print(f"  [Stage 2 / D2] Ошибка: {exc}")
        return _validate_candidate({}, "D2")


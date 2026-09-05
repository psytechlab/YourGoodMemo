"""
stage4_narrator.py
==================
Ступень 4: генерация поддерживающего нарративного текста.

Новый формат вывода:
  <Хук — 1 предложение о том, что у человека есть>

     Например, ты ...:
     → [дата] Имя: «текст
                    продолжение текста»

     А ещё ...:
     → [дата] Имя: «текст»
"""

import json
from typing import Dict, List, Optional

import pandas as pd

from anchor_detection.llm_client import OllamaClient


# ══════════════════════════════════════════════════════════════════════
# Описания точек опоры
# ══════════════════════════════════════════════════════════════════════

ANCHOR_CONTEXT = {
    "S1": {
        "name": "Близкий человек рядом",
        "hint": "Кто-то, с кем можно поделиться чем угодно и получить тепло в ответ.",
    },
    "S2": {
        "name": "Своя компания",
        "hint": "Люди, с которыми ты чувствуешь себя своим — совместные дела, общие воспоминания.",
    },
    "S3": {
        "name": "Тебя ценят",
        "hint": "Люди замечают, что ты делаешь, и говорят об этом — хвалят, благодарят, признают.",
    },
    "S5": {
        "name": "Ты нужен",
        "hint": "Люди обращаются к тебе, ждут тебя, ценят твоё присутствие.",
    },
    "D2": {
        "name": "Финансовая устойчивость",
        "hint": "Нет острой тревоги о деньгах — можно планировать, тратить на близких.",
    },
    "D3": {
        "name": "Твоё дело",
        "hint": "Есть занятие, которое ты делаешь ради самого процесса — и возвращаешься к нему снова.",
    },
    "C1": {
        "name": "Ты смотришь вперёд",
        "hint": "Есть конкретное «завтра» — поездки, планы, события, которых ты ждёшь.",
    },
    "E2": {
        "name": "Ты умеешь справляться",
        "hint": "Когда становилось трудно — ты не застревал. Продолжал жить, общаться, двигаться.",
    },
}


# ══════════════════════════════════════════════════════════════════════
# Промт для LLM
# ══════════════════════════════════════════════════════════════════════

_NARRATOR_SYSTEM = """Ты описываешь конкретный аспект жизни человека на основе фрагментов его переписки.

Верни JSON строго по этой схеме, без пояснений:
{
  "hook": "<одно короткое предложение, 2-е лицо — что у человека есть>",
  "sections": [
    {
      "intro": "<вводная фраза 2-е лицо, ≤12 слов>",
      "message_ids": ["<id1>", "<id2>"]
    }
  ]
}

Правила для hook:
- Обращение на «ты»: «У тебя есть...», «Тебя ценят за...», «Есть люди...»
- Конкретно и просто, максимум 12 слов
- ЗАПРЕЩЕНО: «опора», «ресурс», «поддержка», «благополучие»

Правила для sections (одна секция = одна улика):
- intro начинается с «Например,» / «А ещё» / «Одна из таких вещей»
- Обращение на «ты»: «Например, ты часто...», «А ещё ты...»
- Описывай конкретно ЧТО происходит, не пересказывай why дословно
- НЕ заканчивай intro двоеточием (оно добавится автоматически)
- Копируй message_ids ТОЧНО из улики — ни символа не меняй
- ЗАПРЕЩЕНО в intro: «переписка», «сообщение», «чат», «переписываться»"""


# ══════════════════════════════════════════════════════════════════════
# Форматирование сообщений
# ══════════════════════════════════════════════════════════════════════

_MONTHS_RU = {
    1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель",
    5: "Май", 6: "Июнь", 7: "Июль", 8: "Август",
    9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь",
}


def _fmt_date(ts_str: str) -> str:
    """Преобразует '2026-03-23 12:09:...' → '2026 Март'."""
    try:
        parts = str(ts_str)[:10].split("-")
        year = parts[0]
        month = int(parts[1])
        return f"{year} {_MONTHS_RU.get(month, parts[1])}"
    except (ValueError, IndexError):
        return str(ts_str)[:10]


def _fmt_chat_message(row: Dict, prefix: str = "   → ") -> str:
    """
    Форматирует одно сообщение в стиле чата с выравниванием многострочного текста.

      → 2026 Март — Захар: «Первая строка
                             Вторая строка»
    """
    text = str(row.get("text", "")).strip()
    if not text:
        return ""
    sender = str(row.get("sender", ""))
    ts = _fmt_date(str(row.get("ts") or row.get("date", "")))

    header = f"{prefix}{ts} — {sender}: «"
    padding = " " * len(header)

    parts = [p.strip() for p in text.split("\n") if p.strip()]
    if not parts:
        return ""

    if len(parts) == 1:
        return f"{header}{parts[0]}»"

    result = header + parts[0]
    for p in parts[1:]:
        result += "\n" + padding + p
    result += "»"
    return result


def _format_narrative(hook: str, sections: List[Dict], msg_index: Dict) -> str:
    """Собирает итоговый текст нарратива из хука и секций с сообщениями."""
    # Нормализуем хук: точка в конце
    hook = hook.strip()
    if hook and hook[-1] not in ".!?":
        hook += "."

    lines: List[str] = [hook, ""]

    for section in sections:
        intro = section.get("intro", "").strip().rstrip(":")
        msg_ids = section.get("message_ids", [])
        if not intro or not msg_ids:
            continue

        lines.append(f"   {intro}:")

        for mid_str in msg_ids:
            row = msg_index.get(str(mid_str), {})
            msg_line = _fmt_chat_message(row)
            if msg_line:
                lines.append(msg_line)

        lines.append("")  # пустая строка между секциями

    # Убираем хвостовые пустые строки
    while lines and not lines[-1]:
        lines.pop()

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# Подготовка промта для LLM
# ══════════════════════════════════════════════════════════════════════

def _build_user_prompt(anchor_code: str, evidence: List[Dict], msg_index: Dict) -> str:
    """Строит user-prompt с уликами и примерами текстов сообщений."""
    ctx = ANCHOR_CONTEXT.get(anchor_code, {})
    anchor_name = ctx.get("name", anchor_code)
    anchor_hint = ctx.get("hint", "")

    lines = [
        f"Тема: «{anchor_name}»",
        f"Контекст: {anchor_hint}",
        "",
        "Принятые улики:",
    ]

    for i, ev in enumerate(evidence[:5], 1):
        lines.append(f"\n[Улика {i}]")
        msg_ids = ev.get("message_ids", [])
        lines.append(f"message_ids: {json.dumps(msg_ids, ensure_ascii=False)}")
        why = ev.get("why", ev.get("verdict_text", ""))
        if why:
            lines.append(f"Описание: {why}")

        msg_texts = []
        for mid in msg_ids[:3]:
            row = msg_index.get(str(mid), {})
            text = str(row.get("text", "")).strip()[:150]
            if text:
                sender = str(row.get("sender", ""))
                msg_texts.append(f"  {sender}: «{text}»")
        if msg_texts:
            lines.append("Тексты сообщений:")
            lines.extend(msg_texts)

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# Основной класс
# ══════════════════════════════════════════════════════════════════════

class AnchorNarrator:
    """
    Генерирует нарративный текст для найденных точек опоры.

    Параметры
    ----------
    model : str — модель Ollama
    ollama_url : str
    num_ctx : int
    """

    def __init__(
        self,
        model:      str = "mistral-nemo:12b",
        ollama_url: str = "http://localhost:11434",
        num_ctx:    int = 4096,
    ):
        self.client = OllamaClient(
            model    = model,
            base_url = ollama_url,
            num_ctx  = num_ctx,
            timeout  = 120,
        )
        self.model = model

    def generate(
        self,
        anchor_code:  str,
        evidence:     List[Dict],
        msg_index:    Dict,
        target_name:  str = "",
        df:           Optional[pd.DataFrame] = None,
    ) -> str:
        """
        Генерирует нарративный блок для одной точки опоры.

        Возвращает текст в формате:
          <хук>

             Например, ты ...:
             → [дата] Имя: «текст»
        """
        if not evidence:
            return ""

        user_prompt = _build_user_prompt(anchor_code, evidence, msg_index)

        try:
            result = self.client.chat(
                system      = _NARRATOR_SYSTEM,
                user        = user_prompt,
                temperature = 0.5,
                json_mode   = True,
            )
        except Exception as exc:
            print(f"  [Stage 4 / {anchor_code}] Ошибка: {exc}")
            return ""

        if not isinstance(result, dict):
            return ""

        hook = str(result.get("hook", "")).strip()
        sections = result.get("sections", [])

        if not hook:
            ctx = ANCHOR_CONTEXT.get(anchor_code, {})
            hook = ctx.get("hint", "")

        if not sections:
            return hook

        return _format_narrative(hook, sections, msg_index)

    def generate_all(
        self,
        profile:   Dict,
        msg_index: Dict,
        df:        Optional[pd.DataFrame] = None,
    ) -> Dict[str, str]:
        """
        Генерирует нарративы для всех найденных точек опоры из профиля.

        Возвращает
        ----------
        dict: anchor_code → narrative_text
        """
        narratives: Dict[str, str] = {}
        target = profile.get("meta", {}).get("target", "")

        for code, det in profile.get("detectors", {}).items():
            if not det.get("evidence_found"):
                continue

            accepted = [
                ev for ev in det.get("evidence", [])
                if ev.get("verdict") in ("accepted", None)
            ]
            if not accepted:
                continue

            print(f"  [Stage 4 / {code}] Генерирую нарратив...")
            text = self.generate(code, accepted, msg_index, target, df=df)
            if text:
                narratives[code] = text
                print(f"    ✓ {len(text)} символов")

        return narratives

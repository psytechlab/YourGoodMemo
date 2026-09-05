"""
stage5_composer.py
==================
Ступень 5: сборка полного поддерживающего ответа.

Из найденных точек опоры (с их нарративами) генерирует единый текст:
  — Вступление (2-3 предложения, лично к человеку)
  — Разделы по каждой найденной точке опоры
  — Заключение (2-3 предложения, призыв)

Модель: mistral-nemo:12b (лучший русский из доступных локально).
"""

import re
from typing import Dict, List, Optional
from anchor_detection.llm_client import OllamaClient


ANCHOR_NAMES = {
    "S1": "Близкий человек рядом",
    "S2": "Своя компания",
    "S3": "Тебя ценят",
    "S5": "Ты нужен",
    "D2": "Финансовая устойчивость",
    "D3": "Твоё дело",
    "C1": "Ты смотришь вперёд",
    "E2": "Ты умеешь справляться",
}

_COMPOSER_SYSTEM = """Ты собираешь единый текст для человека из нескольких готовых блоков.

Каждый блок описывает конкретный аспект его жизни. Твоя задача — соединить их
в связный, честный текст с тремя частями.

СТРУКТУРА:
  [ВСТУПЛЕНИЕ — 2-3 предложения]
  Обращение к человеку без пафоса. Не «посмотри, сколько всего хорошего» и
  не «всё будет хорошо». Что-то вроде: «Бывает трудно видеть хорошее, когда
  все не очень хорошо. Но вот что точно есть.» Никакой эйфории.

  [ТЕЛО]
  Каждый блок — отдельный абзац. Используй тексты из блоков почти дословно.
  Между блоками — тихие переходы («Ещё одна вещь», «И отдельно —», «Кроме этого»).
  НЕ добавляй восклицательных знаков и не усиливай эмоции.

  [ЗАКЛЮЧЕНИЕ — 1-2 предложения]
  Без морали. Не «помни об этом» и не «цени это». Просто констатация:
  «Это есть. Не исчезнет.» или подобное.

КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО:
- «Посмотри, сколько всего хорошего», «как прекрасна жизнь»
- «Опора», «ресурс», «поддержка», «психологическое»
- Восклицательные знаки в позитивном контексте
- Клише: «всё будет хорошо», «ты справишься», «верь в себя»

Верни ТОЛЬКО текст — три части подряд, без заголовков."""


class FullResponseComposer:
    """
    Генерирует единый поддерживающий текст из набора нарративов.

    Параметры
    ----------
    model : str
    ollama_url : str
    num_ctx : int
    """

    def __init__(
        self,
        model:      str = "mistral-nemo:12b",
        ollama_url: str = "http://localhost:11434",
        num_ctx:    int = 8192,
    ):
        self.client = OllamaClient(
            model    = model,
            base_url = ollama_url,
            num_ctx  = num_ctx,
            timeout  = 180,
        )
        self.model = model

    def compose(
        self,
        profile:     Dict,
        narratives:  Dict[str, str],
        target_name: str = "",
    ) -> str:
        """
        Собирает полный ответ.

        Параметры
        ----------
        profile    : anchor_profile dict
        narratives : {anchor_code: narrative_text} из Stage 4
        target_name: имя пользователя

        Возвращает
        ----------
        str — полный текст
        """
        if not narratives:
            return ""

        # Собираем блоки — убираем строки-цитаты «→ [date]» перед передачей в LLM,
        # чтобы composer работал с чистым прозаическим текстом
        blocks = []
        for code, text in narratives.items():
            if not text:
                continue
            name = ANCHOR_NAMES.get(code, code)
            prose = _strip_citations(text)
            blocks.append(f"[{name}]\n{prose}")

        if not blocks:
            return ""

        # Источники (чаты)
        chat_names = profile.get("meta", {}).get("chat_names", [])
        chats_str = " и ".join(chat_names) if chat_names else "переписка"

        user_prompt = (
            f"Человека зовут: {target_name or 'пользователь'}\n"
            f"Источник: {chats_str}\n\n"
            f"Блоки для включения в текст ({len(blocks)} шт.):\n\n"
            + "\n\n".join(blocks)
            + "\n\nНапиши единый поддерживающий текст."
        )

        try:
            text = self.client.chat(
                system      = _COMPOSER_SYSTEM,
                user        = user_prompt,
                temperature = 0.6,
                json_mode   = False,
            )
            return str(text).strip()
        except Exception as exc:
            print(f"  [Stage 5] Ошибка генерации: {exc}")
            # Fallback: собираем вручную
            return _manual_compose(narratives, target_name)


def _strip_citations(text: str) -> str:
    """Убирает строки-цитаты «   → [дата]...» из нарратива, оставляя только прозу."""
    lines = []
    for line in text.split("\n"):
        if line.lstrip().startswith("→ "):
            continue
        lines.append(line)
    result = "\n".join(lines)
    while "\n\n\n" in result:
        result = result.replace("\n\n\n", "\n\n")
    return result.strip()


def _manual_compose(narratives: Dict[str, str], target_name: str) -> str:
    """Резервный вариант без LLM — просто соединяем блоки."""
    parts = [t for t in narratives.values() if t]
    if not parts:
        return ""
    intro = "Бывает трудно видеть хорошее, когда не очень хорошо. Но вот что точно есть.\n\n"
    body  = "\n\n".join(parts)
    outro = "\n\nЭто есть. Никуда не денется."
    return intro + body + outro

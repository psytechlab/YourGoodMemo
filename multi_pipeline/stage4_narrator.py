"""
stage4_narrator.py
==================
Ступень 4: генерация поддерживающего нарративного текста.

Для каждой найденной точки опоры модель пишет тёплый, личный абзац (4-6 предложений),
который напоминает человеку о конкретных хороших моментах из его переписки.

Стиль: близкий друг, который знает твою жизнь — без психологических терминов,
с реальными деталями из переписки, с простым призывом к действию.
"""

from typing import Dict, List, Optional

import pandas as pd

from anchor_detection.llm_client import OllamaClient


# ══════════════════════════════════════════════════════════════════════
# Описания точек опоры для нарратора
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
# Системный промт
# ══════════════════════════════════════════════════════════════════════

_NARRATOR_SYSTEM = """Ты пишешь честный, личный текст для человека о конкретном аспекте его жизни.

Задача: на основе реальных фрагментов переписки написать 4-7 предложений,
которые объяснят человеку, ЧТО именно у него есть и почему это имеет значение.

ТОНАЛЬНОСТЬ:
- Спокойно, без эйфории — как разговор с другом, который не делает вид, что всё хорошо
- Обращение на «ты»
- Конкретно: опирайся на детали из переписки, не на абстракции

СТРУКТУРА:
1. Что у тебя есть (конкретное наблюдение, не «сколько всего хорошего»)
2. Откуда это видно — 1-2 конкретных момента из переписки (без ссылок на «переписку»)
3. Что это значит на практике — не «это прекрасно», а «это значит, что...»
4. Опционально: тихое, без восклицательных знаков предложение («можно позвонить», «стоит написать»)

СТРОГО ЗАПРЕЩЕНО — эти фразы делают текст фальшивым:
- «Посмотри, сколько всего хорошего», «как прекрасна жизнь», «радуйся»
- «Не забывай», «помни об этом», «цени это»
- «Опора», «ресурс», «поддержка», «ресурс», «благополучие»
- «Анализ показал», «в переписке», «судя по сообщениям»
- Восклицательные знаки — они звучат как принуждение к радости
- Избыточный сленг из переписки — только если слово естественно вписывается

ФОРМАТ: только текст абзаца, никаких заголовков и пояснений."""


# ══════════════════════════════════════════════════════════════════════
# Основной класс
# ══════════════════════════════════════════════════════════════════════

class AnchorNarrator:
    """
    Генерирует поддерживающий нарративный текст для найденных точек опоры.

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
        Генерирует нарративный абзац для одной точки опоры.

        Параметры
        ----------
        anchor_code : "S1" | "S2" | ...
        evidence    : список evidence-словарей из Stage 3 (с verdict="accepted")
        msg_index   : dict msg_id → row (для извлечения текста)
        target_name : имя пользователя (необязательно)
        df          : объединённый DataFrame всех чатов — для контекста вокруг цитат
        """
        ctx = ANCHOR_CONTEXT.get(anchor_code, {})
        anchor_name = ctx.get("name", anchor_code)
        anchor_hint = ctx.get("hint", "")

        # Строим диалоговые эпизоды (с контекстом) вместо изолированных цитат
        df_sorted = None
        if df is not None:
            df_sorted = df[
                (df["msg_type"] == "message") & df["has_text"]
            ].sort_values("date").reset_index(drop=True)

        episodes = _extract_episodes(evidence, msg_index, df_sorted)
        if not episodes:
            return ""

        proper_names = _extract_proper_names(evidence, msg_index)

        user_prompt = (
            f"Тема: «{anchor_name}»\n"
            f"Контекст: {anchor_hint}\n\n"
            f"Фрагменты переписки (→ отмечает ключевое сообщение):\n\n"
            + "\n\n".join(f"[Момент {i+1}]\n{ep}" for i, ep in enumerate(episodes))
        )
        if proper_names:
            user_prompt += f"\n\nИмена людей в переписке: {', '.join(proper_names)}"
        user_prompt += "\n\nНапиши текст."

        try:
            text = self.client.chat(
                system      = _NARRATOR_SYSTEM,
                user        = user_prompt,
                temperature = 0.7,
                json_mode   = False,
            )
            return str(text).strip()

        except Exception as exc:
            print(f"  [Stage 4 / {anchor_code}] Ошибка: {exc}")
            return ""

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


# ══════════════════════════════════════════════════════════════════════
# Вспомогательные функции
# ══════════════════════════════════════════════════════════════════════

def _extract_proper_names(evidence: List[Dict], msg_index: Dict) -> List[str]:
    """Извлекает только имена собственные (людей) из цитат."""
    import re
    names = []
    seen = set()
    for ev in evidence:
        for mid_str in ev.get("message_ids", []):
            row = msg_index.get(str(mid_str), {})
            if not row:
                continue
            sender = str(row.get("sender", "")).strip()
            if sender and sender.lower() not in seen:
                seen.add(sender.lower())
                # Берём только имя (первое слово), не полное ФИО
                first_name = sender.split()[0]
                if len(first_name) > 2:
                    names.append(first_name)
    return names[:4]  # не больше 4 имён


_STOP_WORDS = frozenset({
    "и", "в", "на", "с", "по", "из", "за", "к", "до", "от", "для", "при",
    "не", "но", "или", "если", "что", "как", "это", "то", "так", "уже",
    "да", "нет", "ну", "вот", "ты", "я", "он", "она", "мы", "вы", "они",
    "его", "её", "их", "мне", "тебе", "все", "свой", "своя", "своё",
    "тут", "там", "здесь", "тогда", "когда", "чтобы", "потому", "просто",
    "ещё", "еще", "был", "была", "были", "быть", "есть", "нет", "бы",
})


def _extract_chat_phrases(evidence: List[Dict], msg_index: Dict) -> List[str]:
    """
    Извлекает характерные слова и короткие фразы из цитат переписки.
    Отбирает слова, которые не входят в стоп-список и длиннее 4 символов.
    """
    import re
    word_freq: dict = {}

    for ev in evidence:
        for mid_str in ev.get("message_ids", []):
            row = msg_index.get(str(mid_str), {})
            if not row:
                continue
            text = str(row.get("text", "")).lower().strip()
            if not text:
                continue
            words = re.findall(r"\b[а-яёa-z][а-яёa-z]{3,}\b", text)
            for w in words:
                if w not in _STOP_WORDS:
                    word_freq[w] = word_freq.get(w, 0) + 1

    # Уникальные слова, встречающиеся 1-2 раза (специфические, не частотные)
    candidates = [w for w, c in word_freq.items() if c <= 3]
    # Приоритет словам с заглавной буквы (имена, названия) — ищем в оригинале
    result = []
    for ev in evidence:
        for mid_str in ev.get("message_ids", []):
            row = msg_index.get(str(mid_str), {})
            if not row:
                continue
            text = str(row.get("text", ""))
            # Имена собственные и названия (заглавная буква посреди предложения)
            names = re.findall(r"(?<!\. )(?<!\n)[А-ЯA-Z][а-яёa-z]{2,}", text)
            result.extend(n for n in names if n.lower() not in _STOP_WORDS)
    # Добавляем характерные слова из переписки
    result.extend(candidates[:10])
    # Уникализируем, ограничиваем
    seen = set()
    unique = []
    for w in result:
        lw = w.lower()
        if lw not in seen:
            seen.add(lw)
            unique.append(w)
    return unique[:12]


def _extract_episodes(
    evidence:  List[Dict],
    msg_index: Dict,
    df_sorted: Optional[pd.DataFrame] = None,
    n_context: int = 2,
    max_episodes: int = 6,
) -> List[str]:
    """
    Строит диалоговые эпизоды вокруг каждой улики.

    Каждый эпизод — это несколько строк:
      несколько сообщений до ключевого
    → ключевое сообщение (помечено стрелкой)
      несколько сообщений после

    Если df_sorted не передан — возвращает только изолированные цитаты.
    """
    def _raw_mid(mid_str: str) -> Optional[int]:
        """Извлекает числовой msg_id из строки вида 'chat_id:msg_N' или 'msg_N'."""
        try:
            return int(str(mid_str).split(":")[-1].replace("msg_", ""))
        except (ValueError, AttributeError):
            return None

    # Строим позиционный индекс для поиска соседей (по голому int из df)
    id_to_pos: Dict = {}
    if df_sorted is not None:
        id_to_pos = {int(row["msg_id"]): idx for idx, row in df_sorted.iterrows()}

    episodes = []
    seen_ids: set = set()

    for ev in evidence:
        msg_ids = ev.get("message_ids", [])
        if not msg_ids:
            continue

        # Центральное сообщение эпизода — первое в списке улики
        main_mid_str = msg_ids[0]
        if main_mid_str in seen_ids:
            continue
        seen_ids.add(main_mid_str)

        main_mid = _raw_mid(main_mid_str)
        if main_mid is None:
            continue

        lines = []

        if df_sorted is not None and main_mid in id_to_pos:
            pos = id_to_pos[main_mid]

            # Контекст до
            for _, r in df_sorted.iloc[max(0, pos - n_context): pos].iterrows():
                t = str(r.get("text", ""))[:150]
                if t:
                    lines.append(f"  {r.get('sender','')}: «{t}»")

            # Все сообщения самой улики (ключевые, с →)
            for mid_str in msg_ids:
                row = msg_index.get(str(mid_str), {})
                text = str(row.get("text", "")).strip()
                if text:
                    lines.append(f"→ {row.get('sender','')}: «{text[:200]}»")

            # Контекст после
            for _, r in df_sorted.iloc[pos + 1: pos + 1 + n_context].iterrows():
                t = str(r.get("text", ""))[:150]
                if t:
                    lines.append(f"  {r.get('sender','')}: «{t}»")

        else:
            # Fallback без df — изолированные цитаты
            for mid_str in msg_ids:
                row = msg_index.get(str(mid_str), {})
                text = str(row.get("text", "")).strip()
                if text:
                    lines.append(f"→ {row.get('sender','')}: «{text[:200]}»")

        if lines:
            episodes.append("\n".join(lines))

        if len(episodes) >= max_episodes:
            break

    return episodes
"""
telegram_parser.py
==================
Парсер экспорта чата Telegram (формат result.json).

Два этапа:
  1. parse_messages()  → DataFrame, одна строка = одно сообщение
  2. build_dialogues() → DataFrame, одна строка = один диалог

Стратегия сегментации диалогов (гибридная):
  - gap > hard_break_hours (по умолчанию 24ч)  → всегда новый диалог
  - gap > soft_break_hours (по умолчанию 4ч)   → семантическая проверка:
        • если предыдущий диалог заканчивается вопросом → продолжаем
        • если следующее сообщение является reply на что-то из предыдущего → продолжаем
        • иначе → новый диалог
  - gap ≤ soft_break_hours                     → тот же диалог

Использование:
    parser = TelegramChatParser("result.json")
    parser.load()
    messages_df  = parser.parse_messages()
    dialogues_df = parser.build_dialogues(soft_break_hours=4, hard_break_hours=24)
"""

import json
import re
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def extract_text(text_field: Any) -> str:
    """
    Извлекает плоский текст из поля text.
    Telegram хранит text либо как строку, либо как список
    (строки + объекты с форматированием).
    """
    if isinstance(text_field, str):
        return text_field
    if isinstance(text_field, list):
        parts = []
        for part in text_field:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text", ""))
        return "".join(parts)
    return ""


def detect_content_type(msg: Dict) -> str:
    """
    Определяет основной тип контента сообщения.
    Порядок проверки: явное поле media_type → фото → текст → unknown.
    """
    if msg.get("media_type"):
        return msg["media_type"]          # voice_message, video_message, sticker …
    if msg.get("photo"):
        return "photo"
    text = extract_text(msg.get("text", ""))
    if text.strip():
        return "text"
    return "unknown"


def has_question(text: str) -> bool:
    """Проверяет, заканчивается ли текст вопросом."""
    return bool(text.strip()) and bool(re.search(r"\?[\s]*$", text.strip()))


def looks_like_response(text: str) -> bool:
    """
    Эвристика: начинается ли текст со слов, типичных для ответа на вопрос.
    """
    starters_ru = [
        "да", "нет", "ну", "окей", "ок", "конечно", "наверное",
        "думаю", "скорее", "возможно", "не знаю", "хз",
    ]
    starters_en = ["yes", "no", "yeah", "yep", "nope", "sure", "definitely", "maybe"]
    first_word = text.strip().lower().split()[0] if text.strip() else ""
    return first_word in starters_ru + starters_en


# ---------------------------------------------------------------------------
# Основной класс
# ---------------------------------------------------------------------------

class TelegramChatParser:
    """
    Парсер экспорта личного или группового чата Telegram.

    Параметры
    ----------
    filepath : str
        Путь к файлу result.json.
    target_user_id : str, optional
        ID пользователя, для которого ищем «точки опоры».
        Например: "user123456789".
        Если задан, в DataFrame появится колонка is_target_user.
    """

    def __init__(self, filepath: str, target_user_id: Optional[str] = None):
        self.filepath = filepath
        self.target_user_id = target_user_id

        self._raw: Dict = {}
        self.chat_name: str = ""
        self.chat_type: str = ""
        self.chat_id: int = 0

        self.messages_df: Optional[pd.DataFrame] = None
        self.dialogues_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Загрузка файла
    # ------------------------------------------------------------------

    def load(self) -> "TelegramChatParser":
        """Читает JSON-файл в память. Возвращает self для chaining."""
        with open(self.filepath, "r", encoding="utf-8") as f:
            self._raw = json.load(f)

        self.chat_name = self._raw.get("name", "")
        self.chat_type = self._raw.get("type", "")
        self.chat_id   = self._raw.get("id", 0)
        return self

    # ------------------------------------------------------------------
    # Этап 1: DataFrame по сообщениям
    # ------------------------------------------------------------------

    def parse_messages(self) -> pd.DataFrame:
        """
        Разбирает все сообщения в плоский DataFrame.

        Колонки
        -------
        msg_id            : int      – уникальный ID сообщения
        msg_type          : str      – 'message' | 'service'
        date              : datetime – дата и время
        date_unixtime     : int      – unix-timestamp
        sender            : str      – имя отправителя
        sender_id         : str      – user-ID отправителя
        is_target_user    : bool     – отправлено ли целевым пользователем (если задан)
        text              : str      – плоский текст сообщения
        text_length       : int      – длина текста в символах
        has_text          : bool     – есть ли непустой текст
        content_type      : str      – text | photo | voice_message | video_message |
                                       video_file | audio_file | sticker | animation | unknown
        is_forwarded      : bool     – пересланное сообщение
        forwarded_from    : str      – откуда переслано
        reply_to_msg_id   : int|None – ID сообщения, на которое отвечают
        service_action    : str      – действие сервисного сообщения (phone_call, …)
        duration_seconds  : int|None – длительность (для аудио/видео)
        file_name         : str      – имя файла, если есть
        mime_type         : str      – MIME-тип файла, если есть
        has_question      : bool     – содержит ли сообщение вопрос
        """
        rows = []

        for msg in self._raw.get("messages", []):
            text         = extract_text(msg.get("text", ""))
            content_type = detect_content_type(msg)

            # Отправитель: в сервисных сообщениях поля называются actor/actor_id
            sender    = msg.get("from") or msg.get("actor", "")
            sender_id = msg.get("from_id") or msg.get("actor_id", "")

            row: Dict[str, Any] = {
                # Идентификация
                "msg_id":           msg.get("id"),
                "msg_type":         msg.get("type", ""),
                "date":             pd.to_datetime(msg.get("date")),
                "date_unixtime":    int(msg.get("date_unixtime", 0)),

                # Отправитель
                "sender":           sender,
                "sender_id":        str(sender_id),

                # Текст и тип контента
                "text":             text,
                "text_length":      len(text),
                "has_text":         bool(text.strip()),
                "content_type":     content_type,
                "has_question":     has_question(text),

                # Пересылка
                "is_forwarded":     "forwarded_from" in msg,
                "forwarded_from":   msg.get("forwarded_from", ""),

                # Ответ на сообщение
                "reply_to_msg_id":  msg.get("reply_to_message_id"),

                # Сервисные сообщения
                "service_action":          msg.get("action", ""),
                "service_discard_reason":  msg.get("discard_reason", ""),

                # Медиа
                "duration_seconds": msg.get("duration_seconds"),
                "file_name":        msg.get("file_name", ""),
                "mime_type":        msg.get("mime_type", ""),
            }

            if self.target_user_id is not None:
                row["is_target_user"] = (str(sender_id) == str(self.target_user_id))

            rows.append(row)

        self.messages_df = pd.DataFrame(rows)
        return self.messages_df

    # ------------------------------------------------------------------
    # Этап 2: DataFrame по диалогам
    # ------------------------------------------------------------------

    def build_dialogues(
        self,
        soft_break_hours: float = 4.0,
        hard_break_hours: float = 24.0,
    ) -> pd.DataFrame:
        """
        Собирает диалоги из сообщений.

        Алгоритм сегментации
        --------------------
        1. Пауза > hard_break_hours  → всегда новый диалог.
        2. Пауза > soft_break_hours  → семантическая проверка:
               - текущий диалог заканчивается вопросом (?)  → продолжаем
               - следующее сообщение — reply на сообщение   → продолжаем
                 из текущего диалога
               - иначе → новый диалог.
        3. Пауза ≤ soft_break_hours  → тот же диалог.

        Параметры
        ----------
        soft_break_hours : float
            Порог «мягкого» разрыва (часы). По умолчанию 4.
        hard_break_hours : float
            Порог «жёсткого» разрыва (часы). По умолчанию 24.

        Колонки результата
        ------------------
        dialogue_id          : int      – порядковый номер диалога
        start_date           : datetime – начало
        end_date             : datetime – конец
        duration_minutes     : float    – длительность диалога в минутах
        num_messages         : int      – всего сообщений
        num_text_messages    : int      – сообщений с текстом
        participants         : list     – список участников
        sender_counts        : dict     – {имя: кол-во сообщений}
        content_type_counts  : dict     – {тип: кол-во}
        conversation_text    : str      – отформатированный текст диалога
        messages             : list     – список dict всех сообщений диалога
        ends_with_question   : bool     – заканчивается ли диалог вопросом
        has_reply_chain      : bool     – есть ли ответы (reply) внутри диалога
        target_msg_count     : int|None – сообщений от целевого пользователя
        """
        if self.messages_df is None:
            self.parse_messages()

        # Работаем только с обычными сообщениями (не сервисными)
        df = (
            self.messages_df[self.messages_df["msg_type"] == "message"]
            .copy()
            .sort_values("date")
            .reset_index(drop=True)
        )

        if df.empty:
            self.dialogues_df = pd.DataFrame()
            return self.dialogues_df

        # Вычисляем паузы между сообщениями
        df["gap_hours"] = df["date"].diff().dt.total_seconds().div(3600).fillna(0.0)

        # Множество ID сообщений для reply-проверки
        # reply_to_msg_id → принадлежит ли это сообщение текущему диалогу
        current_dialogue_msg_ids: set = set()

        dialogue_id = 0
        dialogue_ids: List[int] = []

        records = df.to_dict("records")

        for i, rec in enumerate(records):
            gap = rec["gap_hours"]

            if i == 0:
                # Первое сообщение — всегда начало нового диалога
                current_dialogue_msg_ids = {rec["msg_id"]}
                dialogue_ids.append(dialogue_id)
                continue

            if gap > hard_break_hours:
                # Жёсткий разрыв — всегда новый диалог
                dialogue_id += 1
                current_dialogue_msg_ids = set()

            elif gap > soft_break_hours:
                # Мягкий разрыв — семантическая проверка
                should_split = True

                # Проверка 1: текущий диалог заканчивается вопросом
                last_text_msgs = [
                    records[j]
                    for j in range(i)
                    if dialogue_ids[j] == dialogue_id and records[j]["has_text"]
                ]
                if last_text_msgs and has_question(last_text_msgs[-1]["text"]):
                    should_split = False

                # Проверка 2: это сообщение является reply на что-то в текущем диалоге
                if should_split:
                    reply_id = rec.get("reply_to_msg_id")
                    if reply_id and reply_id in current_dialogue_msg_ids:
                        should_split = False

                # Проверка 3: выглядит как ответ на вопрос
                if should_split and rec["has_text"] and looks_like_response(rec["text"]):
                    if last_text_msgs and has_question(last_text_msgs[-1]["text"]):
                        should_split = False

                if should_split:
                    dialogue_id += 1
                    current_dialogue_msg_ids = set()

            # Добавляем ID текущего сообщения к множеству диалога
            current_dialogue_msg_ids.add(rec["msg_id"])
            dialogue_ids.append(dialogue_id)

        df["dialogue_id"] = dialogue_ids

        # ------------------------------------------------------------------
        # Агрегация в диалоги
        # ------------------------------------------------------------------
        dialogue_rows = []

        for did, group in df.groupby("dialogue_id", sort=True):
            group = group.sort_values("date")
            text_msgs = group[group["has_text"]]

            # Форматированный текст диалога
            conv_lines = []
            for _, row in group.iterrows():
                if row["has_text"]:
                    conv_lines.append(
                        f"[{row['date'].strftime('%Y-%m-%d %H:%M')}] {row['sender']}: {row['text']}"
                    )
                elif row["content_type"] != "unknown":
                    conv_lines.append(
                        f"[{row['date'].strftime('%Y-%m-%d %H:%M')}] {row['sender']}: "
                        f"[{row['content_type']}]"
                    )

            # Заканчивается ли диалог вопросом
            last_texts = text_msgs["text"].tolist()
            ends_with_q = has_question(last_texts[-1]) if last_texts else False

            # Есть ли reply-цепочки внутри диалога
            dialogue_msg_ids = set(group["msg_id"].tolist())
            has_replies = group["reply_to_msg_id"].dropna().apply(
                lambda rid: rid in dialogue_msg_ids
            ).any()

            entry: Dict[str, Any] = {
                "dialogue_id":         did,
                "start_date":          group["date"].min(),
                "end_date":            group["date"].max(),
                "duration_minutes":    (
                    group["date"].max() - group["date"].min()
                ).total_seconds() / 60,
                "num_messages":        len(group),
                "num_text_messages":   len(text_msgs),
                "participants":        sorted(group["sender"].unique().tolist()),
                "sender_counts":       group["sender"].value_counts().to_dict(),
                "content_type_counts": group["content_type"].value_counts().to_dict(),
                "conversation_text":   "\n".join(conv_lines),
                "messages":            group.to_dict("records"),
                "ends_with_question":  ends_with_q,
                "has_reply_chain":     bool(has_replies),
            }

            if self.target_user_id is not None and "is_target_user" in group.columns:
                entry["target_msg_count"] = int(group["is_target_user"].sum())
            else:
                entry["target_msg_count"] = None

            dialogue_rows.append(entry)

        self.dialogues_df = pd.DataFrame(dialogue_rows)
        return self.dialogues_df

    # ------------------------------------------------------------------
    # Утилиты
    # ------------------------------------------------------------------

    def info(self) -> None:
        """Печатает краткую сводку по загруженным данным."""
        print(f"Чат:      {self.chat_name}  (тип: {self.chat_type}, id: {self.chat_id})")
        if self.messages_df is not None:
            df = self.messages_df
            print(f"\n── Сообщения ({len(df)} всего) ──")
            print(df["msg_type"].value_counts().to_string())
            print()
            print(df["content_type"].value_counts().to_string())
            senders = df["sender"].value_counts()
            print(f"\nУчастники:")
            for name, cnt in senders.items():
                print(f"  {name}: {cnt} сообщений")
            print(f"\nПериод: {df['date'].min().date()} → {df['date'].max().date()}")

        if self.dialogues_df is not None:
            dd = self.dialogues_df
            print(f"\n── Диалоги ({len(dd)} всего) ──")
            print(f"  Медиана сообщений в диалоге: {dd['num_messages'].median():.0f}")
            print(f"  Медиана длительности (мин):  {dd['duration_minutes'].median():.0f}")
            print(f"  Диалогов с вопросом в конце: {dd['ends_with_question'].sum()}")
            print(f"  Диалогов с reply-цепочками:  {dd['has_reply_chain'].sum()}")

"""
orchestrator.py
===============
Оркестрация пятиступенчатого пайплайна.

Порядок работы:
  1. Загрузка чатов (reuse telegram_parser)
  2. Stage 1: маркировка всех сообщений в окне
  3. Stage 2: детекция по каждому anchor (только релевантные сообщения)
  4. Stage 3: валидация улик и финальная оценка
  5. Сборка anchor_profile_multi.json + читаемого отчёта
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from multi_pipeline.config import PipelineConfig
from multi_pipeline.stage1_labeler import MessageLabeler
from multi_pipeline.stage2_detector import (
    detect_S1_stage2, detect_S2_stage2,
    detect_S5_stage2, detect_D2_stage2,
)
from multi_pipeline.stage3_validator import EvidenceValidator
from multi_pipeline.stage4_narrator import AnchorNarrator
from multi_pipeline.stage5_composer import FullResponseComposer
from multi_pipeline.stage2_p2_detector import (
    detect_S3_stage2, detect_C1_stage2, detect_D3_stage2, detect_E2_stage2,
)
from telegram_parser import TelegramChatParser
from anchor_detection.llm_client import OllamaClient
from anchor_detection.preprocessors import (
    ChatData, _window,
    _history_months, _days_active, _voice_share, _initiation_balance,
)
from anchor_detection.lexicons import POST_SILENCE_PING_PATTERNS

SEP = "=" * 70


def _fmt_t(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"

ANCHOR_NAMES = {
    "S1": "Эмоциональная поддержка и близкие связи",
    "S2": "Принадлежность к сообществу",
    "S5": "«Я нужен» — ответственность за других",
    "D2": "Финансовая безопасность",
    "S3": "Признание и самооценочная поддержка",
    "C1": "Надежда и оптимизм — конкретные планы",
    "D3": "Хобби, увлечения и поток",
    "E2": "Копинг-репертуар и стратегии регуляции",
}


# ══════════════════════════════════════════════════════════════════════
# Загрузка чатов
# ══════════════════════════════════════════════════════════════════════

def _detect_target(raw: Dict, df: pd.DataFrame):
    contact_name = raw.get("name", "")
    senders = df[
        (df["msg_type"] == "message") &
        (df["sender"].notna()) &
        (df["sender"] != "") &
        # Фильтруем по первому слову имени контакта (частичное совпадение)
        (~df["sender"].apply(
            lambda s: str(s).lower().startswith(contact_name.split()[0].lower())
            if contact_name else False
        ))
    ]
    if senders.empty:
        top = df[df["msg_type"] == "message"]["sender_id"].value_counts()
        tid = top.index[0] if not top.empty else ""
        tname = df[df["sender_id"] == tid]["sender"].iloc[0] if tid else "Unknown"
        return str(tid), str(tname)
    top = senders["sender_id"].value_counts()
    tid   = str(top.index[0])
    tname = str(senders[senders["sender_id"] == tid]["sender"].iloc[0])
    return tid, tname


def load_chats(file_paths: List[str]) -> List[ChatData]:
    chats = []
    for path_str in file_paths:
        path = Path(path_str)
        if not path.exists():
            print(f"[orchestrator] Файл не найден: {path_str}")
            continue
        print(f"[orchestrator] Загружаю {path.name}...")
        parser = TelegramChatParser(str(path))
        parser.load()
        df = parser.parse_messages()
        raw = parser._raw
        chat_id   = str(raw.get("id",   path.stem))
        chat_name = str(raw.get("name", path.stem))
        chat_type = str(raw.get("type", "personal_chat"))
        tid, tname = _detect_target(raw, df)
        print(f"[orchestrator]   Контакт: {chat_name} | Таргет: {tname} ({tid})")
        chats.append(ChatData(
            chat_id=chat_id, chat_name=chat_name, chat_type=chat_type,
            target_id=tid, target_name=tname, df=df,
        ))
    if not chats:
        raise ValueError("Не удалось загрузить ни одного чата.")
    return chats


def build_msg_index(chats: List[ChatData]) -> Dict[str, Dict]:
    """
    Строит индекс 'chat_id:msg_N' → row из всех чатов.
    Составной ключ исключает коллизии: Telegram нумерует сообщения независимо в каждом чате.
    """
    idx: Dict[str, Dict] = {}
    for chat in chats:
        prefix = f"{chat.chat_id}:"
        for row in chat.df.to_dict("records"):
            key = f"{prefix}msg_{int(row['msg_id'])}"
            idx[key] = row
    return idx


def _prefix_candidates(msgs: List[Dict], prefix: str) -> None:
    """Добавляет chat_id-префикс к msg_id всех кандидатов и их контекстных сообщений (in-place)."""
    for m in msgs:
        mid = str(m.get("msg_id", ""))
        if not mid.startswith(prefix):
            m["msg_id"] = f"{prefix}{mid}"
        for ctx in m.get("context_before", []):
            cmid = str(ctx.get("msg_id", ""))
            if not cmid.startswith(prefix):
                ctx["msg_id"] = f"{prefix}{cmid}"
        for ctx in m.get("context_after", []):
            cmid = str(ctx.get("msg_id", ""))
            if not cmid.startswith(prefix):
                ctx["msg_id"] = f"{prefix}{cmid}"


# ══════════════════════════════════════════════════════════════════════
# Агрегаты для детекторов Stage 2
# ══════════════════════════════════════════════════════════════════════

def _merge_aggregates(agg: Dict) -> Dict:
    """Объединяет агрегаты нескольких чатов в один dict для Stage 2."""
    if not agg:
        return {}
    values = list(agg.values())
    if len(values) == 1:
        return values[0]
    return {
        "chat_name":           " + ".join(v["chat_name"] for v in values),
        "chat_type":           values[0]["chat_type"],
        "history_months":      max(v["history_months"] for v in values),
        "msg_count_window":    sum(v["msg_count_window"] for v in values),
        "days_active_window":  max(v["days_active_window"] for v in values),
        "window_days":         values[0]["window_days"],
        "voice_share_target":  sum(v["voice_share_target"] for v in values) / len(values),
        "initiation_balance":  sum(v["initiation_balance"] for v in values) / len(values),
    }


def _build_aggregates(chats: List[ChatData], window_days: Optional[int]) -> Dict:
    """Базовые агрегаты (статистика) для всех детекторов Stage 2."""
    result = {}
    for chat in chats:
        df_full = chat.df
        df_w    = _window(df_full, window_days)
        tid     = chat.target_id

        result[chat.chat_id] = {
            "chat_name":           chat.chat_name,
            "chat_type":           chat.chat_type,
            "history_months":      _history_months(df_full),
            "msg_count_window":    len(df_w[df_w["msg_type"] == "message"]),
            "days_active_window":  _days_active(df_w),
            "window_days":         window_days or "all",
            "voice_share_target":  _voice_share(df_w, tid),
            "initiation_balance":  _initiation_balance(df_w, tid),
        }
    return result


# ══════════════════════════════════════════════════════════════════════
# Сбор пингов после молчания (для S5)
# ══════════════════════════════════════════════════════════════════════

def _collect_pings(chats: List[ChatData], window_days: Optional[int]) -> List[Dict]:
    pings = []
    for chat in chats:
        df = _window(chat.df, window_days)
        tid = chat.target_id
        df_sorted = df[df["msg_type"] == "message"].sort_values("date")
        prev_sender, prev_ts = None, None
        for _, row in df_sorted.iterrows():
            if row["sender_id"] == tid:
                prev_sender, prev_ts = tid, row["date"]
                continue
            if prev_sender == tid and prev_ts is not None and row["has_text"]:
                silence_h = (row["date"] - prev_ts).total_seconds() / 3600
                t_low = row["text"].lower()
                if any(p in t_low for p in POST_SILENCE_PING_PATTERNS) and silence_h >= 2:
                    pings.append({
                        "msg_id":        f"msg_{row['msg_id']}",
                        "chat_id":       chat.chat_id,
                        "silence_hours": round(silence_h, 1),
                        "sender":        row["sender"],
                        "text":          row["text"][:200],
                        "ts":            str(row["date"])[:16],
                    })
            prev_sender, prev_ts = row["sender_id"], row["date"]
    return pings


# ══════════════════════════════════════════════════════════════════════
# Финальный профиль
# ══════════════════════════════════════════════════════════════════════

def _build_profile(
    target_name: str,
    chats:       List[ChatData],
    results:     Dict[str, Any],
    config:      PipelineConfig,
    label_dist:  Dict,
) -> Dict:
    found = [
        code for code, r in results.items()
        if r.get("evidence_found") and r.get("strength", 0) >= 0.4
    ]
    return {
        "meta": {
            "target":          target_name,
            "analysis_date":   datetime.now().strftime("%Y-%m-%d %H:%M"),
            "chats_analyzed":  len(chats),
            "chat_names":      [c.chat_name for c in chats],
            "pipeline":        "multi_stage_p1p2",
            "labeler_model":   config.labeler_model,
            "detector_model":  config.detector_model,
            "validator_model": config.validator_model,
            "narrator_model":  config.narrator_model,
            "composer_model":  config.composer_model,
            "label_distribution": label_dist,
        },
        "summary": {
            "anchors_found":     found,
            "anchors_not_found": [c for c in results if c not in found],
            "strongest_anchor":  max(results, key=lambda c: results[c].get("strength", 0))
                                 if results else None,
        },
        "detectors": results,
    }


def _print_result(code: str, result: Dict) -> None:
    found    = result.get("evidence_found", False)
    strength = result.get("strength", 0.0)
    suff     = result.get("data_sufficiency", "?")
    window   = result.get("window", "?")
    icon     = "✅" if found else "❌"
    accepted = sum(1 for ev in result.get("evidence", [])
                   if ev.get("verdict") == "accepted")
    total    = len(result.get("evidence", []))
    print(f"  {icon} {code}: found={found}, strength={strength:.2f}, "
          f"data={suff}, window={window}, uliki={accepted}/{total}")


def _build_text_report(
    profile: Dict,
    labels: Dict[str, List],
    msg_idx: Optional[Dict] = None,
) -> str:
    lines = [SEP, "  MULTI-STAGE PIPELINE — ПРОФИЛЬ ТОЧЕК ОПОРЫ", SEP]

    meta = profile["meta"]
    lines.append(f"\nТаргет:  {meta['target']}")
    lines.append(f"Дата:    {meta['analysis_date']}")
    lines.append(f"Чаты:    {', '.join(meta.get('chat_names', []))}")
    lines.append(f"Модели:  Labeler={meta['labeler_model']} | "
                 f"Detector={meta['detector_model']} | Validator={meta['validator_model']}")

    # Кандидаты Stage 1
    lines.append("\n" + "─" * 70)
    lines.append("  КАНДИДАТЫ STAGE 1 (после фильтрации)")
    lines.append("─" * 70)
    for code, msgs in labels.items():
        name = ANCHOR_NAMES.get(code, code)
        bar  = "█" * min(len(msgs) // 2, 25)
        lines.append(f"  {code:<4} {name:<48} {bar} {len(msgs)}")

    # Сводка
    summary = profile["summary"]
    found_names = [
        f"{c} — {ANCHOR_NAMES.get(c, c)}" for c in summary["anchors_found"]
    ]
    strongest = summary.get("strongest_anchor", "")
    strongest_str = (
        f"{strongest} — {ANCHOR_NAMES.get(strongest, strongest)}" if strongest else "нет"
    )
    lines.append(f"\n{'═' * 70}")
    lines.append("  ИТОГ")
    lines.append("═" * 70)
    if found_names:
        lines.append(f"\n  НАЙДЕНЫ ({len(found_names)}):")
        for n in found_names:
            lines.append(f"    ✅ {n}")
    else:
        lines.append("  НАЙДЕНЫ: нет")
    not_found = summary["anchors_not_found"]
    lines.append(f"\n  НЕ НАЙДЕНЫ ({len(not_found)}):")
    for c in not_found:
        lines.append(f"    ❌ {c} — {ANCHOR_NAMES.get(c, c)}")
    lines.append(f"\n  СИЛЬНЕЙШАЯ: {strongest_str}")

    # Детали по каждому детектору
    lines.append(f"\n{'═' * 70}")
    lines.append("  ДЕТАЛЬНЫЙ АНАЛИЗ ПО КАЖДОЙ ТОЧКЕ ОПОРЫ")
    lines.append("═" * 70)

    for code, det in profile["detectors"].items():
        name     = ANCHOR_NAMES.get(code, code)
        found    = det.get("evidence_found", False)
        strength = det.get("strength", 0.0)
        suff     = det.get("data_sufficiency", "?")
        icon     = "✅" if found else "❌"

        lines.append(f"\n{'─' * 70}")
        lines.append(f"  {icon} {code} — {name}")
        lines.append(f"     Уверенность модели: {strength:.0%}  |  Достаточность данных: {suff}")

        subscores = det.get("subscores", {})
        if subscores:
            lines.append("     Подоценки:")
            for k, v in subscores.items():
                bar = "█" * int(float(v) * 10) + "░" * (10 - int(float(v) * 10))
                lines.append(f"       {k:<42} {bar} {float(v):.2f}")

        accepted_ev = [ev for ev in det.get("evidence", []) if ev.get("verdict") == "accepted"]
        rejected_ev = [ev for ev in det.get("evidence", []) if ev.get("verdict") == "rejected"]

        if accepted_ev:
            lines.append(f"\n     ✓ Принятые улики ({len(accepted_ev)}):")
            for i, ev in enumerate(accepted_ev, 1):
                lines.append(f"\n       [{i}] {ev.get('why', '')}")
                reason = ev.get("verdict_reason", "")
                if reason:
                    lines.append(f"            Вердикт: {reason}")
                for mid_str in ev.get("message_ids", []):
                    if msg_idx is not None:
                        try:
                            row    = msg_idx.get(str(mid_str), {})
                            text   = str(row.get("text", ""))[:200]
                            sender = row.get("sender", "?")
                            ts     = str(row.get("date", ""))[:16]
                            if text:
                                lines.append(f"            → [{ts}] {sender}: «{text}»")
                                continue
                        except Exception:
                            pass
                    lines.append(f"            → {mid_str}")

        if rejected_ev:
            lines.append(f"\n     ✗ Отклонённые улики ({len(rejected_ev)}):")
            for ev in rejected_ev[:3]:
                lines.append(f"         — {ev.get('verdict_reason', '')[:120]}")

        anti = det.get("anti_signals", [])
        if anti:
            lines.append(f"\n     Анти-сигналы: {', '.join(str(a) for a in anti)}")

        caveat = det.get("caveat", "")
        if caveat:
            lines.append(f"     Примечание: {caveat}")

    # Примеры сообщений для найденных точек опоры
    found_codes = set(summary["anchors_found"])
    if found_codes:
        lines.append(f"\n{'═' * 70}")
        lines.append("  ПРИМЕРЫ СООБЩЕНИЙ (по результатам Stage 1)")
        lines.append("═" * 70)
        for code in sorted(found_codes):
            name = ANCHOR_NAMES.get(code, code)
            msgs = labels.get(code, [])
            if not msgs:
                continue
            lines.append(f"\n  {code} — {name}:")
            for msg in msgs[:4]:
                ts     = msg.get("ts", "")[:10]
                sender = msg.get("sender", "")
                text   = msg.get("text", "")[:160]
                role   = msg.get("role", "")
                lines.append(f"    [{ts}] {sender} ({role}):")
                lines.append(f"      «{text}»")

    # Нарративы
    narratives = profile.get("narratives", {})
    if narratives:
        lines.append(f"\n{'═' * 70}")
        lines.append("  НАРРАТИВНЫЕ ОПИСАНИЯ ТОЧЕК ОПОРЫ")
        lines.append("═" * 70)
        for code, narrative in narratives.items():
            name = ANCHOR_NAMES.get(code, code)
            lines.append(f"\n  {code} — {name}")
            lines.append("  " + "─" * 60)
            for para in narrative.split("\n"):
                if para.strip():
                    lines.append(f"  {para}")

    # Полный поддерживающий текст
    full = profile.get("full_response", "")
    if full:
        lines.append(f"\n{'═' * 70}")
        lines.append("  ПОЛНЫЙ ПОДДЕРЖИВАЮЩИЙ ТЕКСТ")
        lines.append("═" * 70)
        lines.append("")
        for para in full.split("\n"):
            lines.append(para)

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# Главный класс
# ══════════════════════════════════════════════════════════════════════

class MultiStagePipeline:
    """
    Трёхступенчатый пайплайн поиска точек опоры.

    Параметры
    ----------
    config : PipelineConfig — конфигурация моделей и окна
    verbose : bool — подробный вывод
    """

    def __init__(self, config: Optional[PipelineConfig] = None, verbose: bool = True):
        self.config  = config or PipelineConfig()
        self.verbose = verbose

        self.labeler   = MessageLabeler(self.config, verbose=verbose)
        self.validator = EvidenceValidator(self.config)

        self._detector_client = OllamaClient(
            model    = self.config.detector_model,
            base_url = self.config.ollama_url,
            num_ctx  = self.config.num_ctx_detector,
            timeout  = 360,
        )

    # ------------------------------------------------------------------
    # Главный метод
    # ------------------------------------------------------------------

    def run(
        self,
        file_paths:  List[str],
        output_path: Optional[str] = None,
        report_path: Optional[str] = None,
    ) -> Dict:
        """
        Запускает полный трёхступенчатый пайплайн.

        Параметры
        ----------
        file_paths  : пути к result.json
        output_path : куда сохранить JSON-профиль
        report_path : куда сохранить текстовый отчёт
        """
        client = self._detector_client
        if not OllamaClient(base_url=self.config.ollama_url).is_available():
            raise RuntimeError("Ollama недоступна. Запусти ollama serve.")

        t_pipeline = time.time()

        # ── 1. Загрузка ──────────────────────────────────────────────
        print(f"\n{SEP}\n  Шаг 1: Загрузка чатов\n{SEP}")
        t1 = time.time()
        chats    = load_chats(file_paths)
        msg_idx  = build_msg_index(chats)
        target   = chats[0].target_name
        agg      = _build_aggregates(chats, self.config.window_days)
        print(f"  [Шаг 1] готово за {_fmt_t(time.time() - t1)}")

        # ── 2. Stage 1: предфильтрация и LLM-валидация кандидатов ───
        print(f"\n{SEP}\n  Шаг 2: Stage 1 — предфильтрация кандидатов\n{SEP}")
        t2 = time.time()
        # all_labels теперь Dict[anchor_code → List[Dict]] (новый API)
        all_labels: Dict[str, List[Dict]] = {}
        for chat in chats:
            chat_labeled = self.labeler.label_for_anchors(
                chat.df,
                target_id   = chat.target_id,
                window_days = self.config.window_days,
            )
            prefix = f"{chat.chat_id}:"
            for code, msgs in chat_labeled.items():
                _prefix_candidates(msgs, prefix)
                all_labels.setdefault(code, []).extend(msgs)

        label_dist = {code: len(msgs) for code, msgs in all_labels.items()}
        if self.verbose:
            print(f"\n  Stage 1 результат:")
            for code, msgs in all_labels.items():
                print(f"    {code}: {len(msgs)} сообщений прошли фильтр")
        print(f"  [Шаг 2] Stage 1 P1 готово за {_fmt_t(time.time() - t2)}")

        # ── 3. Stage 2: детекция ─────────────────────────────────────
        print(f"\n{SEP}\n  Шаг 3: Stage 2 — детекция по точкам опоры\n{SEP}")

        t3 = time.time()
        candidates: Dict[str, Dict] = {}
        agg_flat = _merge_aggregates(agg)
        pings = _collect_pings(chats, self.config.window_days)

        # Стриппаем context_before/context_after перед Stage 2 — они нужны только Stage 1.
        _CTX_KEYS = ("context_before", "context_after")
        def _slim(msgs: List[Dict]) -> List[Dict]:
            return [{k: v for k, v in m.items() if k not in _CTX_KEYS} for m in msgs]

        def _by_chat(msgs: List[Dict]) -> Dict[str, List[Dict]]:
            """Разбивает плоский список кандидатов по chat_id (из префикса msg_id)."""
            result: Dict[str, List[Dict]] = {}
            for m in msgs:
                mid = str(m.get("msg_id", ""))
                cid = mid.split(":")[0] if ":" in mid else "unknown"
                result.setdefault(cid, []).append(m)
            return result

        def _best(results: List[Dict]) -> Dict:
            """Из нескольких результатов детектора берёт наилучший (по strength)."""
            return max(results, key=lambda r: r.get("strength", 0)) if results else {}

        # labeled_msgs: стрипнутые кандидаты без context
        labeled_msgs = {code: _slim(msgs) for code, msgs in all_labels.items()}

        # Per-chat детекция P1: каждый чат анализируется отдельно, берём лучший результат
        print("\n  [S1] Эмоциональная поддержка...")
        s1_results = []
        for cid, chat_msgs in _by_chat(labeled_msgs.get("S1", [])).items():
            t_emo = [m for m in chat_msgs if m.get("role") == "target_emotional"]
            c_emp = [m for m in chat_msgs if m.get("role") == "contact_empathy"]
            r = detect_S1_stage2(t_emo, c_emp, agg_flat, client)
            print(f"    [{cid}] ", end=""); _print_result("S1", r)
            s1_results.append(r)
        candidates["S1"] = _best(s1_results)

        print("  [S2] Принадлежность к сообществу...")
        s2_results = []
        for cid, chat_msgs in _by_chat(labeled_msgs.get("S2", [])).items():
            r = detect_S2_stage2(chat_msgs, agg_flat, client)
            print(f"    [{cid}] ", end=""); _print_result("S2", r)
            s2_results.append(r)
        candidates["S2"] = _best(s2_results)

        print("  [S5] «Я нужен»...")
        s5_results = []
        for cid, chat_msgs in _by_chat(labeled_msgs.get("S5", [])).items():
            r = detect_S5_stage2(chat_msgs, pings[:15], agg_flat, client)
            print(f"    [{cid}] ", end=""); _print_result("S5", r)
            s5_results.append(r)
        candidates["S5"] = _best(s5_results)

        print("  [D2] Финансовая безопасность...")
        d2_results = []
        for cid, chat_msgs in _by_chat(labeled_msgs.get("D2", [])).items():
            r = detect_D2_stage2(chat_msgs, agg_flat, client)
            print(f"    [{cid}] ", end=""); _print_result("D2", r)
            d2_results.append(r)
        candidates["D2"] = _best(d2_results)

        print(f"  [Шаг 3] Stage 2 P1 готово за {_fmt_t(time.time() - t3)}")

        # ── 4. Stage 3: валидация ─────────────────────────────────────
        print(f"\n{SEP}\n  Шаг 4: Stage 3 — валидация улик\n{SEP}")
        t4 = time.time()
        results: Dict[str, Any] = {}

        for code, candidate in candidates.items():
            print(f"\n  [{code}] Валидирую...")
            window_str = f"{self.config.window_days}d"
            result = self.validator.validate(candidate, msg_idx, window=window_str)
            results[code] = result
            _print_result(code, result)
        print(f"  [Шаг 4] Stage 3 P1 готово за {_fmt_t(time.time() - t4)}")

        # ── 4б. P2 — точки опоры второго приоритета ──────────────────
        print(f"\n{SEP}\n  Шаг 4б: P2 — Признание, Оптимизм, Хобби, Копинг\n{SEP}")

        # Клиент для E2 с длинным контекстом
        e2_client = OllamaClient(
            model    = self.config.e2_model,
            base_url = self.config.ollama_url,
            num_ctx  = self.config.num_ctx_e2,
            timeout  = 300,
        )
        # Клиент для S3, C1, D3 — обычный детектор
        p2_client = self._detector_client

        # Stage 1 для P2
        t_p2 = time.time()
        p2_labels: Dict[str, List[Dict]] = {}
        for chat in chats:
            chat_p2 = self.labeler.label_for_p2_anchors(
                chat.df,
                target_id   = chat.target_id,
                window_days = self.config.window_days,
            )
            prefix = f"{chat.chat_id}:"
            for code, msgs in chat_p2.items():
                if code != "E2":  # E2 — эпизоды, не flat кандидаты
                    _prefix_candidates(msgs, prefix)
                p2_labels.setdefault(code, []).extend(msgs)

        if self.verbose:
            for code, items in p2_labels.items():
                print(f"    {code}: {len(items)} {'эпизодов' if code == 'E2' else 'кандидатов'}")
        print(f"  [Шаг 4б] Stage 1 P2 готово за {_fmt_t(time.time() - t_p2)}")

        # Stage 2 для P2 — стриппаем context перед передачей
        p2_slim = {code: (_slim(msgs) if code != "E2" else msgs)
                   for code, msgs in p2_labels.items()}

        t_p2s2 = time.time()
        print("\n  [S3] Признание и похвала...")
        results["S3"] = detect_S3_stage2(p2_slim.get("S3", []), agg_flat, p2_client)
        _print_result("S3", results["S3"])

        print("  [C1] Надежда и оптимизм...")
        results["C1"] = detect_C1_stage2(p2_slim.get("C1", []), agg_flat, p2_client)
        _print_result("C1", results["C1"])

        print("  [D3] Хобби и увлечения...")
        results["D3"] = detect_D3_stage2(p2_slim.get("D3", []), agg_flat, p2_client)
        _print_result("D3", results["D3"])

        print("  [E2] Копинг-репертуар (длинный контекст)...")
        try:
            results["E2"] = detect_E2_stage2(p2_slim.get("E2", []), agg_flat, e2_client)
        except Exception as exc:
            print(f"  [Stage 2 / E2] Критическая ошибка: {exc}")
            results["E2"] = {
                "anchor_code": "E2", "evidence_found": False,
                "candidate_evidence": [], "data_sufficiency": "low",
            }
        _print_result("E2", results["E2"])
        print(f"  [Шаг 4б] Stage 2 P2 готово за {_fmt_t(time.time() - t_p2s2)}")

        # Stage 3 для P2
        t_p2s3 = time.time()
        print("\n  Валидация P2...")
        for code in ["S3", "C1", "D3", "E2"]:
            if results[code].get("evidence_found") and results[code].get("candidate_evidence"):
                print(f"  [{code}] Валидирую...")
                validated = self.validator.validate(
                    results[code], msg_idx, window=f"{self.config.window_days}d"
                )
                results[code] = validated
            # Если нет evidence — оставляем как есть (data_sufficiency=low)
        print(f"  [Шаг 4б] Stage 3 P2 готово за {_fmt_t(time.time() - t_p2s3)}")

        # ── 5. Профиль ────────────────────────────────────────────────
        profile = _build_profile(target, chats, results, self.config, label_dist)

        # ── 6. Stage 4: нарративный текст ─────────────────────────────
        print(f"\n{SEP}\n  Шаг 6: Stage 4 — нарративный текст\n{SEP}")
        t6 = time.time()
        narrator = AnchorNarrator(
            model      = self.config.narrator_model,
            ollama_url = self.config.ollama_url,
            num_ctx    = self.config.num_ctx_narrator,
        )
        combined_df = pd.concat([c.df for c in chats], ignore_index=True)
        narratives = narrator.generate_all(profile, msg_idx, df=combined_df)
        profile["narratives"] = narratives
        print(f"  [Шаг 6] Stage 4 готово за {_fmt_t(time.time() - t6)}")

        # ── 7. Stage 5: полный поддерживающий ответ ──────────────────
        print(f"\n{SEP}\n  Шаг 7: Stage 5 — полный ответ\n{SEP}")
        t7 = time.time()
        composer = FullResponseComposer(
            model      = self.config.composer_model,
            ollama_url = self.config.ollama_url,
            num_ctx    = self.config.num_ctx_composer,
        )
        full_response = composer.compose(profile, narratives, target_name=target)
        profile["full_response"] = full_response
        if full_response:
            print(f"  ✓ Полный ответ сгенерирован ({len(full_response)} символов)")
        print(f"  [Шаг 7] Stage 5 готово за {_fmt_t(time.time() - t7)}")

        # Сохранение JSON
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(
                json.dumps(profile, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            print(f"\n[+] Профиль сохранён: {output_path}")

        # Текстовый отчёт
        report = _build_text_report(profile, all_labels, msg_idx)
        if report_path:
            Path(report_path).write_text(report, encoding="utf-8")
            print(f"[+] Отчёт сохранён: {report_path}")

        print(f"\n{report}")
        print(f"\n{'═' * 70}")
        print(f"  ИТОГО: пайплайн завершён за {_fmt_t(time.time() - t_pipeline)}")
        print(f"{'═' * 70}")
        return profile
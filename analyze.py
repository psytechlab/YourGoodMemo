"""
analyze.py
==========
Аналитика результатов трёхступенчатого пайплайна.

Для каждой точки опоры показывает:
  - Что нашла Stage 1 (сколько кандидатов прошли фильтр)
  - Что нашёл Stage 2 (candidate_evidence)
  - Вердикт Stage 3 (принято / отклонено + причина)
  - Полные цитаты из переписки для каждой улики

Запуск:
    python analyze.py
    python analyze.py --profile data/anchor_profile_multi.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from telegram_parser import TelegramChatParser

SEP  = "=" * 70
SEP2 = "─" * 70

ANCHOR_NAMES = {
    "S1": "Эмоциональная поддержка и близкие связи",
    "S2": "Принадлежность к сообществу",
    "S5": "«Я нужен» / ответственность за других",
    "D2": "Финансовая безопасность",
    "S3": "Признание и самооценочная поддержка",
    "C1": "Надежда и оптимизм — конкретные планы",
    "D3": "Хобби, увлечения и поток",
    "E2": "Копинг-репертуар и стратегии регуляции",
}


def load_profile(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_msg_index(chat_file: str) -> dict:
    parser = TelegramChatParser(chat_file)
    parser.load()
    df = parser.parse_messages()
    return {row["msg_id"]: row for row in df.to_dict("records")}


def fmt_msg(idx: dict, mid_str: str) -> str:
    try:
        mid = int(str(mid_str).replace("msg_", ""))
        row = idx.get(mid, {})
        if not row:
            return f"  → [{mid_str}] сообщение не найдено"
        ts     = str(row.get("date", ""))[:16]
        sender = row.get("sender", "?")
        text   = str(row.get("text", "")).strip()
        ct     = row.get("content_type", "")
        if not text:
            text = f"[{ct}]"
        return f"  → [{mid_str}] {ts} {sender}: {text[:250]}"
    except Exception:
        return f"  → [{mid_str}] ошибка чтения"


def print_anchor(code: str, det: dict, idx: dict, label_dist: dict,
                 narrative: str = "") -> None:
    name    = ANCHOR_NAMES.get(code, code)
    found   = det.get("evidence_found", False)
    strength = det.get("strength", 0.0)
    suff    = det.get("data_sufficiency", "?")
    window  = det.get("window", "?")
    icon    = "✅" if found else "❌"

    print(f"\n{'━' * 70}")
    print(f"  {icon}  {code} — {name}")
    print(f"     strength={strength:.2f} | data={suff} | окно={window}")
    print(f"{'━' * 70}")

    # Сколько кандидатов прошло Stage 1
    n_candidates = label_dist.get(code, 0)
    print(f"\n  [Stage 1] Кандидатов после фильтра: {n_candidates}")

    # Subscores
    subscores = det.get("subscores", {})
    if subscores:
        print("\n  [Stage 3] Subscores:")
        for k, v in subscores.items():
            try:
                fv  = float(v)
                bar = "█" * int(fv * 12) + "░" * (12 - int(fv * 12))
                print(f"    {k:<42} {bar} {fv:.2f}")
            except (TypeError, ValueError):
                print(f"    {k}: {v}")

    # Evidence с вердиктами
    evidence = det.get("evidence", [])
    if evidence:
        print(f"\n  [Stage 3] Улики ({len(evidence)} шт.):")
        for i, ev in enumerate(evidence, 1):
            verdict = ev.get("verdict", "—")
            v_icon  = "✓" if verdict == "accepted" else "✗"
            why     = ev.get("why", "")
            reason  = ev.get("verdict_reason", "")

            print(f"\n    [{v_icon}] Улика {i}: {why}")
            if reason:
                print(f"         Вердикт: {reason}")
            for mid in ev.get("message_ids", []):
                print(fmt_msg(idx, mid))
    else:
        caveat = det.get("caveat", "")
        print(f"\n  Улик нет. {caveat}")

    # Анти-сигналы
    anti = [a for a in det.get("anti_signals", []) if a]
    if anti:
        print(f"\n  Анти-сигналы: {', '.join(str(a) for a in anti)}")

    # Нарративный текст
    if narrative:
        print(f"\n  {'─' * 66}")
        print(f"  ✍  Поддерживающий текст:")
        print(f"\n  {narrative}\n")


def main():
    parser = argparse.ArgumentParser(description="Аналитика multi-stage пайплайна")
    parser.add_argument("--profile", default="data/anchor_profile_multi.json",
                        help="Путь к JSON-профилю")
    parser.add_argument("--chat",    default="data/result.json",
                        help="Путь к result.json (для цитат)")
    args = parser.parse_args()

    if not Path(args.profile).exists():
        print(f"[ошибка] Профиль не найден: {args.profile}")
        print("  Сначала запусти: python run_multi.py data/result.json")
        sys.exit(1)

    profile = load_profile(args.profile)
    idx     = build_msg_index(args.chat)

    meta    = profile["meta"]
    summary = profile["summary"]
    dets    = profile["detectors"]
    dist    = meta.get("label_distribution", {})

    # ── Шапка ────────────────────────────────────────────────────────
    print(SEP)
    print("  АНАЛИТИКА ТРЁХСТУПЕНЧАТОГО ПАЙПЛАЙНА")
    print(SEP)
    print(f"\n  Таргет:    {meta['target']}")
    print(f"  Дата:      {meta['analysis_date']}")
    print(f"  Чаты:      {', '.join(meta['chat_names'])}")
    print(f"  Labeler:   {meta['labeler_model']}")
    print(f"  Detector:  {meta['detector_model']}")
    print(f"  Validator: {meta['validator_model']}")

    # ── Сводка ───────────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print("  СВОДКА")
    print(SEP2)
    found     = summary.get("anchors_found", [])
    not_found = summary.get("anchors_not_found", [])
    strongest = summary.get("strongest_anchor", "—")

    print(f"\n  Найдены:    {', '.join(found) or 'нет'}")
    print(f"  Не найдены: {', '.join(not_found) or 'нет'}")
    print(f"  Сильнейшая: {strongest}")

    print("\n  Кандидаты Stage 1:")
    for code in ["S1", "S2", "S5", "D2"]:
        n   = dist.get(code, 0)
        bar = "█" * min(n * 2, 20)
        print(f"    {code}  {bar:<20} {n}")

    # ── Детальный разбор по anchor ───────────────────────────────────
    print(f"\n{SEP}")
    print("  ДЕТАЛЬНЫЙ РАЗБОР")

    narratives = profile.get("narratives", {})
    for code in ["S1", "S2", "S5", "D2", "S3", "C1", "D3", "E2"]:
        det = dets.get(code, {})
        print_anchor(code, det, idx, dist, narrative=narratives.get(code, ""))

    print(f"\n{SEP}")


if __name__ == "__main__":
    main()
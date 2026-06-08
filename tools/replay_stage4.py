"""
replay_stage4.py — переиграть только Stage 4 (нарративы) без полного перезапуска

Использование (из корня проекта):
  python tools/replay_stage4.py
"""
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from multi_pipeline.orchestrator import build_msg_index, load_chats
from multi_pipeline.stage4_narrator import AnchorNarrator
from multi_pipeline.config import PipelineConfig

CHAT_FILES = [
    "data/mom.json",
    "data/nastya.json",
    "data/result.json",
    "data/tema.json",
]
PROFILE_IN  = "results/profile_all4.json"
PROFILE_OUT = "results/profile_all4.json"
REPORT_OUT  = "results/report_all4.txt"

cfg = PipelineConfig()


def replay():
    print("Загружаем чаты...")
    chats = load_chats(CHAT_FILES)

    print("Строим msg_index...")
    msg_index = build_msg_index(chats)
    print(f"  msg_index: {len(msg_index)} записей")

    print("Загружаем профиль...")
    with open(PROFILE_IN, encoding="utf-8") as f:
        profile = json.load(f)

    narrator = AnchorNarrator(
        model=cfg.narrator_model,
        ollama_url=cfg.ollama_url,
        num_ctx=cfg.num_ctx_narrator,
    )

    print("Генерируем нарративы (Stage 4)...")
    narratives = narrator.generate_all(profile, msg_index)
    profile["narratives"] = narratives
    print(f"  Сгенерировано нарративов: {len(narratives)}")

    for code, text in narratives.items():
        print(f"\n  [{code}]")
        print(text[:300])
        print("  ...")

    with open(PROFILE_OUT, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    print(f"\nПрофиль сохранён: {PROFILE_OUT}")

    # Обновляем только блок нарративов в report
    try:
        with open(REPORT_OUT, encoding="utf-8") as f:
            report = f.read()

        NARR_HEADER = "  НАРРАТИВНЫЕ ОПИСАНИЯ ТОЧЕК ОПОРЫ"
        NARR_FOOTER = "  ПОЛНЫЙ ПОДДЕРЖИВАЮЩИЙ ТЕКСТ"

        start = report.find(NARR_HEADER)
        end   = report.find(NARR_FOOTER)
        if start != -1 and end != -1:
            sep = "  " + "─" * 50
            narr_lines = [
                "=" * 70,
                NARR_HEADER,
                "=" * 70,
            ]
            first = True
            for code, text in narratives.items():
                if not first:
                    narr_lines.append(sep)
                    narr_lines.append("")
                first = False
                for para in text.split("\n"):
                    narr_lines.append("  " + para)
                narr_lines.append("")

            new_narr_block = "\n".join(narr_lines) + "\n"
            report = report[:start] + new_narr_block + report[end:]

            with open(REPORT_OUT, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"Отчёт обновлён: {REPORT_OUT}")
        else:
            print("Блок нарративов в отчёте не найден — обновите вручную.")

    except FileNotFoundError:
        print(f"Отчёт {REPORT_OUT} не найден — пропускаю обновление.")


if __name__ == "__main__":
    replay()

"""
run_multi.py
============
Точка входа для пайплайна P1+P2.

Использование:
    python run_multi.py data/result.json
    python run_multi.py data/result.json data/nastya.json data/tema.json
    python run_multi.py data/result.json --window 60
"""

import argparse
import sys
from pathlib import Path

from multi_pipeline.config import PipelineConfig
from multi_pipeline.orchestrator import MultiStagePipeline


def parse_args():
    p = argparse.ArgumentParser(
        description="Поиск точек опоры P1+P2 (трёхступенчатый пайплайн)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("chats", nargs="+", metavar="CHAT.json",
                   help="Один или несколько файлов result.json")
    p.add_argument("--labeler",   default="qwen2.5:7b-instruct-q4_K_M")
    p.add_argument("--detector",  default="qwen2.5:14b")
    p.add_argument("--validator", default="qwen2.5:14b")
    p.add_argument("--narrator",  default="mistral-nemo:12b")
    p.add_argument("--composer",  default="mistral-nemo:12b")
    p.add_argument("--window", "-w", type=int, default=90)
    p.add_argument("--output", "-o", default="data/anchor_profile_multi.json")
    p.add_argument("--report", "-r", default="data/anchor_report_multi.txt")
    p.add_argument("--ollama-url",  default="http://localhost:11434")
    return p.parse_args()


def main():
    args = parse_args()

    missing = [f for f in args.chats if not Path(f).exists()]
    if missing:
        print(f"[ошибка] Файлы не найдены: {', '.join(missing)}")
        sys.exit(1)

    sep = "=" * 70
    print(sep)
    print("  Anchor Detection Pipeline — P1 + P2")
    print(f"  Stage 1 (labeler):   {args.labeler}")
    print(f"  Stage 2 (detector):  {args.detector}")
    print(f"  Stage 3 (validator): {args.validator}")
    print(f"  Stage 4 (narrator):  {args.narrator}")
    print(f"  Stage 5 (composer):  {args.composer}")
    print(f"  Окно:   {args.window} дней")
    print(f"  Чаты:   {', '.join(args.chats)}")
    print(sep)

    config = PipelineConfig(
        labeler_model   = args.labeler,
        detector_model  = args.detector,
        validator_model = args.validator,
        narrator_model  = args.narrator,
        composer_model  = args.composer,
        ollama_url      = args.ollama_url,
        window_days     = args.window,
    )

    pipeline = MultiStagePipeline(config=config, verbose=True)

    try:
        profile = pipeline.run(
            file_paths  = args.chats,
            output_path = args.output,
            report_path = args.report,
        )
        # Выводим полный ответ отдельно
        full = profile.get("full_response", "")
        if full:
            print(f"\n{'=' * 70}")
            print("  ПОЛНЫЙ ПОДДЕРЖИВАЮЩИЙ ОТВЕТ")
            print("=" * 70)
            print(f"\n{full}\n")
            # Сохраняем в отдельный файл
            resp_path = args.output.replace(".json", "_response.txt")
            Path(resp_path).write_text(full, encoding="utf-8")
            print(f"[+] Ответ сохранён: {resp_path}")

    except RuntimeError as exc:
        print(f"\n[ошибка] {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[прервано]")
        sys.exit(0)


if __name__ == "__main__":
    main()

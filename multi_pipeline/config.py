"""
config.py — конфигурация пятиступенчатого пайплайна
"""

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    # ── Модели ─────────────────────────────────────────────────────────
    labeler_model:   str = "qwen2.5:7b-instruct-q4_K_M"  # Stage 1: быстрая классификация
    detector_model:  str = "qwen2.5:14b"                  # Stage 2: детекция паттернов
    validator_model: str = "qwen2.5:14b"                  # Stage 3: скептичная проверка
    narrator_model:  str = "mistral-nemo:12b"             # Stage 4: нарративный текст
    composer_model:  str = "mistral-nemo:12b"             # Stage 5: финальный ответ
    e2_model:        str = "mistral-nemo:12b"             # E2: длинный контекст для эпизодов

    # ── Инфраструктура ─────────────────────────────────────────────────
    ollama_url: str = "http://localhost:11434"

    # ── Контекстные окна (токены) ──────────────────────────────────────
    num_ctx_labeler:   int = 8192
    num_ctx_detector:  int = 8192
    num_ctx_validator: int = 8192
    num_ctx_narrator:  int = 4096
    num_ctx_composer:  int = 8192
    num_ctx_e2:        int = 8192   # 5 эпизодов × ~21 сообщение ≈ 6k токенов

    # ── Stage 1 ────────────────────────────────────────────────────────
    min_candidates_threshold:   int = 5   # при нехватке — расширяем окно
    min_time_windows:           int = 2   # минимум уникальных ISO-недель среди кандидатов

    scan_batch_size:       int = 8    # сообщений за один LLM-вызов при full scan
    scan_batch_overlap:    int = 2    # перекрытие батчей — сохраняет диалоговый контекст
    scan_timeout_per_msg:  int = 15   # сек/сообщение → timeout = batch_size * 15
    scan_max_messages:     int = 400  # лимит full scan; 0 = без лимита (медленно)

    # ── Контекст кандидатов ────────────────────────────────────────────
    context_window_minutes: int = 5   # захватываем диалог целиком по времени
    context_min_msgs:       int = 2   # fallback: если в окне меньше N соседей — берём позиционный ±N
    context_n_positional:   int = 3

    # ── Временное окно анализа ─────────────────────────────────────────
    window_days: int = 90   # CLI: --window; расширяется автоматически при нехватке данных

    # ── Sentiment (BERT) ───────────────────────────────────────────────
    use_sentiment:        bool = True  # отключить через --no-sentiment
    sentiment_batch_size: int  = 64


DEFAULT_CONFIG = PipelineConfig()

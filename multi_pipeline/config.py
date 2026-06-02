"""
config.py
=========
Конфигурация пятиступенчатого пайплайна
"""

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    # ── Модели по ступеням ──────────────────────────────────────────
    # Ступень 1: классификатор — быстрая и дешёвая модель
    labeler_model:   str = "qwen2.5:7b-instruct-q4_K_M"
    # Ступень 2: детектор — точная модель с хорошим русским
    detector_model:  str = "qwen2.5:14b"
    # Ступень 3: валидатор — лучшая доступная модель
    validator_model: str = "qwen2.5:14b"
    # Ступень 4: нарратор — модель с хорошим русским и творческим письмом
    narrator_model:  str = "mistral-nemo:12b"
    # Ступень 5: финальный компоузер полного ответа
    composer_model:  str = "mistral-nemo:12b"
    # E2 (копинг) — длинный контекст для эпизодов
    e2_model:        str = "mistral-nemo:12b"

    # ── Инфраструктура ──────────────────────────────────────────────
    ollama_url: str = "http://localhost:11434"

    # ── Контекстные окна (токены) ───────────────────────────────────
    num_ctx_labeler:   int = 8192   # полный скан — нужен запас под батч с контекстом
    num_ctx_detector:  int = 8192   # анализ 40-60 сообщений
    num_ctx_validator: int = 8192   # оценка улик с цитатами
    num_ctx_narrator:  int = 4096   # нарративный текст — небольшой контекст
    num_ctx_composer:  int = 8192   # полный ответ — все нарративы
    num_ctx_e2:        int = 8192   # E2: 5 эпизодов × ~21 сообщение ≈ 6k tokens

    # ── Параметры ступени 1 ─────────────────────────────────────────
    min_candidates_threshold:   int = 5   # минимум кандидатов; иначе расширяем окно
    min_time_windows:           int = 2   # минимум уникальных недель среди кандидатов

    # ── Параметры полного LLM-скрининга (full_scan) ──────────────────
    scan_batch_size:       int = 8   # сообщений за один вызов
    scan_batch_overlap:    int = 2   # перекрытие между батчами (сохраняет диалог)
    scan_timeout_per_msg:  int = 15  # секунд на одно сообщение в батче
    scan_max_messages:     int = 400 # лимит полного скана; 0 = без лимита (медленно)

    # ── Контекст кандидатов (для Stage 1 → Stage 2/4) ───────────────
    # Стратегия 1: временное окно (в минутах), чтобы захватить диалог целиком
    context_window_minutes: int = 5
    # Fallback: если в окне < min_context_msgs соседей — берём positional ±N
    context_min_msgs:       int = 2
    context_n_positional:   int = 3

    # ── Временное окно анализа ──────────────────────────────────────
    window_days: int = 120          # сколько дней анализировать
    # Если данных мало (data_sufficiency=low) — расширяем до всей истории


DEFAULT_CONFIG = PipelineConfig()
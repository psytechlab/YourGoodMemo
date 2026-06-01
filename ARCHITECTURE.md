# Архитектура пайплайна — Anchor Detection v2

> Зафиксировано: 2026-06-01 (обновлено после dead code removal)
> Статус: рабочая версия v3 (скользящие окна + temporal diversity + кросс-чат агрегаты)

---

## Обзор — поток данных

```
Telegram export (result.json, nastya.json, tema.json)
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  ЗАГРУЗКА (orchestrator.py)                                     │
│  TelegramChatParser → DataFrame                                 │
│  _detect_target → определяем target_id по имени контакта       │
└──────────────────────────┬──────────────────────────────────────┘
                           │ List[ChatData]
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1 — Labeler  (qwen2.5:7b-instruct-q4_K_M)               │
│                                                                 │
│  Скользящие окна (W1→W2→...) по window_days=90d:               │
│    1. Лемматизация df (один раз на чат)                         │
│    2. Regex pre-filter → сырые кандидаты                        │
│    3. LLM-валидация кандидатов с контекстом ±2 сообщения        │
│    4. Стоп, если провалидированных >= min_candidates_threshold  │
│    5. Fallback: LLM full scan (субвыборка max 200 сообщений)    │
│                                                                 │
│  P1 якоря: S1, S2, S5, D2                                       │
│  P2 якоря: S3, C1, D3, E2 (E2 — по всей истории)               │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Dict[anchor_code → List[msg+ctx]]
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2 — Detector  (qwen2.5:14b)                              │
│                                                                 │
│  Отдельный детектор на каждый якорь.                            │
│  Получает провалидированные сообщения + aggregates.             │
│  Ищет паттерн устойчивости, формирует candidate_evidence.       │
│                                                                 │
│  P1: detect_S1, detect_S2, detect_S5, detect_D2                 │
│  P2: detect_S3, detect_C1, detect_D3, detect_E2                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Dict[anchor_code → candidate_evidence]
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3 — Validator  (qwen2.5:14b)                             │
│                                                                 │
│  Скептичный независимый валидатор.                              │
│  Цитирует оригинальные сообщения из msg_idx.                    │
│  Выносит вердикт accepted/rejected по каждой улике.             │
│  Вычисляет итоговый strength.                                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │ evidence_found, strength, uliki
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 4 — Narrator  (mistral-nemo:12b)                         │
│                                                                 │
│  Для каждой найденной опоры (strength >= 0.4):                  │
│  пишет тёплый личный абзац (4-6 предложений)                    │
│  с реальными деталями из переписки.                             │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Dict[anchor_code → narrative_text]
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 5 — Composer  (mistral-nemo:12b)                         │
│                                                                 │
│  Собирает единый поддерживающий текст:                          │
│  вступление + блоки по каждой опоре + заключение.               │
│  Стиль: близкий друг, без пафоса.                               │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
results/profile_new.json + results/report_new.txt
```

---

## Stage 1 — Labeler

**Модель:** `qwen2.5:7b-instruct-q4_K_M`  
**Контекстное окно:** 8192 токенов  
**Файл:** `multi_pipeline/stage1_labeler.py`

### Алгоритм скользящих окон

```
Для каждого якоря (S1, S2, S5, D2, S3, C1, D3, E2):

  W1 (последние 90 дней):
    → regex pre-filter → кандидаты
    → LLM-валидация с контекстом ±2 сообщения
    → если провалидировано >= 5: СТОП → передаём в Stage 2
    → иначе: W2 (90-180 дней) → ... → Wn

  Если после всех окон < 5:
    → LLM full scan (субвыборка 200 сообщений, батчи по 8)
    → LLM-валидация результатов скана

  В конце: _enrich_with_context (+context_before/after ±5 мин или ±3 позиции)
```

### Regex pre-filter по якорям

| Якорь | Лексикон | Фильтр по роли |
|-------|----------|----------------|
| S1 | EMOTIONAL_DISCLOSURE + EMPATHY_RESPONSE | target + contact |
| S2 | WE_PRONOUNS | target |
| S5 | HELP_REQUEST_MARKERS | contact |
| D2 | MONEY_KEYWORDS + DEBT_ANXIETY + FINANCIAL_POSITIVE | target |
| S3 | PRAISE_WORDS | contact |
| C1 | FUTURE_PLAN_WORDS + POSITIVE_FUTURE_MARKERS | target |
| D3 | HOBBY_ACTIVITY_WORDS + HOBBY_REGULARITY_MARKERS | target |
| E2 | STRESSOR_WORDS → эпизоды (+21 день aftermath) | target |

### Параметры (config.py)

```python
min_candidates_threshold = 5    # порог для остановки расширения окна
scan_batch_size          = 8    # сообщений в одном LLM-батче при full scan
scan_batch_overlap       = 2    # перекрытие батчей (сохраняет контекст)
scan_timeout_per_msg     = 15   # сек/сообщение → timeout = 8*15 = 120s/батч
scan_max_messages        = 200  # максимум для full scan (субвыборка равномерная)
num_ctx_labeler          = 8192
window_days              = 90   # размер одного окна (CLI: --window)
context_window_minutes   = 5    # временное окно для context_before/after
context_n_positional     = 3    # fallback: ±N позиций если окно пустое
context_min_msgs         = 2    # минимум соседей для временного окна
```

### Формат выхода

```json
{
  "S1": [
    {
      "msg_id": "msg_123",
      "ts": "2026-04-15 14:32",
      "sender": "Захар",
      "text": "устал немного...",
      "role": "target_emotional",
      "context_before": [{"msg_id": "msg_121", "sender": "Рома", "text": "..."}],
      "context_after":  [{"msg_id": "msg_125", "sender": "Рома", "text": "..."}]
    }
  ]
}
```

---

## Stage 2 — Detector

**Модель:** `qwen2.5:14b`  
**Контекстное окно:** 8192 токенов  
**Timeout:** 360 секунд  
**Файл:** `multi_pipeline/stage2_detector.py`, `stage2_p2_detector.py`

### Что делает

Получает провалидированные сообщения от Stage 1 + базовые агрегаты (история чата, активность, voice_share, initiation_balance). Ищет устойчивый паттерн — не единичный случай, а повторяющееся поведение.

### Детекторы

| Якорь | Входные данные | Что ищет |
|-------|---------------|----------|
| S1 | target_msgs (emotional) + contact_msgs (empathy) | взаимная поддержка, не единичная |
| S2 | we_msgs + aggregates | регулярное «мы», совместные события |
| S5 | help_msgs + pings (обращения после молчания) | паттерн «ко мне обращаются» |
| D2 | financial_msgs + aggregates | стабильный финансовый фон без тревоги |
| S3 | praise_msgs | похвала/признание, не ирония |
| C1 | plan_msgs | конкретные планы с горизонтом |
| D3 | hobby_msgs | регулярность + прогресс |
| E2 | episodes [{stressor, aftermath_msgs}] | активный копинг после стрессора |

### Формат выхода

```json
{
  "anchor_code": "S1",
  "evidence_found": true,
  "candidate_evidence": [
    {
      "message_ids": ["msg_123", "msg_125"],
      "why": "Захар делится тревогой, Рома отвечает поддержкой"
    }
  ],
  "subscores": {
    "self_disclosure_depth": 0.7,
    "empathic_response_quality": 0.8,
    "pattern_regularity": 0.6,
    "relationship_longevity": 0.9
  },
  "data_sufficiency": "medium"
}
```

---

## Stage 3 — Validator

**Модель:** `qwen2.5:14b`  
**Контекстное окно:** 8192 токенов  
**Файл:** `multi_pipeline/stage3_validator.py`

### Что делает

Независимый скептичный контроль: берёт `candidate_evidence` от Stage 2, цитирует оригинальные тексты из `msg_idx`, выносит вердикт `accepted`/`rejected` по каждой улике. Вычисляет финальный `strength`.

### Шкала strength

| strength | Значение |
|----------|----------|
| 0.8–1.0 | ≥3 сильных подтверждённых улики, регулярный паттерн |
| 0.5–0.7 | 1–2 улики подтверждены, паттерн неполный |
| 0.2–0.4 | улики слабые или единичные |
| 0.0–0.1 | нет подтверждённых улик |

Точка опоры считается найденной если `evidence_found=True` **и** `strength >= 0.4`.

---

## Stage 4 — Narrator

**Модель:** `mistral-nemo:12b`  
**Контекстное окно:** 4096 токенов  
**Файл:** `multi_pipeline/stage4_narrator.py`

Для каждой опоры с `strength >= 0.4` пишет абзац (4–6 предложений). Стиль: близкий друг, знающий твою жизнь. Конкретные детали из переписки, без психологических терминов.

---

## Stage 5 — Composer

**Модель:** `mistral-nemo:12b`  
**Контекстное окно:** 8192 токенов  
**Файл:** `multi_pipeline/stage5_composer.py`

Собирает нарративы в единый текст: вступление (без пафоса) + блоки по каждой опоре + заключение («Это есть. Не исчезнет.»).

---

## Конфигурация — полная таблица

```python
# Модели
labeler_model   = "qwen2.5:7b-instruct-q4_K_M"
detector_model  = "qwen2.5:14b"
validator_model = "qwen2.5:14b"
narrator_model  = "mistral-nemo:12b"
composer_model  = "mistral-nemo:12b"
e2_model        = "mistral-nemo:12b"

# Контекстные окна (токены)
num_ctx_labeler   = 8192
num_ctx_detector  = 8192
num_ctx_validator = 8192
num_ctx_narrator  = 4096
num_ctx_composer  = 8192
num_ctx_e2        = 8192

# Stage 1 — скользящие окна
window_days              = 90    # CLI: --window (размер одного окна)
min_candidates_threshold = 5     # минимум валидированных для остановки
min_time_windows         = 2     # минимум уникальных ISO-недель (защита от локального экстремума)
scan_batch_size          = 8
scan_batch_overlap       = 2
scan_timeout_per_msg     = 15
scan_max_messages        = 400   # субвыборка при full scan

# Stage 2 — dynamic timeout
# timeout = max(180, n_msgs * 10) для всех детекторов (E2: max(240, n_msgs*10))

# Контекст кандидатов
context_window_minutes   = 5
context_min_msgs         = 2
context_n_positional     = 3
```

---

## Тайминги (эталон, 3 чата ~2200+750+600 сообщений)

| Этап | Время | Примечание |
|------|-------|------------|
| Загрузка | ~20s | парсинг 3 JSON |
| Stage 1 P1 | ~2–5 мин | в осн. мгновенно если regex находит ≥5 |
| Stage 2 P1 | ~20 мин | 4 детектора × qwen2.5:14b |
| Stage 3 P1 | ~1–3 мин | валидация улик |
| Stage 1 P2 | ~5s | regex быстро |
| Stage 2 P2 | ~10 мин | 4 детектора P2 |
| Stage 3 P2 | ~3 мин | |
| Stage 4 | ~2–5 мин | нарративы (если есть опоры) |
| Stage 5 | ~1–2 мин | компоузер |
| **Итого** | **~36–50 мин** | при хорошем Ollama |

---

## Результаты прогонов

### Прогон 2026-05-31 — v1 (полный скан без субвыборки)
- Время: ~1h 22m
- Найдено: D3 (боулинг, strength неизвестен)
- Проблема: full scan 2200 сообщений без лимита = ~7h на Stage 1

### Прогон 2026-06-01 — v2 (regex pass-through, без валидации)
- Время: 36 мин
- Найдено: ничего (все якоря: data=low, uliki=0/0)
- Проблема: Stage 1 передавал сырой regex-шум → Stage 2 корректно сказал «нет паттерна»

### Прогон 2026-06-01 — v3 (текущая архитектура)
- Изменения: скользящие окна + LLM-валидация с контекстом ±2 сообщения
- Статус: ожидается

---

## Известные проблемы и риски

| Проблема | Статус | Решение |
|----------|--------|---------|
| Ollama периодически зависает на батчах (>10 мин без ответа) | известна | самостоятельно разрешается; таймаут 120s/батч |
| E2 OOM на mistral-nemo:12b | решена | num_ctx_e2=8192 (было 32768) |
| D2/S5 validation timeout | решена | динамический timeout = max(180, n*6) |
| Stage 2 получает шум без предфильтрации | решена | LLM-валидация в Stage 1 с контекстом |
| Валидатор не видел диалоговый контекст | решена | _fmt_candidate_with_ctx ±2 сообщения |

"""
stage3_validator.py
===================
Ступень 3: независимая валидация улик и финальная оценка.

Получает кандидатов от Stage 2, цитирует оригинальные сообщения
и выносит скептичный вердикт по каждой улике.

Ключевая роль:
  - Отсекает галлюцинации и ошибочные интерпретации Stage 2.
  - Присваивает итоговый strength на основе доли подтверждённых улик.
  - Формирует финальный объект anchor-результата.
"""

import json
from typing import Any, Dict, List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from multi_pipeline.config import PipelineConfig
from anchor_detection.llm_client import OllamaClient


# ══════════════════════════════════════════════════════════════════════
# Промт валидатора
# ══════════════════════════════════════════════════════════════════════

_VALIDATOR_SYSTEM = """Ты — строгий валидатор доказательств точки опоры.

Тебе передан предварительный анализ (от детектора) с кандидатами на улики.
К каждой улике прилагаются ОРИГИНАЛЬНЫЕ тексты сообщений.

Твоя задача — проверить каждую улику и вынести НЕЗАВИСИМЫЙ ВЕРДИКТ.

КРИТЕРИИ ПРИНЯТИЯ улики:
  ✓ Улика ссылается на конкретный паттерн поведения, а не абстракцию.
  ✓ Сообщения, на которые ссылается улика, реально существуют в контексте.
  ✓ Паттерн прослеживается в нескольких сообщениях (не единичный эпизод).
  ✓ Интерпретация логична: факты подтверждают вывод.

КРИТЕРИИ ОТКЛОНЕНИЯ улики:
  ✗ Сообщение является мемом, шуткой, шаблонной фразой.
  ✗ Слово использовано не в том смысле (например, «должен» = обязан ≠ финансовый долг).
  ✗ Паттерн единичный — один случай не делает опору.
  ✗ Детектор додумал смысл, которого нет в тексте.
  ✗ Сообщение не от того отправителя (например, ответственность приписана не пользователю).

ИТОГОВАЯ ОЦЕНКА strength:
  - 0.8-1.0: >= 3 сильных подтверждённых улик с регулярным паттерном.
  - 0.5-0.7: 1-2 улики подтверждены, но паттерн неполный.
  - 0.2-0.4: улики слабые или единичные.
  - 0.0-0.1: нет подтверждённых улик.

ПРАВИЛА:
1. Будь скептичен. Лучше недооценить, чем придумать опору там, где её нет.
2. Если данных мало → data_sufficiency="low", strength <= 0.4.
3. Все текстовые поля — только на русском языке.
4. Ответ строго в JSON.

Формат ответа:
{
  "anchor_code": "<код>",
  "evidence_found": true | false,
  "strength": 0.0-1.0,
  "evidence": [
    {
      "message_ids": ["msg_123"],
      "why": "подтверждено: ...",
      "verdict": "accepted" | "rejected",
      "verdict_reason": "почему принято/отклонено"
    }
  ],
  "subscores": { ... },
  "anti_signals": ["строка"],
  "caveat": "ограничения анализа",
  "data_sufficiency": "low" | "medium" | "high",
  "window": "<передай из входных данных>"
}"""


# ══════════════════════════════════════════════════════════════════════
# Вспомогательные функции
# ══════════════════════════════════════════════════════════════════════

def _enrich_evidence_with_texts(
    candidate_evidence: List[Dict],
    msg_index: Dict[int, Dict],
) -> List[Dict]:
    """
    К каждому элементу evidence прикрепляет полные тексты сообщений.
    msg_index: {msg_id (int) -> row_dict}
    """
    enriched = []
    for ev in candidate_evidence:
        citations = []
        for mid_str in ev.get("message_ids", []):
            try:
                row = msg_index.get(str(mid_str), {})
                if row:
                    text = str(row.get("text", ""))[:300]
                    ct   = row.get("content_type", "")
                    if not text and ct:
                        text = f"[{ct}]"
                    citations.append({
                        "msg_id": mid_str,
                        "ts":     str(row.get("date", ""))[:16],
                        "sender": row.get("sender", "?"),
                        "text":   text,
                    })
                else:
                    citations.append({"msg_id": mid_str, "error": "не найдено"})
            except Exception:
                citations.append({"msg_id": mid_str, "error": "ошибка"})

        enriched.append({**ev, "original_messages": citations})
    return enriched


def _default_result(code: str, window: str) -> Dict:
    return {
        "anchor_code":      code,
        "evidence_found":   False,
        "strength":         0.0,
        "evidence":         [],
        "subscores":        {},
        "anti_signals":     [],
        "caveat":           "Валидация завершилась ошибкой.",
        "data_sufficiency": "low",
        "window":           window,
    }


# ══════════════════════════════════════════════════════════════════════
# Основной класс
# ══════════════════════════════════════════════════════════════════════

class EvidenceValidator:
    """
    Валидирует кандидатов от Stage 2 и формирует финальный результат.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.client = OllamaClient(
            model    = config.validator_model,
            base_url = config.ollama_url,
            num_ctx  = config.num_ctx_validator,
            timeout  = 180,
        )

    def validate(
        self,
        candidate: Dict,
        msg_index: Dict[int, Dict],
        window: str = "90d",
    ) -> Dict:
        """
        Принимает кандидата от Stage 2, возвращает финальный anchor-результат.

        Параметры
        ----------
        candidate : результат Stage 2 для одного anchor
        msg_index : {msg_id → row_dict} для получения полных текстов
        window    : строка окна (например "90d" или "full_history")
        """
        code = candidate.get("anchor_code", "?")

        # Если у Stage 2 нет улик — не тратим вызов, сразу возвращаем пустой результат
        cand_evidence = candidate.get("candidate_evidence", [])
        if not cand_evidence:
            result = _default_result(code, window)
            result["caveat"] = (
                f"Stage 2 не нашёл кандидатов для {code}. "
                "Нет сообщений с релевантными метками."
            )
            result["data_sufficiency"] = candidate.get("data_sufficiency", "low")
            return result

        # Прикрепляем оригинальные тексты
        enriched = _enrich_evidence_with_texts(cand_evidence, msg_index)

        user_prompt = json.dumps({
            "anchor_code":         code,
            "window":              window,
            "stage2_subscores":    candidate.get("subscores", {}),
            "stage2_data_suff":    candidate.get("data_sufficiency", "low"),
            "candidate_evidence":  enriched,
        }, ensure_ascii=False, indent=2)

        try:
            result = self.client.chat(
                system      = _VALIDATOR_SYSTEM,
                user        = user_prompt,
                temperature = 0.0,   # валидатор должен быть детерминированным
            )

            # Гарантируем обязательные поля
            result.setdefault("anchor_code",      code)
            result.setdefault("evidence_found",   False)
            result.setdefault("strength",         0.0)
            result.setdefault("evidence",         [])
            result.setdefault("subscores",        {})
            result.setdefault("anti_signals",     [])
            result.setdefault("caveat",           "")
            result.setdefault("data_sufficiency", "low")
            result["window"] = window

            # Зажимаем strength в [0, 1]
            result["strength"] = max(0.0, min(1.0, float(result["strength"])))

            # evidence_found = True только если strength >= 0.4 и есть accepted evidence
            accepted = [
                ev for ev in result.get("evidence", [])
                if ev.get("verdict") == "accepted"
            ]
            if result["strength"] < 0.4 or not accepted:
                result["evidence_found"] = False

            return result

        except Exception as exc:
            print(f"  [Stage 3 / {code}] Ошибка: {exc}")
            return _default_result(code, window)
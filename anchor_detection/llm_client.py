"""
llm_client.py
=============
Тонкая обёртка над Ollama REST API

Поддерживает:
  - JSON-режим (гарантирует парсируемый ответ)
  - Настраиваемые параметры (модель, температура, контекст)
  - Базовую валидацию ответа
  - Проверку доступности сервера
"""

import json
import re
import time
import requests
from typing import Any, Dict, List, Optional


class OllamaError(Exception):
    """Ошибка при работе с Ollama API."""


class OllamaClient:
    """
    Клиент для Ollama API

    Параметры
    ----------
    model : str
        Имя модели в Ollama. По умолчанию qwen2.5:7b-instruct-q4_K_M.
    base_url : str
        Базовый URL сервера Ollama.
    timeout : int
        Таймаут запроса в секундах.
    num_ctx : int
        Размер контекстного окна (токены)
    """

    DEFAULT_MODEL = "qwen2.5:7b-instruct-q4_K_M"
    DEFAULT_URL   = "http://localhost:11434"

    def __init__(
        self,
        model:    str = DEFAULT_MODEL,
        base_url: str = DEFAULT_URL,
        timeout:  int = 180,
        num_ctx:  int = 8192,
    ):
        self.model    = model
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        self.num_ctx  = num_ctx

    # ------------------------------------------------------------------
    # Публичные методы
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Проверяет, запущен ли Ollama-сервер."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def chat(
        self,
        system:      str,
        user:        str,
        temperature: float = 0.1,
        json_mode:   bool  = True,
        retries:     int   = 2,
        timeout:     Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Отправляет запрос в LLM и возвращает JSON.

        Параметры
        ----------
        system : str
            Системный промт (роль / инструкция)
        user : str
            Пользовательский промт (данные для анализа)
        temperature : float
            Температура генерации
        json_mode : bool
            Если True — Ollama возвращает JSON
        retries : int
            Кол-во повторных попыток при сетевой ошибке

        Возвращает
        ----------
        dict — JSON из ответа модели
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

        payload: Dict[str, Any] = {
            "model":   self.model,
            "messages": messages,
            "stream":  False,
            "options": {
                "temperature": temperature,
                "num_ctx":     self.num_ctx,
            },
        }
        if json_mode:
            payload["format"] = "json"

        effective_timeout = timeout if timeout is not None else self.timeout
        for attempt in range(retries + 1):
            try:
                raw = self._post("/api/chat", payload, timeout=effective_timeout)
                content = raw["message"]["content"]
                if json_mode:
                    return self._parse_json(content)
                return content          # сырой текст для нарративных задач
            except (requests.RequestException, OllamaError) as exc:
                if attempt == retries:
                    raise
                time.sleep(2 ** attempt)  # экспоненциальная пауза

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _post(self, path: str, payload: Dict, timeout: Optional[int] = None) -> Dict:
        """Низкоуровневый POST-запрос к Ollama."""
        url = self.base_url + path
        t = timeout if timeout is not None else self.timeout
        try:
            resp = requests.post(url, json=payload, timeout=t)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            raise OllamaError(f"HTTP {resp.status_code}: {resp.text[:300]}") from exc
        except requests.ConnectionError as exc:
            raise OllamaError(
                "Не удалось подключиться к Ollama. "
                "Убедись, что сервер запущен: ollama serve"
            ) from exc
        except requests.Timeout as exc:
            raise OllamaError(f"Таймаут запроса ({t}s)") from exc

    @staticmethod
    def _parse_json(text: str) -> Dict:
        """
        Парсит JSON из ответа модели
        Если модель обернула JSON в ```json ... ```, извлекает его
        """
        text = text.strip()

        # Прямой парсинг
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Поиск JSON-блока в ответе
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))

        # Поиск первого {...}
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())

        raise OllamaError(f"Не удалось распарсить JSON из ответа модели:\n{text[:500]}")

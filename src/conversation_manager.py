import csv
from datetime import datetime


CSV_PATH = "logs/directives.csv"

_csv_file = None
_csv_writer = None


def _ensure_csv():
    global _csv_file, _csv_writer
    if _csv_file is None:
        _csv_file = open(CSV_PATH, "w", newline="", encoding="utf-8")
        _csv_writer = csv.writer(_csv_file)
        _csv_writer.writerow(["timestamp", "directive", "user_message", "response"])


class ConversationManager:
    #  сам хранит историю, обрезает, решает первый ход или продолжение, собирает messages
    def __init__(
        self,
        llm_user,
        system_prompt,
        csv_logging=True,
        filler_instruction="",
        sides=None,
        first_message=None,
        history_limit=10,
    ):
        self.llm_user = llm_user
        self.system_prompt = system_prompt
        self.csv_logging = csv_logging
        self.filler_instruction = filler_instruction
        self.sides = sides or {}
        self.first_message = first_message # first_message — текст первого сообщения
        self.history_limit = history_limit
        self.history = []

    def get_response(self, directive, side, is_filler=False, incoming_message=None):
        # Реплика собеседника, которая используется как user_message
        if incoming_message:
            self.history.append({"role": "user", "content": incoming_message})


        # Решает первый ход или продолжение
        if not self.history and self.first_message:
            user_message = self.first_message
            history_for_llm = []
        else:
            user_message = "Продолжи."
            if self.history:
                # Продолжение после последней реплики собеседника 
                user_message = self.history[-1]["content"]
            # Обрезка истории внутри
            history_for_llm = self.history[-self.history_limit:]

        messages = []
        messages.append({"role": "system", "content": self.system_prompt})

        if is_filler:
            # Filler: тема + инструкция
            messages.append({"role": "system", "content": f"TOPIC: {directive}"})
            if self.filler_instruction:
                messages.append({"role": "system", "content": self.filler_instruction})
        else:
            # Anchor: директива + side
            messages.append({"role": "system", "content": f"DIRECTIVE: {directive}"})
            side_instruction = self.sides.get(side, "")
            if side_instruction:
                messages.append({"role": "system", "content": side_instruction})

        messages += history_for_llm
        messages.append({"role": "user", "content": user_message})

        response = self.llm_user.generate(messages)

        # Сохраняет ответ в свою историю
        self.history.append({"role": "assistant", "content": response})

        if self.csv_logging:
            _ensure_csv()
            _csv_writer.writerow([
                datetime.now().isoformat(),
                directive,
                user_message,
                response,
            ])

        return response
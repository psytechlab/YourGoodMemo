import csv
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler()
                    ])

# --- CSV logging setup ---
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       f"logs")
CSV_PATH = os.path.join(CSV_DIR, f"directives_{TIMESTAMP}.csv")
os.makedirs(CSV_DIR, exist_ok=True)
_csv_file = None
_csv_writer = None


def _ensure_csv():
    global _csv_file, _csv_writer
    if _csv_file is None:
        _csv_file = open(CSV_PATH, "w", newline="", encoding="utf-8")
        _csv_writer = csv.writer(_csv_file)
        _csv_writer.writerow(["timestamp", "directive", "user_message", "response"])


class ConversationManager:
    def __init__(self, llm_client, system_prompt, reasoner, csv_logging=True):
        self.llm_client = llm_client
        self.system_prompt = system_prompt
        self.reasoner = reasoner
        self.csv_logging = csv_logging

    def get_response(self, history, user_message, directive: str):
        messages = []
        messages.append({"role": "system", "content": self.system_prompt})
        messages += history

        messages.append({"role": "user", "content": user_message + f"| {directive}"})

        response = self.llm_client.generate(messages)

        if self.csv_logging:
            _ensure_csv()
            _csv_writer.writerow([
                datetime.now().isoformat(),
                directive,
                user_message,
                response
            ])

        return response

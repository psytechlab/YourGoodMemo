import telebot
from telebot import types
import os
from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
import time
import datetime
from requests.exceptions import RequestException
from src.models import UserSettings
from string import Template

from src.llm_client import LLMClient
from src.reasoner import RandomReasoner, LLMReasoner, DummyReasoner
from src.db_ops import init_database, save_to_database, get_chat_history
from src.utils import format_dialog
from src.conversation_manager import ConversationManager

import logging
log_dir = Path("./logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_dir/f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

load_dotenv()

# Конфигурация
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
BOTHUB_API_KEY = os.getenv("BOTHUB_API_KEY")
OPENAI_API_URL = 'https://bothub.chat/api/v2/openai/v1/chat/completions'
MODEL_NAME = "gpt-4o"
CONTEXT_LIMIT = 999999999
REASONER_TYPE = os.getenv("REASONER_TYPE")

if REASONER_TYPE not in ["LLM", "RANDOM", "DUMMY"]:
    raise ValueError("REASONER_TYPE must be explicitly set to 'LLM', 'RANDOM' or 'DUMMY' in the .env file")

def get_reasoner():
    if REASONER_TYPE == "LLM":
        return LLMReasoner(MODEL_CLIENT, "data/situations/angry_.yaml")
    if REASONER_TYPE == "DUMMY":
        return DummyReasoner()
    return RandomReasoner()

# Инициализация бота
bot = telebot.TeleBot(TELEGRAM_TOKEN, num_threads=6)
user_settings_glob = {}

DB_PATH = Path.cwd()/"db/chat_history.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
DB_PATH.touch()
   
def get_history_as_json(user_id, session_id, context_limit):
    history = get_chat_history(DB_PATH, user_id, session_id, context_limit) 
    message_hist = []
    for msg, resp in history[::-1]:  # Reverse to get chronological order
        message_hist.extend([
            {"role": "user", "content": msg},
            {"role": "assistant", "content": resp}
        ])
    return message_hist

MODEL_CLIENT = LLMClient(model_name=MODEL_NAME, base_url=OPENAI_API_URL, auth_token=BOTHUB_API_KEY)
MODEL_ASSESTMENT = LLMClient(model_name=MODEL_NAME, base_url=OPENAI_API_URL, auth_token=BOTHUB_API_KEY, temperature=0.2)

def assest_session(user_id, session_id, context_limit):
    history = get_chat_history(DB_PATH, user_id, session_id, context_limit)
    if len(history) == 0:
        return ""
    dialog = format_dialog(history)
    with open("prompts/assestment.txt") as f:
        prompt = Template(f.read())
    return MODEL_ASSESTMENT.respond(prompt.substitute(session_text=dialog))

def send_survey(message):
    bot.send_message(message, text="Пожалуйста, пройдите опрос по работе бота, когда попробуете его: https://forms.gle/hBHpdomnXsZcadbG7")

@bot.message_handler(commands=['start'])
def send_welcome(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    #btn1 = types.KeyboardButton("Агрессивный клиент #1")
    #btn2 = types.KeyboardButton("Агрессивный клиент #2")
    btn1 = types.KeyboardButton("Сбросить контекст")
    btn2 = types.KeyboardButton("Оценить сессию")
    btn3 = types.KeyboardButton("Справка")
    markup.add(btn1, btn2, btn3)
    bot.send_message(message.chat.id, text="Добро пожаловать в бот-тренажер для психотерапевтов. С вами будет говорить агрессивный клиент. Поприветствуйте клиента, чтобы начать. " \
    "\n" \
    "Как закончите, напишите 'Оценить сессию' (или /assest_session). " \
    "\n" \
    "Вызвать справку по боту: 'Справка' или /help", reply_markup=markup)
    
    with open(Path.cwd()/"prompts/agressive_1.txt") as f:
        system_prompt = f.read()
    
    conv_manager = ConversationManager(MODEL_CLIENT, system_prompt, get_reasoner())
    user_settings_glob[message.from_user.id] = UserSettings(session_id=uuid4().hex, conversation_manager=conv_manager)

@bot.message_handler(commands=['help'])
def send_help(message):
    with open(Path.cwd()/"reference/HELP.txt") as f:
        help_text = f.read()
    bot.send_message(message.chat.id, text=help_text)

@bot.message_handler(commands=['reset_context'])
def reset_context(message):
    """Обработчик команды /reset_context"""
    if message.from_user.id not in user_settings_glob:
        send_welcome(message)
        return
    user_settings_glob[message.from_user.id].session_id = uuid4().hex
    bot.reply_to(message, "Контекст сброшен")

@bot.message_handler(commands=['assest_session'])
def assest_sess(message):
    """Обработчик команды /assest_session"""
    if message.from_user.id not in user_settings_glob:
        send_welcome(message)
        return
    assestment = assest_session(message.from_user.id, user_settings_glob[message.from_user.id].session_id, CONTEXT_LIMIT)
    bot.send_message(message.chat.id, text="Началась оценка вашей сессии. Это может занять некоторое время.")
    if assestment == "":
        bot.send_message(message.chat.id, text="Диалог пуст. Проведите сессию, чтобы ее можно было оценить.")
        return
    bot.send_message(message.chat.id, text=assestment)
    reset_context(message)
    send_survey(message)

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    """Основной обработчик сообщений"""
    user_id = message.from_user.id

    if user_id not in user_settings_glob:
        send_welcome(message)
    elif (message.text == "Агрессивный клиент #1"):
        bot.reply_to(message, "Выбран агрессивный клиент #1. Когда будете готовы, напишите 'здравствуйте, чем я могу вам помочь?'")
        with open(Path.cwd()/"prompts/agressive_1.txt") as f:
            system_prompt = f.read()
        user_settings_glob[message.from_user.id].conversation_manager.system_prompt = system_prompt
        user_settings_glob[message.from_user.id].session_id = uuid4().hex
    elif (message.text == "Агрессивный клиент #2"):
        bot.reply_to(message, "Выбран агрессивный клиент #2. Когда будете готовы, напишите 'здравствуйте, чем я могу вам помочь?'")
        with open(Path.cwd()/"prompts/agressive_2.txt") as f:
            system_prompt = f.read()
        user_settings_glob[message.from_user.id].conversation_manager.system_prompt = system_prompt
        user_settings_glob[message.from_user.id].session_id = uuid4().hex
    elif message.text.lower() == "Справка".lower():
        send_help(message)
    elif message.text.lower() == "Сбросить контекст".lower():
        reset_context(message)
    elif message.text.lower() == "Оценить сессию".lower():
        assest_sess(message)
    elif user_settings_glob[message.from_user.id].conversation_manager is not None:
        session_id = user_settings_glob[user_id].session_id
        user_message = message.text
        exceed_history_limit = False

        history = get_history_as_json(user_id, session_id, CONTEXT_LIMIT)

        ai_response = user_settings_glob[user_id].conversation_manager.get_response(
            history=history,
            user_message=user_message
        )
        
        messages_len = len(history) + 2 # system + user
        if messages_len > CONTEXT_LIMIT + 1:
            exceed_history_limit = True

        # Сохраняем диалог в базу данных
        save_to_database(DB_PATH, user_id, user_settings_glob[user_id].session_id, user_message, ai_response)
        
        # Отправляем ответ пользователю
        bot.send_message(message.chat.id, text=ai_response)
        if exceed_history_limit:
            bot.send_message(message.chat.id, text="Лимит контекста превышен")
            reset_context(message)
    else:
        send_help(message)

def main():
    """Основная функция"""
    print("Инициализации базы данных...")
    init_database(DB_PATH)
    
    # Запуск бота
    print("Запуск бота...")
    # https://qna.habr.com/q/1319792?ysclid=mde9loy925638658670
    while True:
        try:
            bot.infinity_polling(timeout=10, long_polling_timeout = 5)
        except RequestException as err:
            print(err)
            print('* Connection failed, waiting to reconnect...')
            time.sleep(15)
            print('* Reconnecting.')

if __name__ == '__main__':
    main()


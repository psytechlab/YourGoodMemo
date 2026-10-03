import sqlite3
from datetime import datetime

def init_database(db_path):
    """Инициализация базы данных"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            session_id TEXT,
            message TEXT,
            response TEXT,
            timestamp DATETIME
        )
    ''')
    
    conn.commit()
    conn.close()

def save_to_database(db_path, user_id, session_id, message, response):
    """Сохранение диалога в базу данных"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO chat_history (user_id, session_id, message, response, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, session_id, message, response, datetime.now()))
    
    conn.commit()
    conn.close()

def get_chat_history(db_path, user_id, session_id, limit):
    """Получение истории диалога пользователя"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT message, response FROM chat_history
        WHERE user_id = ? AND session_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (user_id, session_id, limit))
    
    history = cursor.fetchall()
    conn.close()
    
    return history

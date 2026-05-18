import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "transcripts.db")

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Safely migrate by dropping for development purposes
    c.execute("DROP TABLE IF EXISTS records")
    c.execute('''
        CREATE TABLE records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            summary TEXT,
            english_analysis TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_record(text: str, summary: str, english_analysis: str = ""):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO records (text, summary, english_analysis, timestamp) VALUES (?, ?, ?, ?)",
        (text, summary, english_analysis, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def search_records(query: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Simple LIKE search for both text and summary
    search_term = f"%{query}%"
    c.execute(
        "SELECT id, text, summary, english_analysis, timestamp FROM records WHERE text LIKE ? OR summary LIKE ? ORDER BY timestamp DESC LIMIT 50",
        (search_term, search_term)
    )
    results = [
        {"id": row[0], "text": row[1], "summary": row[2], "english_analysis": row[3], "timestamp": row[4]}
        for row in c.fetchall()
    ]
    conn.close()
    return results

def get_recent_records(limit=10):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, text, summary, english_analysis, timestamp FROM records ORDER BY timestamp DESC LIMIT ?",
        (limit,)
    )
    results = [
        {"id": row[0], "text": row[1], "summary": row[2], "english_analysis": row[3], "timestamp": row[4]}
        for row in c.fetchall()
    ]
    conn.close()
    return results

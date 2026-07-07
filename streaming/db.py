import sqlite3
from pathlib import Path

DB_PATH = Path("data") / "snitch.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS classified_files (
            file_hash TEXT PRIMARY KEY,
            file_path TEXT,
            file_name TEXT,
            category TEXT,
            risk_tier TEXT,
            decided_by_tier INTEGER,
            confidence REAL,
            compliance_flags TEXT,
            notes TEXT,
            detected_at TEXT,
            classified_at TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_hash TEXT,
            file_name TEXT,
            risk_tier TEXT,
            reason TEXT,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"db ready at {DB_PATH}")
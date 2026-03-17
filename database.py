import sqlite3

def create_db():

    conn = sqlite3.connect("sales.db")
    c = conn.cursor()

    # ================= USERS TABLE =================
    c.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password BLOB
    )
    """)

    # ================= SALES DATA TABLE =================
    c.execute("""
    CREATE TABLE IF NOT EXISTS sales(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        date TEXT,
        product TEXT,
        country TEXT,
        quantity INTEGER,
        revenue REAL,
        cost REAL
    )
    """)

    # ================= SYSTEM ACTIVITY TABLE =================
    c.execute("""
    CREATE TABLE IF NOT EXISTS system_logs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        action TEXT,
        records INTEGER,
        log_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()
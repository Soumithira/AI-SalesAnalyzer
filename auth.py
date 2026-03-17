
import sqlite3
import bcrypt


# ================= HASH PASSWORD =================
def hash_password(p):
    return bcrypt.hashpw(p.encode(), bcrypt.gensalt())


# ================= REGISTER USER =================
def register_user(u, p):

    conn = sqlite3.connect("sales.db")
    c = conn.cursor()

    try:
        c.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (u, hash_password(p))
        )
        conn.commit()
        return True

    except sqlite3.IntegrityError:
        return False

    finally:
        conn.close()


# ================= LOGIN USER =================
def login_user(u, p):

    conn = sqlite3.connect("sales.db")
    c = conn.cursor()

    c.execute("SELECT password FROM users WHERE username=?", (u,))
    r = c.fetchone()

    conn.close()

    if r:
        return bcrypt.checkpw(p.encode(), r[0])

    return False

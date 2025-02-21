import sqlite3

def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS feedback 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, response TEXT, rating INTEGER)''')
    # Sample user
    c.execute("INSERT OR IGNORE INTO users VALUES (?, ?)", ("testuser", "testpass"))
    conn.commit()
    conn.close()

def authenticate_user(username: str, password: str) -> bool:
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    return result and result[0] == password

def save_feedback(username: str, response: str, rating: int):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("INSERT INTO feedback (username, response, rating) VALUES (?, ?, ?)", 
              (username, response, rating))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
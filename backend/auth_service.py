import os
import sqlite3
import secrets
import hashlib
from typing import Optional, Dict, Any
from pydantic import BaseModel, EmailStr
from fastapi import APIRouter, HTTPException, Header, Depends

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "backend", "users.db")

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000).hex()


def init_db():
    conn = get_db()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL COLLATE NOCASE,
                full_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT DEFAULT 'Field Agronomist',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
        """)

        # Seed default demonstration account
        cursor = conn.execute("SELECT id FROM users WHERE email = 'demo@plantcare.ai'")
        if not cursor.fetchone():
            salt = secrets.token_bytes(16)
            p_hash = hash_password("PlantCare123!", salt)
            conn.execute(
                "INSERT INTO users (email, full_name, password_hash, salt, role) VALUES (?, ?, ?, ?, ?)",
                ("demo@plantcare.ai", "Dr. Elena Vance", p_hash, salt.hex(), "Senior Crop Pathologist")
            )
            print("Auth: Default demo account seeded (demo@plantcare.ai / PlantCare123!)")
    conn.close()


init_db()


# Pydantic Schemas
class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: str


class LoginRequest(BaseModel):
    email: str
    password: str


def get_current_user_from_token(authorization: Optional[str] = Header(None)) -> Optional[Dict[str, Any]]:
    if not authorization:
        return None
    token = authorization.replace("Bearer ", "").strip()
    if not token:
        return None

    conn = get_db()
    try:
        cursor = conn.execute("""
            SELECT u.id, u.email, u.full_name, u.role, u.created_at
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.token = ?
        """, (token,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    finally:
        conn.close()


@router.post("/register")
def register(req: RegisterRequest):
    email = req.email.strip().lower()
    full_name = req.full_name.strip()
    password = req.password

    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Please provide a valid email address.")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters long.")
    if not full_name:
        raise HTTPException(status_code=400, detail="Please provide your full name.")

    salt = secrets.token_bytes(16)
    p_hash = hash_password(password, salt)

    conn = get_db()
    try:
        with conn:
            cursor = conn.execute(
                "INSERT INTO users (email, full_name, password_hash, salt, role) VALUES (?, ?, ?, ?, ?)",
                (email, full_name, p_hash, salt.hex(), "Certified Agronomist")
            )
            user_id = cursor.lastrowid
            token = secrets.token_hex(32)
            conn.execute("INSERT INTO sessions (token, user_id) VALUES (?, ?)", (token, user_id))

        return {
            "token": token,
            "user": {
                "id": user_id,
                "email": email,
                "full_name": full_name,
                "role": "Certified Agronomist"
            },
            "message": "Account created successfully."
        }
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="An account with this email already exists.")
    finally:
        conn.close()


@router.post("/login")
def login(req: LoginRequest):
    email = req.email.strip().lower()
    password = req.password

    conn = get_db()
    try:
        cursor = conn.execute(
            "SELECT id, email, full_name, password_hash, salt, role FROM users WHERE email = ?",
            (email,)
        )
        user = cursor.fetchone()
        if not user:
            raise HTTPException(status_code=401, detail="Invalid email or password.")

        salt = bytes.fromhex(user["salt"])
        expected_hash = hash_password(password, salt)
        if not secrets.compare_digest(user["password_hash"], expected_hash):
            raise HTTPException(status_code=401, detail="Invalid email or password.")

        token = secrets.token_hex(32)
        with conn:
            conn.execute("INSERT INTO sessions (token, user_id) VALUES (?, ?)", (token, user["id"]))

        return {
            "token": token,
            "user": {
                "id": user["id"],
                "email": user["email"],
                "full_name": user["full_name"],
                "role": user["role"]
            },
            "message": "Signed in successfully."
        }
    finally:
        conn.close()


@router.get("/me")
def get_me(authorization: Optional[str] = Header(None)):
    user = get_current_user_from_token(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated or session expired.")
    return user


@router.post("/logout")
def logout(authorization: Optional[str] = Header(None)):
    if not authorization:
        return {"status": "ok"}
    token = authorization.replace("Bearer ", "").strip()
    conn = get_db()
    try:
        with conn:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        return {"status": "ok", "message": "Signed out successfully."}
    finally:
        conn.close()

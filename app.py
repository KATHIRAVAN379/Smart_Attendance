from flask import Flask, request, jsonify, send_from_directory, send_file
import os, base64, uuid, sqlite3
from datetime import datetime, date
from flask_cors import CORS
import numpy as np
import cv2
import face_recognition

# Import anti-spoofing (from your live_detection.py)
try:
    from live_detection import is_live_bytes
except ImportError:
    def is_live_bytes(image_bytes):
        # fallback: assume all are live
        return True

app = Flask(__name__)
CORS(app)

DB_PATH = "attendance.db"
IMAGE_DIR = "attendance_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# -------------------- DATABASE SETUP --------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS attendance (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    timestamp TEXT,
                    date TEXT,
                    image_path TEXT
                )''')
    conn.commit()
    conn.close()

init_db()

# -------------------- HELPERS --------------------
def has_face(image_bytes):
    """Check if at least one real face is detected."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return False
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb)
    return len(faces) > 0


def already_marked_today(name="Detected"):
    """Check if person already marked attendance today."""
    today = date.today().isoformat()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM attendance WHERE name=? AND date=?", (name, today))
    count = c.fetchone()[0]
    conn.close()
    return count > 0


# -------------------- ROUTES --------------------
@app.route("/mark_attendance", methods=["POST"])
def mark_attendance():
    data = request.get_json()
    image_base64 = data.get("image_base64")
    timestamp = data.get("timestamp", datetime.utcnow().isoformat())

    if not image_base64:
        return jsonify({"ok": False, "error": "No image received"}), 400

    try:
        image_bytes = base64.b64decode(image_base64)

        # Step 1: Check for real face
        if not has_face(image_bytes):
            return jsonify({"ok": False, "error": "No face detected"}), 403

        # Step 2: Liveness detection
        if not is_live_bytes(image_bytes):
            return jsonify({"ok": False, "error": "Spoof detected"}), 403

        # Step 3: Person name (default now, can be recognized later)
        name = "Unknown Person"

        today = date.today().isoformat()

        # Step 4: Prevent duplicate attendance for the same day
        if already_marked_today(name):
            return jsonify({"ok": False, "error": "Attendance already marked for today"}), 409

        # Step 5: Save attendance
        filename = f"{uuid.uuid4().hex}.jpg"
        image_path = os.path.join(IMAGE_DIR, filename)
        with open(image_path, "wb") as f:
            f.write(image_bytes)

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO attendance (id, name, timestamp, date, image_path) VALUES (?, ?, ?, ?, ?)",
                  (uuid.uuid4().hex, name, timestamp, today, image_path))
        conn.commit()
        conn.close()

        return jsonify({
            "ok": True,
            "timestamp": timestamp,
            "image_path": f"/attendance_images/{filename}",
            "name": name
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500



@app.route("/get_recent_attendance", methods=["GET"])
def get_recent_attendance():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT name, timestamp, image_path FROM attendance ORDER BY timestamp DESC LIMIT 10")
        rows = c.fetchall()
        conn.close()
        result = [{"person_text": r[0], "timestamp": r[1], "image_path": f"/{r[2]}"} for r in rows]
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/attendance_images/<path:filename>")
def serve_image(filename):
    return send_from_directory(IMAGE_DIR, filename)


@app.route("/")
def home():
    return send_file("AI_Attendance_Dashboard.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

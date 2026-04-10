import os
import re
import sys
import sqlite3
from io import BytesIO
from pathlib import Path

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from typing import Dict

# Make sure local venv packages are available even if app is launched
# with a different interpreter (common in IDEs/Windows shortcuts).
venv_root = Path(__file__).resolve().parent / ".venv"
candidate_sites = [
    venv_root / "Lib" / "site-packages",  # Windows
]
unix_site = next(
    (p for p in (venv_root / "lib").glob("python*/site-packages")),
    None,
)
if unix_site:
    candidate_sites.append(unix_site)
for site in candidate_sites:
    if site and site.exists() and str(site) not in sys.path:
        sys.path.insert(0, str(site))
from model import (
    FRAUD_KEYWORDS,
    preprocess_text,
    predict_fraud,
    predict_fraud_details,
    validate_job_post,
    structured_features,
    rule_based_score,
)
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Optional Gemini import (new package). Kept optional so app still runs without it.
try:
    from google import genai  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    genai = None

app = Flask(__name__)
app.secret_key = "supersecretkey"

DB_NAME = "database.db"
SLA_AVG_MS = 2200  # average processing time in milliseconds (placeholder)
SLA_UPTIME = 99.95  # uptime percentage (placeholder)


def load_env_file(path: str = ".env") -> None:
    """
    Minimal .env loader (avoids extra dependency).
    Supports KEY=value lines; ignores blanks and comments.
    Does not overwrite already-set environment variables.
    """
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key not in os.environ:
                os.environ[key] = value


# Load local .env (if present) so users can set GEMINI_API_KEY there.
load_env_file()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ---------------- Database Connection ----------------
def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------- Gemini Helper ----------------
def ask_gemini(prompt: str) -> str:
    """
    Sends the prompt to Google Gemini (if configured) and returns a plain-text answer.
    Errors are returned as readable strings so the UI can surface them.
    """
    if not GEMINI_API_KEY:
        return "Gemini API key not set. Add GEMINI_API_KEY to your environment."

    if genai:
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            return (getattr(response, "text", "") or "").strip() or "No response received."
        except Exception as exc:  # pragma: no cover - defensive
            return f"Gemini error: {exc}"

    return "Gemini client library missing. Install `pip install google-genai`."

# ---------------- Home ----------------
@app.route("/")
def home():
    return redirect(url_for("login"))

# ---------------- Register ----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    error = None
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        contact = request.form.get("contact")

        username = username.strip() if username else ""
        password = password.strip() if password else ""
        contact = contact.strip() if contact else ""

        email_pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
        # Accept any international number in E.164 style (7-15 digits, optional leading +)
        phone_pattern = r"^\+?[1-9]\d{7,14}$"

        if not username or not password or not contact:
            error = "All fields are required."
        elif not re.match(email_pattern, contact) and not re.match(phone_pattern, contact):
            error = "Enter a valid email or phone number (use +CountryCode)."

        if error:
            return render_template("pages/register.html", error=error)

        conn = get_db()
        try:
            conn.execute(
                "INSERT INTO users (username, contact, password) VALUES (?, ?, ?)",
                (username, contact, password)
            )
            conn.commit()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            error = "Username already exists."
        finally:
            conn.close()

    return render_template("pages/register.html", error=error)

# ---------------- Login ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        username = username.strip() if username else ""
        password = password.strip() if password else ""

        if not username or not password:
            error = "Enter both username and password."
            return render_template("pages/login.html", error=error)

        conn = get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (username, password)
        ).fetchone()
        conn.close()

        if user:
            session.clear()
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            flash(f"Welcome {username}!", "success")
            return redirect(url_for("dashboard"))
        else:
            error = "Invalid username or password."

    return render_template("pages/login.html", error=error)

# ---------------- Dashboard ----------------
@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        flash("Please login first!", "error")
        return redirect(url_for("login"))

    conn = get_db()
    jobs = conn.execute(
        """
        SELECT
            id,
            description,
            result,
            datetime(timestamp, 'localtime') AS ts_local
        FROM job_posts
        WHERE user_id=?
        ORDER BY timestamp DESC
        """,
        (session["user_id"],),
    ).fetchall()
    total = conn.execute(
        "SELECT COUNT(*) FROM job_posts WHERE user_id=? AND result IN ('Fake','Real')",
        (session["user_id"],),
    ).fetchone()[0]
    fraud = conn.execute(
        "SELECT COUNT(*) FROM job_posts WHERE user_id=? AND result='Fake'",
        (session["user_id"],),
    ).fetchone()[0]
    real = conn.execute(
        "SELECT COUNT(*) FROM job_posts WHERE user_id=? AND result='Real'",
        (session["user_id"],),
    ).fetchone()[0]

    # 30-day trend data
    trend_rows = conn.execute(
        """
        SELECT
            date(datetime(timestamp, 'localtime')) AS day,
            SUM(CASE WHEN result='Fake' THEN 1 ELSE 0 END) AS fake_count,
            SUM(CASE WHEN result='Real' THEN 1 ELSE 0 END) AS real_count
        FROM job_posts
        WHERE user_id=? AND result IN ('Fake','Real') AND timestamp >= datetime('now', '-30 day')
        GROUP BY day
        ORDER BY day ASC
        """,
        (session["user_id"],),
    ).fetchall()

    # Top keyword signals (last 30 days)
    desc_rows = conn.execute(
        """
        SELECT description, result
        FROM job_posts
        WHERE user_id=? AND result IN ('Fake','Real') AND timestamp >= datetime('now', '-30 day')
        """,
        (session["user_id"],),
    ).fetchall()
    conn.close()

    # Build trend arrays for chart
    trend_labels = [row["day"] for row in trend_rows]
    trend_fake = [row["fake_count"] for row in trend_rows]
    trend_real = [row["real_count"] for row in trend_rows]

    # Count top keywords
    keyword_counts: Dict[str, int] = {kw: 0 for kw in FRAUD_KEYWORDS}
    for row in desc_rows:
        cleaned = preprocess_text(row["description"])
        for kw in FRAUD_KEYWORDS:
            if kw in cleaned:
                keyword_counts[kw] += 1
    # Filter non-zero and take top 6
    # Scatter points: word count vs rule score
    scatter_points = []
    # Heatmap prep
    signal_keys = [
        ("link_count", "Links"),
        ("digit_count", "Digits"),
        ("has_crypto", "Crypto"),
        ("has_payment", "Fees"),
    ]
    signal_stats = {k: {"total": 0, "fake": 0} for k, _ in signal_keys}

    for row in desc_rows:
        feats = structured_features(row["description"])
        rule_score = rule_based_score(preprocess_text(row["description"]))
        scatter_points.append(
            {
                "x": feats["length_words"],
                "y": rule_score,
                "label": row["result"],
            }
        )
        for key, _ in signal_keys:
            hit = 0
            if key == "digit_count":
                hit = 1 if feats["digit_count"] > 0 else 0
            elif key == "link_count":
                hit = 1 if feats["link_count"] > 0 else 0
            else:
                hit = feats.get(key, 0)
            if hit:
                signal_stats[key]["total"] += 1
                if row["result"] == "Fake":
                    signal_stats[key]["fake"] += 1

    heatmap_cells = []
    for key, label in signal_keys:
        total_hits = signal_stats[key]["total"]
        fake_hits = signal_stats[key]["fake"]
        rate = (fake_hits / total_hits * 100) if total_hits else 0
        heatmap_cells.append({"signal": label, "rate": round(rate, 1)})

    accuracy = round((real / (real + fraud)) * 100) if (real + fraud) > 0 else 0

    return render_template(
        "pages/dashboard.html",
        jobs=jobs,
        total=total,
        fraud=fraud,
        real=real,
        accuracy=accuracy,
        trend_labels=trend_labels,
        trend_fake=trend_fake,
        trend_real=trend_real,
        sla_avg_ms=SLA_AVG_MS,
        sla_uptime=SLA_UPTIME,
        scatter_points=scatter_points,
        heatmap_cells=heatmap_cells,
    )

# ---------------- Submit Job ----------------
@app.route("/submit-job", methods=["GET", "POST"])
def submit_job():
    if "user_id" not in session:
        flash("You must login first!", "error")
        return redirect("/login")

    job_text = ""
    result = None

    if request.method == "POST":
        job_text = request.form.get("job_text")
        if not job_text:
            flash("Job description cannot be empty!", "error")
            return render_template("pages/submit_job.html")

        # Stage 1: validate job-ness before fraud detection
        job_validation = validate_job_post(job_text)
        if not job_validation.get("is_job", False):
            # Store as a distinct class so dashboard/timeline stay in sync
            conn = get_db()
            conn.execute(
                "INSERT INTO job_posts(user_id, description, result) VALUES (?, ?, ?)",
                (session["user_id"], job_text, "NotJob")
            )
            conn.commit()
            conn.close()

            flash("This doesn't look like a job post. Logged as 'Not a Job Post'.", "error")
            return render_template(
                "pages/result.html",
                result="Not a Job Post",
                job_text=job_text,
                details={**job_validation, "model_info": None},
            )

        # Stage 2: fraud classification
        details = predict_fraud_details(job_text)
        raw_label = details.get("label", "Unknown")
        # Map to display-friendly labels while keeping model semantics
        result_label = "Fake Job Post" if raw_label == "Fake" else "Real Job Post"

        conn = get_db()
        conn.execute(
            "INSERT INTO job_posts(user_id, description, result) VALUES (?, ?, ?)",
            (session["user_id"], job_text, raw_label)
        )
        conn.commit()
        conn.close()

        # Show detailed result page after submission
        return render_template(
            "pages/result.html",
            result=result_label,
            job_text=job_text,
            details=details,
        )

    return render_template("pages/submit_job.html")

# ---------------- Submit Feedback ----------------
@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    if "user_id" not in session:
        flash("Please login first!", "error")
        return redirect(url_for("login"))

    success = None
    if request.method == "POST":
        message = request.form.get("message")
        if message:
            conn = get_db()
            conn.execute(
                "INSERT INTO feedback(user_id, message) VALUES (?, ?)",
                (session["user_id"], message)
            )
            conn.commit()
            conn.close()
            success = "Thank you for your feedback!"
        else:
            flash("Feedback cannot be empty!", "error")

    return render_template("pages/feedback.html", success=success)


# ---------------- Gemini Chat ----------------
@app.route("/chat", methods=["GET", "POST"])
def chat():
    if "user_id" not in session:
        flash("Please login first!", "error")
        return redirect(url_for("login"))

    answer = None
    error = None

    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if not question:
            flash("Please enter a question.", "error")
        else:
            reply = ask_gemini(question)
            # Surface configuration/runtime issues as errors
            if reply.startswith("Gemini client") or reply.startswith("Gemini API"):
                error = reply
            elif reply.startswith("Gemini error"):
                error = reply
            else:
                answer = reply

    return render_template("pages/chat.html", answer=answer, error=error)


# ---------------- Generate PDF Report ----------------
@app.route("/report")
def generate_report():
    if "user_id" not in session:
        flash("Please login first!", "error")
        return redirect(url_for("login"))

    conn = get_db()
    jobs = conn.execute(
        "SELECT description, result, timestamp FROM job_posts WHERE user_id=?",
        (session["user_id"],)
    ).fetchall()
    conn.close()

    # Create PDF in memory
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    pdf.setFont("Helvetica", 12)
    y = height - 40

    pdf.drawString(50, y, f"Job Submissions Report for {session['username']}")
    y -= 30
    if not jobs:
        pdf.drawString(50, y, "No job submissions found.")
    else:
        for job in jobs:
            pdf.drawString(50, y, f"Job: {job['description'][:50]}{'...' if len(job['description'])>50 else ''}")
            y -= 20
            pdf.drawString(70, y, f"Result: {job['result']} | Submitted: {job['timestamp']}")
            y -= 30
            if y < 50:  # new page
                pdf.showPage()
                pdf.setFont("Helvetica", 12)
                y = height - 40

    pdf.save()
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="job_report.pdf", mimetype="application/pdf")

# ---------------- Logout ----------------
@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))

# ---------------- Run App ----------------
if __name__ == "__main__":
    app.run(debug=True)

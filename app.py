import json
import math
import os
import pickle
import random
import sqlite3
from datetime import datetime

import pandas as pd
from flask import Flask, flash, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

app = Flask(__name__)
app.secret_key = "atm_fraud_secret_key_123"

DATABASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database.db")
MODEL_FILE = "model.pkl"
PREPROCESSOR_FILE = "preprocessor.pkl"
CONFIG_FILE = "model_config.json"

VALID_TRANSACTION_TYPES = ["ATM Withdrawal", "Cash Deposit"]
VALID_AUTH_METHODS = ["PIN", "Biometric", "OTP", "Card + PIN"]
VALID_ATM_LOCATION_PROFILES = ["Usual Area", "Nearby Area", "Out of Area"]


def get_db_connection():
    conn = sqlite3.connect(DATABASE, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            username TEXT,
            password TEXT NOT NULL,
            created_at TEXT
        )
    """)

    user_columns = {
        row["name"] for row in cursor.execute("PRAGMA table_info(users)").fetchall()
    }
    if "name" not in user_columns:
        cursor.execute("ALTER TABLE users ADD COLUMN name TEXT")
    if "email" not in user_columns:
        cursor.execute("ALTER TABLE users ADD COLUMN email TEXT")
    if "username" not in user_columns:
        cursor.execute("ALTER TABLE users ADD COLUMN username TEXT")
    if "created_at" not in user_columns:
        cursor.execute("ALTER TABLE users ADD COLUMN created_at TEXT")

    cursor.execute("""
        UPDATE users
        SET name = COALESCE(name, username),
            created_at = COALESCE(created_at, ?)
        WHERE name IS NULL OR created_at IS NULL
    """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),))

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            transaction_amount REAL,
            transaction_type TEXT,
            account_balance REAL,
            location TEXT,
            previous_fraudulent_activity INTEGER,
            daily_transaction_count INTEGER,
            transactions_last_1h INTEGER,
            card_type TEXT,
            transaction_distance REAL,
            authentication_method TEXT,
            risk_score REAL,
            prediction_label TEXT,
            fraud_probability REAL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    prediction_columns = {
        row["name"] for row in cursor.execute("PRAGMA table_info(predictions)").fetchall()
    }
    if "transactions_last_1h" not in prediction_columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN transactions_last_1h INTEGER")

    conn.commit()
    conn.close()


def load_pickle_file(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def load_json_file(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def clamp_for_model(value, max_value):
    return min(value, max_value)


def estimate_distance_from_profile(user_id, transaction_amount, transaction_type, atm_location_profile):
    seed = f"{user_id}:{transaction_amount}:{transaction_type}:{atm_location_profile}"
    rng = random.Random(seed)

    if atm_location_profile == "Usual Area":
        return round(rng.uniform(1.0, 10.0), 2)
    if atm_location_profile == "Nearby Area":
        return round(rng.uniform(12.0, 55.0), 2)
    if transaction_type == "Cash Deposit":
        return round(rng.uniform(45.0, 120.0), 2)
    if transaction_amount >= 20000:
        return round(rng.uniform(120.0, 240.0), 2)
    return round(rng.uniform(60.0, 180.0), 2)


def calculate_risk_score(
    transaction_amount,
    transaction_type,
    transaction_distance,
    authentication_method,
    failed_pin_attempts,
    transactions_last_1h,
    atm_location_profile,
):
    risk_score = 0.10

    if transaction_amount >= 20000:
        risk_score += 0.28
    elif transaction_amount >= 10000:
        risk_score += 0.18
    elif transaction_amount >= 5000:
        risk_score += 0.08

    if transaction_distance >= 150:
        risk_score += 0.18
    elif transaction_distance >= 60:
        risk_score += 0.10
    elif transaction_distance >= 20:
        risk_score += 0.05

    if transaction_type == "ATM Withdrawal":
        risk_score += 0.08
    elif transaction_type == "Cash Deposit":
        risk_score += 0.03

    if atm_location_profile == "Out of Area":
        risk_score += 0.10
    elif atm_location_profile == "Nearby Area":
        risk_score += 0.04

    if authentication_method == "OTP":
        risk_score += 0.10
    elif authentication_method == "Card + PIN":
        risk_score += 0.06
    elif authentication_method == "PIN":
        risk_score += 0.03
    elif authentication_method == "Biometric":
        risk_score -= 0.10

    if failed_pin_attempts >= 3:
        risk_score += 0.20
    elif failed_pin_attempts == 2:
        risk_score += 0.12
    elif failed_pin_attempts == 1:
        risk_score += 0.05

    if transactions_last_1h >= 6:
        risk_score += 0.22
    elif transactions_last_1h >= 4:
        risk_score += 0.14
    elif transactions_last_1h >= 2:
        risk_score += 0.06

    return round(min(max(risk_score, 0.01), 0.99), 4)


preprocessor = load_pickle_file(PREPROCESSOR_FILE)
model = load_pickle_file(MODEL_FILE)
model_config = load_json_file(CONFIG_FILE)


def is_logged_in():
    return "user_id" in session


def current_user():
    if not is_logged_in():
        return None

    try:
        conn = get_db_connection()
        user = conn.execute(
            "SELECT * FROM users WHERE id = ?",
            (session["user_id"],)
        ).fetchone()
        conn.close()
        if user:
            return user
    except sqlite3.Error:
        pass

    return {
        "id": session.get("user_id"),
        "name": session.get("user_name", "User"),
    }


def get_model_meta():
    if not model_config:
        return {
            "high_risk_cutoff": 0.7,
            "prediction_threshold": 0.75,
            "input_caps": {
                "Transaction_Amount": 500.0,
                "Transaction_Distance": 5000.0,
                "Failed_Transaction_Count_7d": 4.0,
                "Transactions_Last_1H": 4.0,
                "Risk_Score": 1.0,
            },
        }

    return {
        "high_risk_cutoff": float(model_config.get("high_risk_cutoff", 0.7)),
        "prediction_threshold": float(model_config.get("prediction_threshold", 0.75)),
        "input_caps": model_config.get("input_caps", {
            "Transaction_Amount": 500.0,
                "Transaction_Distance": 5000.0,
                "Failed_Transaction_Count_7d": 4.0,
                "Transactions_Last_1H": 4.0,
                "Risk_Score": 1.0,
        }),
    }


def refresh_model_artifacts():
    global preprocessor
    global model
    global model_config

    preprocessor = load_pickle_file(PREPROCESSOR_FILE)
    model = load_pickle_file(MODEL_FILE)
    model_config = load_json_file(CONFIG_FILE)


def build_dashboard_stats(history):
    total_checks = len(history)
    fraud_flags = sum(1 for item in history if item["prediction_label"] == "Fraud")
    review_flags = sum(1 for item in history if item["prediction_label"] == "Review")
    suspicious_flags = sum(1 for item in history if item["prediction_label"] == "Suspicious")
    safe_checks = sum(1 for item in history if item["prediction_label"] == "Safe")

    return {
        "total_checks": total_checks,
        "fraud_flags": fraud_flags,
        "review_flags": review_flags,
        "suspicious_flags": suspicious_flags,
        "safe_checks": safe_checks,
    }


def format_history_record(record):
    if not record:
        return None

    return {
        "id": record["id"],
        "transaction_amount": f"{float(record['transaction_amount'] or 0):,.2f}",
        "transaction_type": record["transaction_type"],
        "transactions_last_1h": int(record["transactions_last_1h"] or 0),
        "risk_score": f"{float(record['risk_score'] or 0):.2f}",
        "fraud_probability": f"{float(record['fraud_probability'] or 0):.2f}",
        "prediction_label": record["prediction_label"],
        "created_at": record["created_at"],
    }


def build_prediction_reasons(
    transaction_amount,
    transaction_distance,
    failed_pin_attempts,
    transactions_last_1h,
    atm_location_profile,
    authentication_method,
):
    reasons = []

    if atm_location_profile == "Out of Area":
        reasons.append("Out of area ATM usage")
    elif atm_location_profile == "Nearby Area":
        reasons.append("Nearby but not usual ATM area")

    if transaction_amount >= 10000:
        reasons.append("High transaction amount")
    elif transaction_amount >= 5000:
        reasons.append("Moderately high transaction amount")

    if transaction_distance >= 60:
        reasons.append("Long travel distance from usual zone")

    if failed_pin_attempts >= 2:
        reasons.append("Multiple failed PIN attempts")
    elif failed_pin_attempts == 1:
        reasons.append("Single failed PIN attempt")

    if transactions_last_1h >= 4:
        reasons.append("Frequent transactions in the last hour")
    elif transactions_last_1h >= 2:
        reasons.append("Repeated recent transactions")

    if authentication_method == "Biometric":
        reasons.append("Biometric authentication lowered overall risk")

    return reasons[:3]


def determine_prediction_label(probability_score, risk_score, model_meta):
    probability_threshold = float(model_meta["prediction_threshold"])
    high_risk_cutoff = float(model_meta["high_risk_cutoff"])

    if probability_score >= 0.97:
        return "Fraud"

    if probability_score >= 0.85 and risk_score >= max(high_risk_cutoff, 0.75):
        return "Fraud"

    if probability_score >= 0.70 or (probability_score >= probability_threshold and risk_score >= 0.60):
        return "Suspicious"

    if probability_score >= 0.55 or risk_score >= 0.55:
        return "Review"

    return "Safe"


@app.route("/")
def home():
    if is_logged_in():
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()

        if not name or not email or not password:
            flash("Please fill all fields.", "danger")
            return render_template("register.html")

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return render_template("register.html")

        hashed_password = generate_password_hash(password)
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            conn = get_db_connection()
            user_columns = get_user_table_columns()
            lookup_parts = []
            lookup_values = []

            if "email" in user_columns:
                lookup_parts.append("LOWER(COALESCE(email, '')) = ?")
                lookup_values.append(email)
            if "name" in user_columns:
                lookup_parts.append("LOWER(COALESCE(name, '')) = ?")
                lookup_values.append(name.lower())
            if "username" in user_columns:
                lookup_parts.append("LOWER(COALESCE(username, '')) = ?")
                lookup_values.append(name.lower())

            existing_user = None
            if lookup_parts:
                existing_user = conn.execute(
                    f"SELECT * FROM users WHERE {' OR '.join(lookup_parts)}",
                    tuple(lookup_values)
                ).fetchone()

            if existing_user:
                conn.close()
                flash("Email or username already registered. Please sign in.", "warning")
                return redirect(url_for("login"))

            insert_columns = ["name", "email", "password", "created_at"]
            insert_values = [name, email, hashed_password, created_at]
            if "username" in user_columns:
                insert_columns.append("username")
                insert_values.append(name)

            conn.execute(
                f"INSERT INTO users ({', '.join(insert_columns)}) VALUES ({', '.join(['?'] * len(insert_columns))})",
                tuple(insert_values)
            )
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            flash(f"Database error: {str(e)}", "danger")
            return render_template("register.html")

        flash("Registration successful. Please sign in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        identifier = request.form.get("identifier", "").strip()
        password = request.form.get("password", "").strip()

        if not identifier or not password:
            flash("Please enter email/username and password.", "danger")
            return render_template("login.html")

        try:
            conn = get_db_connection()
            user_columns = get_user_table_columns()
            lookup_parts = []
            lookup_values = []

            if "email" in user_columns:
                lookup_parts.append("LOWER(COALESCE(email, '')) = ?")
                lookup_values.append(identifier.lower())
            if "name" in user_columns:
                lookup_parts.append("LOWER(COALESCE(name, '')) = ?")
                lookup_values.append(identifier.lower())
            if "username" in user_columns:
                lookup_parts.append("LOWER(COALESCE(username, '')) = ?")
                lookup_values.append(identifier.lower())

            user = None
            if lookup_parts:
                user = conn.execute(
                    f"SELECT * FROM users WHERE {' OR '.join(lookup_parts)}",
                    tuple(lookup_values)
                ).fetchone()
            conn.close()
        except sqlite3.Error as e:
            flash(f"Database error: {str(e)}", "danger")
            return render_template("login.html")

        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["user_name"] = user["name"] or user["username"] or "User"
            flash("Login successful.", "success")
            return redirect(url_for("dashboard"))

        flash("Invalid email/username or password.", "danger")

    return render_template("login.html")


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if not is_logged_in():
        flash("Please sign in first.", "warning")
        return redirect(url_for("login"))

    refresh_model_artifacts()
    empty_stats = build_dashboard_stats([])
    form_values = {
        "transaction_amount": "",
        "transaction_type": "",
        "failed_pin_attempts": "",
        "transactions_last_1h": "",
        "atm_location_profile": VALID_ATM_LOCATION_PROFILES[0],
        "authentication_method": VALID_AUTH_METHODS[0],
    }

    if model is None or preprocessor is None or model_config is None:
        return render_template(
            "dashboard.html",
            user=current_user(),
            model_missing=True,
            transaction_types=VALID_TRANSACTION_TYPES,
            auth_methods=VALID_AUTH_METHODS,
            atm_location_profiles=VALID_ATM_LOCATION_PROFILES,
            dashboard_stats=empty_stats,
            recent_history=[],
            form_values=form_values,
        )

    if request.method == "POST":
        try:
            model_meta = get_model_meta()
            transaction_amount = float(request.form["transaction_amount"])
            transaction_type = request.form["transaction_type"]
            authentication_method = request.form["authentication_method"]
            failed_pin_attempts = int(request.form["failed_pin_attempts"])
            transactions_last_1h = int(request.form["transactions_last_1h"])
            atm_location_profile = request.form["atm_location_profile"]

            form_values = {
                "transaction_amount": request.form.get("transaction_amount", "").strip(),
                "transaction_type": transaction_type,
                "failed_pin_attempts": request.form.get("failed_pin_attempts", "").strip(),
                "transactions_last_1h": request.form.get("transactions_last_1h", "").strip(),
                "atm_location_profile": atm_location_profile,
                "authentication_method": authentication_method,
            }

            if transaction_amount < 0:
                raise ValueError("Transaction Amount cannot be negative")
            if failed_pin_attempts < 0:
                raise ValueError("Failed PIN Attempts cannot be negative")
            if transactions_last_1h < 0:
                raise ValueError("Transactions in Last 1 Hour cannot be negative")

            if transaction_type not in VALID_TRANSACTION_TYPES:
                raise ValueError("Invalid Transaction Type")
            if authentication_method not in VALID_AUTH_METHODS:
                raise ValueError("Invalid Authentication Method")
            if atm_location_profile not in VALID_ATM_LOCATION_PROFILES:
                raise ValueError("Invalid ATM Location Profile")

            transaction_distance = estimate_distance_from_profile(
                session["user_id"],
                transaction_amount,
                transaction_type,
                atm_location_profile
            )

            risk_score = calculate_risk_score(
                transaction_amount,
                transaction_type,
                transaction_distance,
                authentication_method,
                failed_pin_attempts,
                transactions_last_1h,
                atm_location_profile,
            )

            input_caps = model_meta["input_caps"]
            model_transaction_amount = clamp_for_model(
                transaction_amount,
                float(input_caps["Transaction_Amount"])
            )
            model_transaction_distance = clamp_for_model(
                transaction_distance,
                float(input_caps["Transaction_Distance"])
            )
            model_failed_pin_attempts = int(clamp_for_model(
                failed_pin_attempts,
                float(input_caps["Failed_Transaction_Count_7d"])
            ))
            model_transactions_last_1h = int(clamp_for_model(
                transactions_last_1h,
                float(input_caps["Transactions_Last_1H"])
            ))
            model_risk_score = clamp_for_model(
                risk_score,
                float(input_caps["Risk_Score"])
            )

            if (
                model_transaction_amount != transaction_amount
                or model_transaction_distance != transaction_distance
                or model_failed_pin_attempts != failed_pin_attempts
                or model_transactions_last_1h != transactions_last_1h
                or model_risk_score != risk_score
            ):
                flash(
                    "Some inputs were capped to the model's trained range to improve prediction reliability.",
                    "info"
                )

            amount_log = math.log1p(model_transaction_amount)
            distance_risk = model_transaction_distance * model_risk_score

            flash(
                f"System signals used: distance {transaction_distance} km, failed PIN attempts {failed_pin_attempts}, last 1 hour transactions {transactions_last_1h}, risk score {risk_score}, distance risk {round(distance_risk, 2)}.",
                "info"
            )

            input_df = pd.DataFrame({
                "Transaction_Amount": [model_transaction_amount],
                "Transaction_Type": [transaction_type],
                "Transaction_Distance": [model_transaction_distance],
                "Failed_Transaction_Count_7d": [model_failed_pin_attempts],
                "Transactions_Last_1H": [model_transactions_last_1h],
                "ATM_Location_Profile": [atm_location_profile],
                "Authentication_Method": [authentication_method],
                "Risk_Score": [model_risk_score],
                "Amount_Log": [amount_log],
                "Distance_Risk": [distance_risk]
            })

            transformed_input = preprocessor.transform(input_df)
            if hasattr(transformed_input, "toarray"):
                transformed_input = transformed_input.toarray()

            prob = float(model.predict_proba(transformed_input)[0][1])

            prediction = determine_prediction_label(prob, model_risk_score, model_meta)
            prediction_reasons = build_prediction_reasons(
                transaction_amount,
                transaction_distance,
                failed_pin_attempts,
                transactions_last_1h,
                atm_location_profile,
                authentication_method,
            )

            probability = round(prob * 100, 2)

            try:
                conn = get_db_connection()
                conn.execute("""
                    INSERT INTO predictions (
                        user_id, transaction_amount, transaction_type, transaction_distance,
                        authentication_method, transactions_last_1h, risk_score, prediction_label, fraud_probability, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session["user_id"],
                    transaction_amount,
                    transaction_type,
                    transaction_distance,
                    authentication_method,
                    transactions_last_1h,
                    risk_score,
                    prediction,
                    probability,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ))
                conn.commit()
                conn.close()
            except sqlite3.Error:
                flash(
                    "Prediction was generated, but saving to history is temporarily unavailable.",
                    "warning"
                )

            flash(
                f"Prediction: {prediction} | Fraud Probability: {probability}%",
                (
                    "prediction-fraud" if prediction == "Fraud"
                    else "prediction-suspicious" if prediction == "Suspicious"
                    else "prediction-review" if prediction == "Review"
                    else "prediction-safe"
                )
            )
            if prediction_reasons:
                flash(
                    "Reason: " + ", ".join(prediction_reasons) + ".",
                    "info"
                )
            return redirect(url_for("dashboard"))

        except Exception as e:
            flash(f"Input error: {str(e)}", "danger")
            return redirect(url_for("dashboard"))

    try:
        conn = get_db_connection()
        history = conn.execute("""
            SELECT * FROM predictions
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 10
        """, (session["user_id"],)).fetchall()
        conn.close()
    except sqlite3.Error:
        history = []
        flash(
            "Recent transaction history is temporarily unavailable, but predictions still work.",
            "warning"
        )

    dashboard_stats = build_dashboard_stats(history)
    recent_history = [format_history_record(item) for item in history]

    return render_template(
        "dashboard.html",
        user=current_user(),
        model_missing=False,
        transaction_types=VALID_TRANSACTION_TYPES,
        auth_methods=VALID_AUTH_METHODS,
        atm_location_profiles=VALID_ATM_LOCATION_PROFILES,
        dashboard_stats=dashboard_stats,
        recent_history=recent_history,
        form_values=form_values,
    )


@app.route("/predictions/<int:prediction_id>/delete", methods=["POST"])
def delete_prediction(prediction_id):
    if not is_logged_in():
        flash("Please sign in first.", "warning")
        return redirect(url_for("login"))

    try:
        conn = get_db_connection()
        deleted = conn.execute(
            "DELETE FROM predictions WHERE id = ? AND user_id = ?",
            (prediction_id, session["user_id"])
        )
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        flash(f"Delete failed: {str(e)}", "danger")
        return redirect(url_for("dashboard"))

    if deleted.rowcount:
        flash("Transaction deleted successfully.", "success")
    else:
        flash("Transaction not found or cannot be deleted.", "warning")

    return redirect(url_for("dashboard"))


def get_user_table_columns():
    conn = get_db_connection()
    columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(users)").fetchall()
    }
    conn.close()
    return columns


try:
    init_db()
except sqlite3.Error:
    pass


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("login"))


if __name__ == "__main__":
    init_db()
    app.run(debug=True)

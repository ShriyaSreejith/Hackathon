"""
pi_server.py - Raspberry Pi Flask API for triage orchestration.

Deploy on the Pi as:
    /home/sana/pi_server.py

The Pi does not run AI vision or LLM models. It forwards video and clinical
payloads to the Jetson server, optionally runs a local ESI Random Forest model,
and returns a combined triage response.
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from typing import Any

import requests
from flask import Flask, g, jsonify, request
from flask_cors import CORS


# Config
JETSON_URL = "http://144.167.208.135:6000"
PI_PORT = 5000
ESI_MODEL_PATH = "/home/sana/esi_model.pkl"

VIDEO_TIMEOUT_SECONDS = 30
LLM_TIMEOUT_SECONDS = 90
HEALTH_TIMEOUT_SECONDS = 5

ESI_LABELS = {
    1: "Immediate",
    2: "Emergent",
    3: "Urgent",
    4: "Less Urgent",
    5: "Non-Urgent",
}

LLM_FALLBACK = {
    "esi_level": 3,
    "esi_label": "Urgent",
    "reasoning": "LLM unavailable",
    "red_flags": [],
    "recommended_action": "Manual assessment required",
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("pi_server")

app = Flask(__name__)
CORS(app)


def load_esi_model() -> Any | None:
    if not os.path.exists(ESI_MODEL_PATH):
        log.warning("ESI model not found at %s; continuing without local model", ESI_MODEL_PATH)
        return None

    try:
        with open(ESI_MODEL_PATH, "rb") as model_file:
            model = pickle.load(model_file)
        log.info("Loaded ESI model from %s", ESI_MODEL_PATH)
        return model
    except Exception as exc:
        log.warning("Failed to load ESI model from %s: %s", ESI_MODEL_PATH, exc)
        return None


esi_model = load_esi_model()


@app.before_request
def log_request_start() -> None:
    g.request_started_at = time.perf_counter()
    log.info(
        "request start method=%s path=%s remote_addr=%s",
        request.method,
        request.path,
        request.remote_addr,
    )


@app.after_request
def log_request_end(response):
    elapsed = time.perf_counter() - getattr(g, "request_started_at", time.perf_counter())
    log.info(
        "request end method=%s path=%s status=%s elapsed_seconds=%.3f",
        request.method,
        request.path,
        response.status_code,
        elapsed,
    )
    return response


def parse_int_form(name: str, *, required: bool = False, default: int | None = None) -> int | None:
    value = request.form.get(name)
    if value in (None, ""):
        if required:
            raise ValueError(f"missing required field '{name}'")
        return default

    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"field '{name}' must be an integer") from exc


def parse_float_form(name: str, *, default: float) -> float:
    value = request.form.get(name)
    if value in (None, ""):
        return default

    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"field '{name}' must be a number") from exc


def parse_int_json(data: dict[str, Any], name: str, *, required: bool = False, default: int | None = None) -> int | None:
    value = data.get(name)
    if value in (None, ""):
        if required:
            raise ValueError(f"missing required field '{name}'")
        return default

    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"field '{name}' must be an integer") from exc


def parse_float_json(data: dict[str, Any], name: str, *, default: float) -> float:
    value = data.get(name)
    if value in (None, ""):
        return default

    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"field '{name}' must be a number") from exc


def normalize_esi_level(value: Any, default: int = 3) -> int:
    try:
        level = int(value)
    except (TypeError, ValueError):
        return default
    return level if level in ESI_LABELS else default


def label_for_level(level: int) -> str:
    return ESI_LABELS.get(level, "Urgent")


def ensure_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return [str(value)]


def normalize_llm_response(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        data = LLM_FALLBACK

    level = normalize_esi_level(data.get("esi_level"))
    return {
        "esi_level": level,
        "esi_label": str(data.get("esi_label") or label_for_level(level)),
        "reasoning": str(data.get("reasoning") or ""),
        "red_flags": ensure_string_list(data.get("red_flags")),
        "recommended_action": str(data.get("recommended_action") or ""),
    }


def llm_reasoning_payload(llm_response: dict[str, Any]) -> dict[str, Any]:
    return {
        "esi_level": int(llm_response["esi_level"]),
        "reasoning": llm_response["reasoning"],
        "red_flags": llm_response["red_flags"],
        "recommended_action": llm_response["recommended_action"],
    }


def encode_chief_complaint(chief_complaint: str) -> int:
    text = chief_complaint.lower()
    if not text:
        return 0
    if "chest" in text or "heart" in text:
        return 1
    if "breath" in text or "shortness" in text or "asthma" in text:
        return 2
    if "stroke" in text or "weakness" in text or "confusion" in text:
        return 3
    if "pain" in text:
        return 4
    if "fever" in text or "infection" in text:
        return 5
    return 0


def build_model_features(
    vitals: dict[str, Any],
    distress: dict[str, Any],
    chief_complaint: str,
) -> list[list[float]]:
    return [[
        float(vitals["age"]),
        float(vitals["pulse"]),
        float(vitals["spo2"]),
        float(vitals["temp_c"]),
        float(vitals["respiratory_rate"]),
        float(vitals["systolic_bp"]),
        float(vitals["diastolic_bp"]),
        float(distress["score"]),
        float(encode_chief_complaint(chief_complaint)),
    ]]


def run_local_esi_model(
    vitals: dict[str, Any],
    distress: dict[str, Any],
    chief_complaint: str,
) -> dict[str, Any] | None:
    if esi_model is None:
        return None

    features = build_model_features(vitals, distress, chief_complaint)

    try:
        level = normalize_esi_level(esi_model.predict(features)[0])
        confidence = None
        if hasattr(esi_model, "predict_proba"):
            probabilities = esi_model.predict_proba(features)[0]
            confidence = round(float(max(probabilities)), 3)

        return {
            "level": level,
            "label": label_for_level(level),
            "confidence": confidence,
            "source": "model",
        }
    except Exception as exc:
        log.warning("Local ESI model prediction failed: %s", exc)
        return None


def choose_esi(llm_response: dict[str, Any], model_result: dict[str, Any] | None) -> dict[str, Any]:
    llm_level = normalize_esi_level(llm_response.get("esi_level"))
    llm_unavailable = llm_response.get("reasoning") == LLM_FALLBACK["reasoning"]

    if model_result and not llm_unavailable:
        model_level = normalize_esi_level(model_result.get("level"))
        combined_level = min(model_level, llm_level)
        return {
            "level": combined_level,
            "label": label_for_level(combined_level),
            "confidence": model_result.get("confidence") if combined_level == model_level else None,
            "source": "combined",
        }

    if model_result:
        model_level = normalize_esi_level(model_result.get("level"))
        return {
            "level": model_level,
            "label": label_for_level(model_level),
            "confidence": model_result.get("confidence"),
            "source": "model",
        }

    return {
        "level": llm_level,
        "label": str(llm_response.get("esi_label") or label_for_level(llm_level)),
        "confidence": None,
        "source": "llm",
    }


def post_video_to_jetson(video_file) -> dict[str, Any]:
    filename = video_file.filename or "patient_video.mp4"
    content_type = video_file.content_type or "video/mp4"
    video_file.stream.seek(0)

    response = requests.post(
        f"{JETSON_URL}/process/video",
        files={"video": (filename, video_file.stream, content_type)},
        timeout=VIDEO_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Jetson /process/video returned non-object JSON")
    return data


def post_llm_to_jetson(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        response = requests.post(
            f"{JETSON_URL}/analyze/llm",
            json=payload,
            timeout=LLM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return normalize_llm_response(response.json())
    except Exception as exc:
        log.warning("Jetson /analyze/llm failed: %s", exc)
        return normalize_llm_response(LLM_FALLBACK)


def build_llm_payload(
    vitals: dict[str, Any],
    distress: dict[str, Any],
    chief_complaint: str,
) -> dict[str, Any]:
    return {
        "age": vitals["age"],
        "distress_score": distress["score"],
        "active_aus": distress["active_aus"],
        "respiratory_rate": vitals["respiratory_rate"],
        "systolic_bp": vitals["systolic_bp"],
        "diastolic_bp": vitals["diastolic_bp"],
        "spo2": vitals["spo2"],
        "pulse": vitals["pulse"],
        "temp_c": vitals["temp_c"],
        "chief_complaint": chief_complaint,
    }


def build_response(
    start_time: float,
    vitals: dict[str, Any],
    distress: dict[str, Any],
    chief_complaint: str,
    llm_response: dict[str, Any],
) -> dict[str, Any]:
    model_result = run_local_esi_model(vitals, distress, chief_complaint)
    esi = choose_esi(llm_response, model_result)

    return {
        "esi": {
            "level": int(esi["level"]),
            "label": esi["label"],
            "confidence": esi["confidence"],
            "source": esi["source"],
        },
        "llm_reasoning": llm_reasoning_payload(llm_response),
        "vitals": {
            "age": int(vitals["age"]),
            "pulse": int(vitals["pulse"]),
            "spo2": float(vitals["spo2"]),
            "temp_c": float(vitals["temp_c"]),
            "respiratory_rate": int(vitals["respiratory_rate"]),
            "systolic_bp": int(vitals["systolic_bp"]),
            "diastolic_bp": int(vitals["diastolic_bp"]),
        },
        "distress": {
            "score": int(distress["score"]),
            "active_aus": distress["active_aus"],
            "face_detected": bool(distress["face_detected"]),
        },
        "elapsed_seconds": round(time.perf_counter() - start_time, 3),
    }


def estimated_birth_date_from_age(age: int, hl7_timestamp: str) -> str:
    current_year = int(hl7_timestamp[:4])
    birth_year = max(current_year - int(age), 1900)
    return f"{birth_year:04d}0101"


def add_obx(message, set_id: int, value_type: str, observation_id: str, value: Any, units: str = "") -> None:
    from hl7apy.consts import VALIDATION_LEVEL
    from hl7apy.core import Segment

    obx = Segment("OBX", version="2.5", validation_level=VALIDATION_LEVEL.TOLERANT)
    obx.obx_1 = str(set_id)
    obx.obx_2 = value_type
    obx.obx_3 = observation_id
    obx.obx_5 = str(value)
    if units:
        obx.obx_6 = units
    obx.obx_11 = "F"
    message.add(obx)


def generate_hl7(
    first_name,
    last_name,
    age,
    weight_kg,
    height_cm,
    pulse,
    spo2,
    temp_c,
    systolic_bp,
    diastolic_bp,
    respiratory_rate,
    esi_level,
    esi_label,
    reasoning,
):
    from hl7apy.consts import VALIDATION_LEVEL
    from hl7apy.core import Message, Segment

    now = time.time()
    hl7_timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime(now))
    file_timestamp = f"{hl7_timestamp}{int((now % 1) * 1000):03d}"
    file_path = f"/tmp/patient_{file_timestamp}.hl7"
    message_id = f"TRIAGE{file_timestamp}"

    message = Message("ORU_R01", version="2.5", validation_level=VALIDATION_LEVEL.TOLERANT)
    message.msh.msh_3 = "PI_TRIAGE"
    message.msh.msh_4 = "RASPBERRY_PI"
    message.msh.msh_5 = "EHR"
    message.msh.msh_6 = "HOSPITAL"
    message.msh.msh_7 = hl7_timestamp
    message.msh.msh_9 = "ORU^R01^ORU_R01"
    message.msh.msh_10 = message_id
    message.msh.msh_11 = "P"
    message.msh.msh_12 = "2.5"

    pid = Segment("PID", version="2.5", validation_level=VALIDATION_LEVEL.TOLERANT)
    pid.pid_1 = "1"
    pid.pid_3 = f"{message_id}^^^PI_TRIAGE^MR"
    pid.pid_5 = f"{last_name}^{first_name}"
    pid.pid_7 = estimated_birth_date_from_age(int(age), hl7_timestamp)
    message.add(pid)

    obr = Segment("OBR", version="2.5", validation_level=VALIDATION_LEVEL.TOLERANT)
    obr.obr_1 = "1"
    obr.obr_3 = f"{message_id}^PI_TRIAGE"
    obr.obr_4 = "TRIAGE^Emergency triage assessment^L"
    obr.obr_7 = hl7_timestamp
    obr.obr_22 = hl7_timestamp
    obr.obr_25 = "F"
    message.add(obr)

    observations = [
        ("NM", "29463-7^Body weight^LN", weight_kg, "kg"),
        ("NM", "8302-2^Body height^LN", height_cm, "cm"),
        ("NM", "8867-4^Heart rate^LN", pulse, "/min"),
        ("NM", "59408-5^Oxygen saturation in Arterial blood by Pulse oximetry^LN", spo2, "%"),
        ("NM", "8310-5^Body temperature^LN", temp_c, "Cel"),
        ("NM", "8480-6^Systolic blood pressure^LN", systolic_bp, "mm[Hg]"),
        ("NM", "8462-4^Diastolic blood pressure^LN", diastolic_bp, "mm[Hg]"),
        ("NM", "9279-1^Respiratory rate^LN", respiratory_rate, "/min"),
        ("TX", "ESI^Emergency Severity Index and reasoning^L", f"Level {esi_level} ({esi_label}): {reasoning}", ""),
    ]

    for index, (value_type, observation_id, value, units) in enumerate(observations, start=1):
        add_obx(message, index, value_type, observation_id, value, units)

    hl7_message = message.to_er7()
    if not hl7_message.endswith("\r"):
        hl7_message += "\r"

    with open(file_path, "w", encoding="utf-8", newline="\r") as hl7_file:
        hl7_file.write(hl7_message)

    return file_path, hl7_message


@app.route("/health", methods=["GET"])
def health():
    jetson_status: dict[str, Any] = {
        "status": "unreachable",
        "url": JETSON_URL,
    }

    try:
        response = requests.get(f"{JETSON_URL}/health", timeout=HEALTH_TIMEOUT_SECONDS)
        jetson_status.update({
            "status": "ok" if response.ok else "error",
            "status_code": response.status_code,
        })
        try:
            jetson_status["response"] = response.json()
        except ValueError:
            jetson_status["response"] = response.text[:500]
    except Exception as exc:
        jetson_status["error"] = str(exc)

    return jsonify({
        "status": "ok" if jetson_status.get("status") == "ok" else "degraded",
        "pi": {
            "status": "ok",
            "port": PI_PORT,
            "esi_model_loaded": esi_model is not None,
            "esi_model_path": ESI_MODEL_PATH,
        },
        "jetson": jetson_status,
    })


@app.route("/assess", methods=["POST"])
def assess():
    start_time = time.perf_counter()
    video_file = request.files.get("video")
    if video_file is None:
        return jsonify({"error": "missing required MP4 upload field 'video'"}), 400

    try:
        systolic_bp = parse_int_form("systolic_bp", required=True)
        diastolic_bp = parse_int_form("diastolic_bp", required=True)
        spo2 = parse_float_form("spo2", default=98.0)
        pulse = parse_int_form("pulse", default=75)
        temp_c = parse_float_form("temp_c", default=37.0)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    chief_complaint = request.form.get("chief_complaint", "") or ""

    try:
        video_analysis = post_video_to_jetson(video_file)
        age = int(video_analysis["age"])
        respiratory_rate = int(video_analysis["respiratory_rate"])
        distress_score = int(video_analysis["distress_score"])
    except Exception as exc:
        log.warning("Jetson /process/video failed: %s", exc)
        return jsonify({"error": "Jetson video processing failed", "detail": str(exc)}), 502

    vitals = {
        "age": age,
        "pulse": int(pulse),
        "spo2": float(spo2),
        "temp_c": float(temp_c),
        "respiratory_rate": respiratory_rate,
        "systolic_bp": int(systolic_bp),
        "diastolic_bp": int(diastolic_bp),
    }
    distress = {
        "score": distress_score,
        "active_aus": ensure_string_list(video_analysis.get("active_aus")),
        "face_detected": bool(video_analysis.get("face_detected", False)),
    }

    llm_response = post_llm_to_jetson(build_llm_payload(vitals, distress, chief_complaint))
    return jsonify(build_response(start_time, vitals, distress, chief_complaint, llm_response))


@app.route("/assess/manual", methods=["POST"])
def assess_manual():
    start_time = time.perf_counter()
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "JSON object body required"}), 400

    try:
        vitals = {
            "age": int(parse_int_json(data, "age", required=True)),
            "pulse": int(parse_int_json(data, "pulse", default=75)),
            "spo2": float(parse_float_json(data, "spo2", default=98.0)),
            "temp_c": float(parse_float_json(data, "temp_c", default=37.0)),
            "respiratory_rate": int(parse_int_json(data, "respiratory_rate", required=True)),
            "systolic_bp": int(parse_int_json(data, "systolic_bp", required=True)),
            "diastolic_bp": int(parse_int_json(data, "diastolic_bp", required=True)),
        }
        distress = {
            "score": int(parse_int_json(data, "distress_score", default=0)),
            "active_aus": ensure_string_list(data.get("active_aus")),
            "face_detected": bool(data.get("face_detected", False)),
        }
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    chief_complaint = str(data.get("chief_complaint") or "")
    llm_response = post_llm_to_jetson(build_llm_payload(vitals, distress, chief_complaint))
    return jsonify(build_response(start_time, vitals, distress, chief_complaint, llm_response))


@app.route("/generate/hl7", methods=["POST"])
def generate_hl7_endpoint():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "JSON object body required"}), 400

    required_fields = [
        "first_name",
        "last_name",
        "age",
        "weight_kg",
        "height_cm",
        "pulse",
        "spo2",
        "temp_c",
        "systolic_bp",
        "diastolic_bp",
        "respiratory_rate",
        "esi_level",
        "esi_label",
        "reasoning",
    ]
    missing_fields = [field for field in required_fields if data.get(field) in (None, "")]
    if missing_fields:
        return jsonify({"error": "missing required fields", "fields": missing_fields}), 400

    try:
        file_path, hl7_content = generate_hl7(
            first_name=str(data["first_name"]),
            last_name=str(data["last_name"]),
            age=int(data["age"]),
            weight_kg=float(data["weight_kg"]),
            height_cm=float(data["height_cm"]),
            pulse=int(data["pulse"]),
            spo2=float(data["spo2"]),
            temp_c=float(data["temp_c"]),
            systolic_bp=int(data["systolic_bp"]),
            diastolic_bp=int(data["diastolic_bp"]),
            respiratory_rate=int(data["respiratory_rate"]),
            esi_level=int(data["esi_level"]),
            esi_label=str(data["esi_label"]),
            reasoning=str(data["reasoning"]),
        )
    except ImportError as exc:
        return jsonify({"error": "hl7apy is required to generate HL7", "detail": str(exc)}), 500
    except (TypeError, ValueError) as exc:
        return jsonify({"error": "invalid field value", "detail": str(exc)}), 400

    return jsonify({
        "file_path": file_path,
        "hl7": hl7_content,
    })


if __name__ == "__main__":
    log.info("Starting Pi triage server on port %s", PI_PORT)
    app.run(host="0.0.0.0", port=PI_PORT, debug=False)

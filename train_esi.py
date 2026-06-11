import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


MODEL_PATH = "/home/sana/esi_model.pkl"
N_RECORDS = 10000
RANDOM_SEED = 42


def generate_records(n_records):
    rng = np.random.default_rng(RANDOM_SEED)

    age = rng.integers(1, 96, size=n_records)
    pulse = rng.integers(40, 181, size=n_records)
    spo2 = rng.uniform(80, 100, size=n_records)
    temp_c = rng.uniform(34, 41, size=n_records)
    respiratory_rate = rng.integers(8, 41, size=n_records)
    systolic_bp = rng.integers(70, 221, size=n_records)
    diastolic_bp = rng.integers(40, 131, size=n_records)
    distress_score = rng.integers(0, 11, size=n_records)
    chief_complaint_encoded = rng.integers(0, 10, size=n_records)

    return np.column_stack([
        age,
        pulse,
        spo2,
        temp_c,
        respiratory_rate,
        systolic_bp,
        diastolic_bp,
        distress_score,
        chief_complaint_encoded,
    ])


def label_records(records):
    pulse = records[:, 1]
    spo2 = records[:, 2]
    temp_c = records[:, 3]
    respiratory_rate = records[:, 4]
    systolic_bp = records[:, 5]
    diastolic_bp = records[:, 6]
    distress_score = records[:, 7]

    labels = np.full(records.shape[0], 5, dtype=int)

    esi4 = (
        (pulse < 60)
        | (pulse > 100)
        | (spo2 < 97)
        | (temp_c < 36.0)
        | (temp_c > 37.8)
        | (respiratory_rate < 12)
        | (respiratory_rate > 20)
        | (systolic_bp > 140)
        | (diastolic_bp < 60)
        | (diastolic_bp > 90)
        | (distress_score >= 2)
    )
    labels[esi4] = 4

    esi3 = (
        (spo2 < 95)
        | (systolic_bp < 100)
        | (pulse > 110)
        | (temp_c > 38.5)
        | (distress_score >= 5)
    )
    labels[esi3] = 3

    esi2 = (
        (spo2 < 92)
        | (systolic_bp < 90)
        | (pulse > 130)
        | (temp_c > 39.5)
        | (distress_score >= 8)
    )
    labels[esi2] = 2

    esi1 = (
        (spo2 < 85)
        | (systolic_bp < 70)
        | (pulse > 150)
        | (respiratory_rate > 35)
    )
    labels[esi1] = 1

    return labels


def main():
    records = generate_records(N_RECORDS)
    labels = label_records(records)

    x_train, x_test, y_train, y_test = train_test_split(
        records,
        labels,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_SEED,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.4f}")

    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(model, model_file)

    print("ESI model saved to /home/sana/esi_model.pkl")


if __name__ == "__main__":
    main()

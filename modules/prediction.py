import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REQUIRED_COLUMNS = ["Department", "Semester", "Subject", "Marks", "Attendance"]


def load_training_data(df):
    """Validate and clean the dataset uploaded by the user."""
    if df is None:
        raise ValueError("Please upload a student dataset first.")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    clean_df = df.copy()
    clean_df["Marks"] = pd.to_numeric(clean_df["Marks"], errors="coerce")
    clean_df["Attendance"] = pd.to_numeric(clean_df["Attendance"], errors="coerce")
    clean_df["Semester"] = pd.to_numeric(clean_df["Semester"], errors="coerce")
    clean_df = clean_df.dropna(subset=REQUIRED_COLUMNS)

    if len(clean_df) < 2:
        raise ValueError("At least 2 valid student records are required for prediction.")

    if not clean_df["Marks"].between(0, 100).all():
        raise ValueError("Marks must be between 0 and 100.")
    if not clean_df["Attendance"].between(0, 100).all():
        raise ValueError("Attendance must be between 0 and 100.")

    return clean_df


def _build_model():
    features = ["Department", "Semester", "Subject", "Attendance"]
    categorical = ["Department", "Subject"]
    preprocessing = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical)
    ], remainder="passthrough")
    return Pipeline([
        ("preprocessing", preprocessing),
        ("regression", LinearRegression())
    ])


def predict_marks(department, semester, subject, attendance, df):
    df = load_training_data(df)
    features = ["Department", "Semester", "Subject", "Attendance"]
    X = df[features].copy()
    y = df["Marks"]
    model = _build_model()
    model.fit(X, y)
    student = pd.DataFrame([{
        "Department": department,
        "Semester": semester,
        "Subject": subject,
        "Attendance": attendance
    }])
    prediction = model.predict(student)[0]
    return round(max(0, min(100, float(prediction))), 2)


def evaluate_model(df):
    df = load_training_data(df)
    if len(df) < 8:
        raise ValueError("At least 8 valid student records are required for model evaluation.")

    features = ["Department", "Semester", "Subject", "Attendance"]
    X = df[features]
    y = df["Marks"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    model = _build_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    predictions = [round(max(0, min(100, float(p))), 2) for p in predictions]

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    comparison_df = pd.DataFrame({
        "Actual Marks": list(y_test),
        "Predicted Marks": predictions
    })
    return {
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "R2 Score": round(float(r2), 4),
        "Training Records": len(X_train),
        "Testing Records": len(X_test),
        "Comparison": comparison_df
    }

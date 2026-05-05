import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS: List[str] = [
    "Size (sqft)",
    "BHK",
    "Bathrooms",
    "Age of Property (years)",
    "Floor Number",
    "Total Floors",
    "Parking",
]
TARGET_COLUMN = "Price (INR)"


@dataclass(frozen=True)
class TrainResult:
    model: RandomForestRegressor
    mse: float
    r2: float


def load_data(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def make_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def train_and_evaluate(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
) -> TrainResult:
    X, y = make_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    return TrainResult(model=model, mse=mse, r2=r2)


def save_model(model: RandomForestRegressor, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Save both the estimator and the expected feature order for robust loading.
    dump({"model": model, "feature_columns": FEATURE_COLUMNS}, out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train RF model and save to joblib.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("maharashtra_house_prices.csv"),
        help="Path to maharashtra_house_prices.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("rf_model.pkl"),
        help="Output path for saved model (joblib)",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=100)
    args = parser.parse_args()

    df = load_data(args.data)
    result = train_and_evaluate(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
    )
    save_model(result.model, args.out)

    print(f"Saved model to: {args.out}")
    print(f"MSE: {result.mse:,.2f}")
    print(f"R2: {result.r2:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


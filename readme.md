# Hotel Booking Cancellation Predictor

## Project Overview

This project aims to predict whether a hotel booking will be canceled using machine learning models (Random Forest and XGBoost). It includes end-to-end data preprocessing, feature engineering, model training with hyperparameter tuning, model serialization, and a Streamlit web application for interactive predictions.

## Table of Contents

1. [Dataset](#dataset)
2. [Feature Engineering &amp; Preprocessing](#feature-engineering--preprocessing)
3. [Model Training](#model-training)
4. [Hyperparameter Tuning](#hyperparameter-tuning)
5. [Model Export](#model-export)
6. [Streamlit Application](#streamlit-application)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Project Structure](#project-structure)
10. [Contributing](#contributing)
11. [License](#license)

---

## Dataset

- The dataset (`Dataset_B_hotel.csv`) contains 36,275 records with the following columns:
  - `Booking_ID`, `no_of_adults`, `no_of_children`, `no_of_weekend_nights`, `no_of_week_nights`,
    `type_of_meal_plan`, `required_car_parking_space`, `room_type_reserved`, `lead_time`,
    `arrival_date_dt`, `market_segment_type`, `repeated_guest`, `no_of_previous_cancellations`,
    `no_of_previous_bookings_not_canceled`, `avg_price_per_room`, `no_of_special_requests`,
    `booking_status` (target).

## Feature Engineering & Preprocessing

- **Missing Value Imputation**
  - `type_of_meal_plan`: filled with the mode.
  - `required_car_parking_space`: missing assumed as 0.
  - `avg_price_per_room`: filled with median price.
- **Derived Features**
  - `total_nights` = `no_of_weekend_nights` + `no_of_week_nights`
  - `total_guests` = `no_of_adults` + `no_of_children`
  - Extracted `arrival_month` and `arrival_weekday` from `arrival_date_dt`.
- **Preprocessing Pipeline**
  - Numeric features: median imputation.
  - Categorical features: most-frequent imputation + one-hot encoding.

## Model Training

- **Train/Test Split**: 80% training, 20% testing, stratified on `booking_status`.
- **Models**:
  - Random Forest (`sklearn.ensemble.RandomForestClassifier`)
  - XGBoost (`xgboost.XGBClassifier`)

## Hyperparameter Tuning

- `RandomizedSearchCV` with 3-fold CV optimizing ROC AUC.
- **Random Forest parameters**:
  - `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `bootstrap`.
- **XGBoost parameters**:
  - `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`.

## Model Export

- The best XGBoost pipeline (including preprocessing) is serialized to `best_xgb_model.pkl` using `pickle`.

## Streamlit Application

- **File**: `streamlit_app.py`
- **Features**:
  - Interactive form to input booking details.
  - Displays cancellation probability and binary prediction.
- **Run**:
  ```bash
  # Install dependencies
  pip install streamlit scikit-learn xgboost pandas

  # Launch app
  streamlit run streamlit_app.py
  ```

## Installation

```bash
git clone <repo_url>
cd hotel-cancellation-predictor
pip install -r requirements.txt
```

## Usage

* **Notebook** : Follow the Jupyter notebook for data exploration, model training, and evaluation.
* **Streamlit** : Launch the web app as described above for real-time predictions.

## Project Structure

```bash
├── Dataset_B_hotel.csv
├── notebook.ipynb           # EDA, preprocessing, modeling
├── best_xgb_model.pkl       # Serialized pipeline
├── streamlit_app.py         # Streamlit application
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements.

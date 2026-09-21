# 🚕 Uber Fare Prediction using Artificial Neural Network

An end-to-end **Deep Learning regression project** that predicts Uber trip fares using an **Artificial Neural Network (ANN)** built with TensorFlow/Keras.

The project includes data cleaning, feature engineering, geographical distance calculation using the **Haversine formula**, feature and target scaling, ANN model development, hyperparameter experimentation, early stopping, and model evaluation.

---

## 📌 Project Overview

Uber fares depend on several factors such as:

* Trip distance
* Pickup and drop-off locations
* Passenger count
* Time of the trip
* Day of the week
* Month and year

This project uses these features to train a neural network that predicts the expected **Uber fare amount** for a trip.

### Problem Type

**Supervised Learning → Regression**

### Target Variable

`fare_amount`

---

## 📊 Dataset

The project uses the **Uber Fares Dataset** containing approximately 200,000 trip records.

### Features

| Feature             | Description                            |
| ------------------- | -------------------------------------- |
| `pickup_longitude`  | Pickup longitude                       |
| `pickup_latitude`   | Pickup latitude                        |
| `dropoff_longitude` | Drop-off longitude                     |
| `dropoff_latitude`  | Drop-off latitude                      |
| `passenger_count`   | Number of passengers                   |
| `Year`              | Year of the trip                       |
| `Month`             | Month of the trip                      |
| `Day`               | Day of the month                       |
| `Hour`              | Hour of the trip                       |
| `Day_of_Week`       | Day of the week                        |
| `trip_distance`     | Calculated trip distance in kilometers |

### Target

`fare_amount`

---

## 🔄 Project Workflow

```text
Uber Fare Dataset
        ↓
Data Cleaning
        ↓
Datetime Feature Extraction
        ↓
Haversine Distance Calculation
        ↓
Outlier Removal
        ↓
Train-Test Split
        ↓
Feature Scaling
        ↓
Target Scaling
        ↓
ANN Model
        ↓
Early Stopping
        ↓
Prediction
        ↓
Inverse Scaling
        ↓
Model Evaluation
```

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

### 1. Remove unnecessary columns

The `Unnamed: 0` column was removed when present.

The following columns were also removed after feature extraction:

* `pickup_datetime`
* `key`

### 2. Handle missing values

Rows containing missing values were removed.

### 3. Extract datetime features

The original `pickup_datetime` column was converted into separate temporal features:

```python
df["Year"] = df["pickup_datetime"].dt.year
df["Month"] = df["pickup_datetime"].dt.month
df["Day"] = df["pickup_datetime"].dt.day
df["Hour"] = df["pickup_datetime"].dt.hour
df["Day_of_Week"] = df["pickup_datetime"].dt.dayofweek
```

This allows the model to learn relationships between fare prices and trip timing.

---

## 📍 Trip Distance Calculation

Instead of directly using latitude and longitude values, the project calculates the geographical distance between pickup and drop-off locations using the **Haversine formula**.

```python
lat1 = np.radians(df["pickup_latitude"])
lon1 = np.radians(df["pickup_longitude"])
lat2 = np.radians(df["dropoff_latitude"])
lon2 = np.radians(df["dropoff_longitude"])

dlat = lat2 - lat1
dlon = lon2 - lon1

a = np.sin(dlat / 2) ** 2 + \
    np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2

c = 2 * np.arcsin(np.sqrt(a))

df["trip_distance"] = 6371 * c
```

The resulting `trip_distance` represents the approximate geographical distance between the pickup and drop-off coordinates in kilometers.

---

## 🚨 Outlier Handling

Several filtering rules were applied to remove unrealistic observations:

* Removed negative fares
* Removed fares above 200
* Removed trips with zero passengers
* Removed trips with more than 6 passengers
* Removed trips with distances greater than 100 km
* Removed trips with zero distance

This helps reduce the effect of extreme or invalid observations on model training.

---

## ⚙️ Feature Scaling

The input features were standardized using `StandardScaler`.

```python
x_scaler = StandardScaler()

X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)
```

The target variable was also standardized before training:

```python
y_scaler = StandardScaler()

y_train_scaled = y_scaler.fit_transform(
    y_train.to_numpy().reshape(-1, 1)
)

y_test_scaled = y_scaler.transform(
    y_test.to_numpy().reshape(-1, 1)
)
```

After prediction, the predicted values were converted back to the original fare scale using inverse transformation.

---

## 🧠 ANN Architecture

The model was developed using **TensorFlow/Keras Sequential API**.

```text
Input Layer
     ↓
Dense Layer (64 neurons, ReLU)
     ↓
Batch Normalization
     ↓
Dropout (20%)
     ↓
Dense Layer (32 neurons, ReLU)
     ↓
Batch Normalization
     ↓
Dropout (20%)
     ↓
Dense Layer (16 neurons, ReLU)
     ↓
Output Layer (1 neuron, Linear)
```

### Model Configuration

| Parameter               | Value          |
| ----------------------- | -------------- |
| Architecture            | Sequential ANN |
| Hidden Layers           | 3              |
| Hidden Units            | 64 → 32 → 16   |
| Activation              | ReLU           |
| Output Activation       | Linear         |
| Dropout                 | 0.2            |
| Optimizer               | Adam           |
| Learning Rate           | 0.001          |
| Loss Function           | MSE            |
| Metric                  | MAE            |
| Batch Size              | 64             |
| Maximum Epochs          | 50             |
| Early Stopping Patience | 5              |

---

## 🛑 Early Stopping

Early stopping was used to prevent unnecessary training and reduce overfitting.

```python
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
```

Training stops when validation loss does not improve for several consecutive epochs.

---

## 📈 Model Evaluation

The model is evaluated using three regression metrics:

### RMSE

Root Mean Squared Error measures the average magnitude of prediction errors while giving greater weight to larger errors.

### MAE

Mean Absolute Error represents the average absolute difference between actual and predicted fares.

### R² Score

R² measures how much of the variation in fare prices is explained by the model.

```python
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
```

The notebook prints the final test-set:

```text
RMSE
MAE
R²
```


## 💾 Saved Model

The trained model is saved in Keras format:

```text
uber_fare_ann_model.keras
```

This allows the trained ANN to be loaded later without retraining.

---

## 🛠️ Technologies Used

* **Python**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Scikit-learn**
* **TensorFlow**
* **Keras**

### Machine Learning Concepts

* Regression
* Artificial Neural Networks
* Feature Engineering
* Haversine Distance
* Feature Scaling
* Target Scaling
* Batch Normalization
* Dropout
* Early Stopping
* Model Evaluation

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/uber-fare-prediction-ann.git
cd uber-fare-prediction-ann
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

### 3. Run the notebook

Open:

```text
uber-fares-ann.ipynb
```

and run the cells sequentially.

---

## 🔮 Future Improvements

Possible improvements include:

* Hyperparameter optimization using Optuna
* Comparing ANN performance with Random Forest and XGBoost
* Adding a Streamlit prediction interface
* Deploying the trained model
* Using additional geographical features
* Experimenting with different neural network architectures
* Applying cross-validation for more robust evaluation

---

## 👩‍💻 Author

**Sree Harshita Morla**

B.Tech, Electronics and Communication Engineering
Sreenidhi Institute of Science and Technology

[LinkedIn](https://www.linkedin.com/in/harshita-morla-41b1b9268/)
[GitHub](https://github.com/harshita617)

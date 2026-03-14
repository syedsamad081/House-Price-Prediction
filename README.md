# House Price Prediction using Regression

This repository contains a machine learning pipeline developed in Python to predict property prices. The project utilizes structured housing data and implements multiple regression algorithms to identify the most accurate predictive model.



## 📌 Project Overview
The goal of this project is to analyze the factors influencing real estate prices and build a robust model for price estimation. The pipeline includes:
* **Data Preprocessing:** Handling anomalies (zero-price entries), feature extraction (house age, renovation status), and label encoding for categorical variables like city and state-zip.
* **Exploratory Data Analysis (EDA):** Visualizing feature correlations and price distributions to understand underlying patterns.
* **Model Implementation:** Comparing **Linear Regression** and **Random Forest Regressor** to determine the best fit for the data.
* **Evaluation:** Using standard metrics such as Root Mean Squared Error (**RMSE**) and the Coefficient of Determination (**$R^2$**).

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **Visualization:** Matplotlib, Seaborn

## 📊 Key Features & Engineering
To improve model performance, the following transformations were applied in `model.py`:
* **Feature Extraction:** Calculated `house_age` from `yr_built` and created a boolean `is_renovated` flag.
* **Data Cleaning:** Removed low-variance columns (street, country, date) and filtered out invalid price records.
* **Scaling:** Applied `StandardScaler` to normalize numerical features for the Linear Regression model.



## 🚀 How to Use
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/house-price-prediction.git](https://github.com/your-username/house-price-prediction.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
3.  **Add your data:** Place your dataset in the root folder and name it `data.csv`.
4.  **Run the model:**
    ```bash
    python model.py
    ```

## 📈 Results & Visualizations
The script automatically generates four key visualizations to the root directory:
* `correlation_heatmap.png`: Shows the relationship between features.
* `price_distribution.png`: Visualizes the target variable spread.
* `feature_importance.png`: Displays which factors (e.g., sqft_living, city) impact price the most.
* `actual_vs_predicted.png`: A scatter plot comparing the Random Forest predictions against real values.

---

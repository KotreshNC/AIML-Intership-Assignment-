import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    # Toy house dataset (square feet, number of bedrooms, age in years)
    data = {
        'sqft': [850, 900, 1200, 1500, 1700, 2100, 2300, 2500, 2900, 3200],
        'bedrooms': [2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
        'age': [10, 15, 8, 12, 5, 20, 7, 3, 2, 1],
        'price': [120000, 130000, 170000, 195000, 220000, 250000, 285000, 310000, 345000, 370000]
    }
    df = pd.DataFrame(data)

    print("House Pricing Dataset:")
    print(df)

    X = df[['sqft', 'bedrooms', 'age']]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    print("\nTrained Linear Regression model coefficients:")
    print(f"Intercept: {model.intercept_:.2f}")
    print(f"Coefficients: sqft={model.coef_[0]:.2f}, bedrooms={model.coef_[1]:.2f}, age={model.coef_[2]:.2f}")

    y_pred = model.predict(X_test)
    print("\nTest set predictions:")
    for i, (features, true_val, pred_val) in enumerate(zip(X_test.values, y_test.values, y_pred)):
        print(f"Row {i+1}: features={features.tolist()}, true={true_val:.0f}, pred={pred_val:.0f}")

    print("\nEvaluation metrics:")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"R2: {r2_score(y_test, y_pred):.4f}")

    # New input to predict price
    new_house = np.array([[1800, 3, 5], [2600, 4, 2], [1200, 2, 10]])
    print("\nNew house inputs (sqft, bedrooms, age):")
    print(new_house)
    new_preds = model.predict(new_house)
    print("Predicted prices for new inputs:")
    for features, p in zip(new_house, new_preds):
        print(f"  {features.tolist()} -> ${p:.0f}")


if __name__ == '__main__':
    main()

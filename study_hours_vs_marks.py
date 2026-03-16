import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def main():
    # Build a simple dataset: study hours vs marks
    data = {
        'study_hours': [1, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6],
        'marks': [35, 45, 50, 55, 60, 65, 70, 75, 78, 82]
    }
    df = pd.DataFrame(data)

    print("Dataset (features and label):")
    print(df)

    # Feature is study_hours, label is marks
    X = df[['study_hours']]
    y = df['marks']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nModel coefficients:")
    print(f"Intercept: {model.intercept_:.2f}")
    print(f"Coefficient (study_hours): {model.coef_[0]:.2f}")

    print("\nTest predictions:")
    for hours, true_mark, pred_mark in zip(X_test['study_hours'], y_test, y_pred):
        print(f"Study {hours} hrs -> true {true_mark:.0f}, predicted {pred_mark:.1f}")

    print("\nMetrics:")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"R2: {r2_score(y_test, y_pred):.4f}")

    print("\nPredicting new values:")
    for h in [1.5, 3.2, 4.8, 6.5]:
        pred = model.predict([[h]])[0]
        print(f"  {h} hours -> predicted marks {pred:.1f}")

    print("\nFeature = study_hours, Label = marks.")
    print("Relationship: More study hours correspond to higher marks in this linear model.")


if __name__ == '__main__':
    main()

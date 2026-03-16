import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def main():
    # Synthetic dataset for student performance
    # Inputs (features): attendance %, hours studied per week, assignment average, class participation (0-1)
    # Output (label): final exam score (out of 100)
    data = {
        'attendance': [80, 85, 90, 60, 70, 95, 88, 76, 92, 65, 82, 78, 96, 69, 84],
        'study_hours': [5, 7, 9, 2, 3, 10, 8, 4, 11, 3, 6, 5, 12, 2, 7],
        'assignment_avg': [72, 78, 88, 55, 60, 92, 85, 66, 94, 58, 79, 70, 97, 62, 80],
        'participation': [0.6, 0.7, 0.9, 0.3, 0.4, 0.95, 0.8, 0.5, 0.98, 0.35, 0.7, 0.55, 1.0, 0.38, 0.75],
        'final_score': [72, 79, 91, 52, 60, 95, 87, 64, 99, 56, 80, 68, 100, 58, 82]
    }
    df = pd.DataFrame(data)

    print("Dataset (features and label):")
    print(df)

    X = df[['attendance', 'study_hours', 'assignment_avg', 'participation']]
    y = df['final_score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nTrained model coefficients:")
    print(f"Intercept: {model.intercept_:.2f}")
    print("Coefficients:")
    for name, coef in zip(X.columns, model.coef_):
        print(f"  {name}: {coef:.2f}")

    print("\nTest predictions:")
    for features, true, pred in zip(X_test.values, y_test.values, y_pred):
        print(f"  Features={features.tolist()}, true={true:.1f}, pred={pred:.1f}")

    print("\nEvaluation:")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"R2: {r2_score(y_test, y_pred):.4f}")

    new_students = pd.DataFrame({
        'attendance': [90, 72, 85],
        'study_hours': [8, 3, 7],
        'assignment_avg': [88, 62, 80],
        'participation': [0.85, 0.4, 0.72]
    })
    new_preds = model.predict(new_students)
    print("\nNew predictions (input -> predicted final score):")
    for inp, out in zip(new_students.values, new_preds):
        print(f"  {inp.tolist()} -> {out:.1f}")

    print("\nFeature = [attendance, study_hours, assignment_avg, participation], label = final_score")
    print("This shows a predicted relationship from student habits to performance.")


if __name__ == '__main__':
    main()

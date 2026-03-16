import pandas as pd
import numpy as np
import time


def dataset_detective(data_path):
    # 1) Load dataset
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: file not found: {data_path}")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("\n===== Dataset Detective Report =====")
    print(f"Loaded dataset: {data_path}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # 2) Display top rows
    print("\n-- Top 5 rows --")
    print(df.head(5).to_string(index=False))

    # 3) Highest value column (numeric columns only)
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_columns) == 0:
        print("\nNo numeric columns to compute highest value.")
    else:
        max_values = df[numeric_columns].max(numeric_only=True)
        highest_col = max_values.idxmax()
        highest_val = max_values.max()
        print(f"\nColumn with highest single numeric value: {highest_col} = {highest_val}")

    # 4) Count missing values
    missing_per_column = df.isna().sum()
    total_missing = missing_per_column.sum()
    print(f"\nTotal missing values in dataset: {total_missing}")
    print("Missing values by column:")
    print(missing_per_column[missing_per_column > 0].to_string())

    # 5) 5 insights (simple heuristics)
    print("\n-- 5 automatic insights --")
    insights = []

    # Insight 1: row count and column count extremes
    if df.shape[0] > 10000:
        insights.append("Large dataset with more than 10,000 rows. Consider sampling for quick model experiments.")
    else:
        insights.append(f"Dataset is relatively small with {df.shape[0]} rows.")

    # Insight 2: missing values
    if total_missing == 0:
        insights.append("No missing values detected, so you can train models without missing-value imputation.")
    else:
        nonzero = missing_per_column[missing_per_column > 0].sort_values(ascending=False)
        top_missing = nonzero.index[0]
        ratio = nonzero.iloc[0] / df.shape[0]
        insights.append(f"Column '{top_missing}' has the most missing values ({nonzero.iloc[0]} missing, {ratio:.1%} of rows).")

    # Insight 3: numeric column range info
    if len(numeric_columns) > 0:
        means = df[numeric_columns].mean(numeric_only=True)
        top_mean = means.idxmax()
        insights.append(f"Numeric column '{top_mean}' has the highest average value ({means[top_mean]:.3f}).")
    else:
        insights.append("No numeric columns for statistical insights.")

    # Insight 4: unique value insights
    high_cardinality = []
    for c in df.columns:
        unique_count = df[c].nunique(dropna=False)
        ratio = unique_count / max(1, df.shape[0])
        if ratio > 0.8:
            high_cardinality.append(c)
    if len(high_cardinality) > 0:
        insights.append(f"Columns with high cardinality (mostly unique values): {', '.join(high_cardinality[:3])}.")
    else:
        insights.append("No columns with extremely high cardinality found.")

    # Insight 5: dtypes
    dtype_counts = df.dtypes.value_counts()
    dtypes_summary = ", ".join([f"{dtype}:{count}" for dtype, count in dtype_counts.items()])
    insights.append(f"Column types summary: {dtypes_summary}.")

    for i, insight in enumerate(insights[:5], start=1):
        print(f"{i}. {insight}")


def numpy_speed_test(n=1_000_000):
    print("\n===== NumPy Speed Test =====")
    # Prepare data
    py_list = list(range(n))
    np_array = np.arange(n)

    # Python list sum
    t0 = time.perf_counter()
    total_list = sum(py_list)
    t1 = time.perf_counter()
    list_time = t1 - t0
    print(f"Python list sum ({n} elements): {total_list} (time: {list_time:.6f}s)")

    # NumPy array sum
    t2 = time.perf_counter()
    total_np = np_array.sum()
    t3 = time.perf_counter()
    np_time = t3 - t2
    print(f"NumPy array sum ({n} elements): {total_np} (time: {np_time:.6f}s)")

    # Python list comprehension add
    t4 = time.perf_counter()
    _ = [x + 1 for x in py_list]
    t5 = time.perf_counter()
    list_op_time = t5 - t4

    # NumPy vectorized add
    t6 = time.perf_counter()
    _ = np_array + 1
    t7 = time.perf_counter()
    np_op_time = t7 - t6

    print(f"Python list add 1 (comprehension): {list_op_time:.6f}s")
    print(f"NumPy vectorized add 1: {np_op_time:.6f}s")

    print("\nObservations:")
    print("1. For large numeric arrays, NumPy’s vectorized operations are significantly faster than Python loops/list comprehensions.")
    print("2. NumPy’s built-in reductions (sum, mean) generally outperform Python’s built-in sum on lists for large data.")
    print("3. NumPy uses contiguous memory and C loops, reducing Python interpreter overhead for numeric workloads.")


def student_data_manager():
    print("\n===== Student Data Manager =====")
    students = [
        {'name': 'Alice', 'math': 85, 'science': 90, 'english': 78},
        {'name': 'Bob', 'math': 92, 'science': 88, 'english': 84},
        {'name': 'Carol', 'math': 74, 'science': 70, 'english': 82},
        {'name': 'Dave', 'math': 88, 'science': 95, 'english': 91},
        {'name': 'Eve', 'math': 79, 'science': 83, 'english': 76},
    ]

    for s in students:
        s['total'] = s['math'] + s['science'] + s['english']
        s['avg'] = s['total'] / 3
        if s['avg'] >= 90:
            s['grade'] = 'A+'
        elif s['avg'] >= 80:
            s['grade'] = 'A'
        elif s['avg'] >= 70:
            s['grade'] = 'B'
        elif s['avg'] >= 60:
            s['grade'] = 'C'
        else:
            s['grade'] = 'D'

    topper = max(students, key=lambda x: x['avg'])
    class_avg = sum(s['avg'] for s in students) / len(students)

    print("\nStudent details:")
    for s in students:
        print(f"{s['name']} - total: {s['total']}, avg: {s['avg']:.2f}, grade: {s['grade']}")

    print(f"\nTopper: {topper['name']} with average {topper['avg']:.2f}")
    print(f"Class average: {class_avg:.2f}")


def smart_input_program(name=None, age=None, hobby=None):
    print("\n===== Smart Input Program =====")
    if name is None:
        name = input("Enter your name: ").strip()
    if age is None:
        age_input = input("Enter your age: ").strip()
        try:
            age = int(age_input)
        except ValueError:
            print("Invalid age input. Age must be an integer.")
            return
    if hobby is None:
        hobby = input("Enter your hobby: ").strip()

    if age < 13:
        category = "child"
    elif age < 20:
        category = "teen"
    elif age < 60:
        category = "adult"
    else:
        category = "senior"

    print(f"Hello {name}! You are {age} years old, a {category}. It's great that you enjoy {hobby}.")


def fizzbuzz_logic_builder(start=1, end=50):
    print("\n===== Logic Builder: FizzBuzz 1-50 =====")
    fizz_count = buzz_count = fizzbuzz_count = 0
    for i in range(start, end + 1):
        if i % 15 == 0:
            print("FizzBuzz")
            fizzbuzz_count += 1
        elif i % 3 == 0:
            print("Fizz")
            fizz_count += 1
        elif i % 5 == 0:
            print("Buzz")
            buzz_count += 1
        else:
            print(i)

    print("\nOccurrence counts:")
    print(f"Fizz: {fizz_count}")
    print(f"Buzz: {buzz_count}")
    print(f"FizzBuzz: {fizzbuzz_count}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Dataset Detective: quick dataset inspection')
    parser.add_argument('--data', type=str, default='mall_customers.csv', help='Path to CSV dataset file')
    parser.add_argument('--numpy-test', action='store_true', help='Run NumPy speed test (1M elements)')
    parser.add_argument('--student-data', action='store_true', help='Run student data manager demo')
    parser.add_argument('--logic-builder', action='store_true', help='Run FizzBuzz logic builder demo')
    parser.add_argument('--smart-input', action='store_true', help='Run Smart Input Program')
    args = parser.parse_args()

    dataset_detective(args.data)
    if args.numpy_test:
        numpy_speed_test()
    if args.student_data:
        student_data_manager()
    if args.logic_builder:
        fizzbuzz_logic_builder()
    if args.smart_input:
        smart_input_program()
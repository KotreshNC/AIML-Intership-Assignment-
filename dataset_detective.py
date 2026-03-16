import pandas as pd


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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Dataset Detective: quick dataset inspection')
    parser.add_argument('--data', type=str, default='mall_customers.csv', help='Path to CSV dataset file')
    args = parser.parse_args()

    dataset_detective(args.data)

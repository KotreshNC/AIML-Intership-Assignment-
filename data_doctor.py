import pandas as pd


def load_sample_data():
    # Create sample dirty dataset
    data = {
        'Name': ['Alice', 'BOB', 'Carol', 'Alice', None, 'Eve', 'Frank', 'Grace', 'heidi', 'Ivan'],
        'Email': ['alice@example.com', 'bob@example.COM', 'carol@example.com', 'alice@example.com', 'dan@example.com', None, 'frank@example.com', 'grace@x.com', 'heidi@x.COM', 'ivan@example.com'],
        'Age': [25, 35, 45, 25, 50, 29, None, 32, 40, 22],
        'Salary': [50000, 60000, 80000, 50000, 72000, 55000, 62000, None, 68000, 49000],
        'JoinDate': ['2022-01-15', '2021-06-30', '2020-11-22', '2022-01-15', '2022-03-01', '2021-09-10', '2022-05-12', '2023-01-20', '2021-12-01', None]
    }
    return pd.DataFrame(data)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Original dataset:\n", df, "\n")

    # 1) Handle missing values
    # - For numeric columns, fill with median
    # - For text columns, fill with placeholder or mode
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Salary'] = df['Salary'].fillna(df['Salary'].median())
    df['Name'] = df['Name'].fillna('Unknown')
    df['Email'] = df['Email'].fillna('unknown@example.com')
    df['JoinDate'] = df['JoinDate'].fillna('1970-01-01')

    # 2) Remove exact duplicates
    before_dups = len(df)
    df = df.drop_duplicates()
    after_dups = len(df)
    print(f"Removed duplicates: {before_dups - after_dups}\n")

    # 3) Standardize text columns
    df['Name'] = df['Name'].str.title().str.strip()
    df['Email'] = df['Email'].str.lower().str.strip()

    # 4) Convert data types and parse dates
    df['Age'] = df['Age'].astype(int)
    df['Salary'] = df['Salary'].astype(float)
    df['JoinDate'] = pd.to_datetime(df['JoinDate'], errors='coerce').fillna(pd.Timestamp('1970-01-01'))

    # 5) Add a derived feature (tenure days)
    df['TenureDays'] = (pd.Timestamp('today') - df['JoinDate']).dt.days

    # 6) Reorder columns for readability
    df = df[['Name', 'Email', 'Age', 'Salary', 'JoinDate', 'TenureDays']]

    print("Cleaned dataset:\n", df, "\n")
    return df


def explain_cleaning():
    print("Why data cleaning matters:")
    print("1) Missing values can bias or break models and analytics.")
    print("2) Duplicates can overrepresent certain records and distort results.")
    print("3) Inconsistent text casing and whitespace can prevent correct grouping or matching.")
    print("4) Invalid datatypes block calculations and conversion steps.")
    print("5) Clean data leads to better model performance, more accurate insights, and robust decision-making.\n")


def main():
    df = load_sample_data()
    cleaned = clean_data(df)
    explain_cleaning()


if __name__ == '__main__':
    main()

import matplotlib.pyplot as plt
import pandas as pd


def main():
    # Sample dataset for a small retail store
    data = {
        'Category': ['Electronics', 'Clothing', 'Grocery', 'Home', 'Books'],
        'Sales': [120000, 90000, 60000, 50000, 30000],
        'Transactions': [1200, 1500, 2000, 800, 700],
        'CustomerSatisfaction': [4.5, 4.0, 4.2, 4.3, 4.7]
    }
    df = pd.DataFrame(data)

    # Bar chart: sales by category
    plt.figure(figsize=(8, 5))
    plt.bar(df['Category'], df['Sales'], color=['#4c72b0', '#55a868', '#c44e52', '#8172b2', '#ccb974'])
    plt.title('Sales by Product Category')
    plt.ylabel('Sales ($)')
    plt.xlabel('Category')
    plt.tight_layout()
    plt.savefig('bar_sales_by_category.png')
    plt.close()

    # Pie chart: distribution of transactions by category
    plt.figure(figsize=(6, 6))
    plt.pie(df['Transactions'], labels=df['Category'], autopct='%1.1f%%', startangle=140, colors=['#4c72b0', '#55a868', '#c44e52', '#8172b2', '#ccb974'])
    plt.title('Transaction Share by Category')
    plt.tight_layout()
    plt.savefig('pie_transactions_by_category.png')
    plt.close()

    # Histogram: synthetic daily sales distribution
    daily_sales = [1200, 1500, 1100, 1300, 1800, 2000, 1700, 900, 1400, 1600,
                   1300, 1350, 1450, 1550, 1650, 1750, 1250, 1850, 1900, 950,
                   2100, 2200, 1700, 1600, 1500, 1400, 1800, 2000, 2300, 1750]
    plt.figure(figsize=(8, 5))
    plt.hist(daily_sales, bins=6, color='#4c72b0', edgecolor='black')
    plt.title('Daily Sales Distribution (Sample Month)')
    plt.xlabel('Daily Sales ($)')
    plt.ylabel('Number of Days')
    plt.tight_layout()
    plt.savefig('hist_daily_sales.png')
    plt.close()

    print("Data Story:\n")
    print("In this retail store, Electronics leads revenue with $120K, followed by Clothing at $90K.")
    print("Though Grocery has lower revenue, its high transaction volume (33% share) shows many low-value purchases.")
    print("Home and Books have smaller shares, indicating areas for promotional focus to grow sales.")
    print("The daily sales histogram reveals most days fall between $1300 and $1900, with occasional peaks above $2100.")
    print("This suggests consistent demand with room to increase top-end sales through campaigns.")


if __name__ == '__main__':
    main()

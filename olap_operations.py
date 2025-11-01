import pandas as pd
import sqlite3

# Create an in-memory SQLite database
conn = sqlite3.connect(':memory:')
cursor = conn.cursor()

# ============================================
# 1. Create Tables (Star Schema)
# ============================================

cursor.execute("""
CREATE TABLE Product_Dim (
    Product_ID INTEGER PRIMARY KEY,
    Product_Name TEXT,
    Category TEXT,
    Brand TEXT
);
""")

cursor.execute("""
CREATE TABLE Branch_Dim (
    Branch_ID INTEGER PRIMARY KEY,
    Branch_Name TEXT,
    City TEXT,
    Region TEXT
);
""")

cursor.execute("""
CREATE TABLE Time_Dim (
    Time_ID INTEGER PRIMARY KEY,
    Day INTEGER,
    Month TEXT,
    Quarter TEXT,
    Year INTEGER
);
""")

cursor.execute("""
CREATE TABLE Sales_Fact (
    Sale_ID INTEGER PRIMARY KEY,
    Product_ID INTEGER,
    Branch_ID INTEGER,
    Time_ID INTEGER,
    Quantity INTEGER,
    Revenue INTEGER,
    FOREIGN KEY (Product_ID) REFERENCES Product_Dim(Product_ID),
    FOREIGN KEY (Branch_ID) REFERENCES Branch_Dim(Branch_ID),
    FOREIGN KEY (Time_ID) REFERENCES Time_Dim(Time_ID)
);
""")

# ============================================
# 2. Insert Sample Data
# ============================================

# Product Dimension
products = [
    (1, 'Laptop', 'Electronics', 'HP'),
    (2, 'Smartphone', 'Electronics', 'Samsung'),
    (3, 'Refrigerator', 'Home Appliance', 'LG'),
    (4, 'Washing Machine', 'Home Appliance', 'Whirlpool'),
    (5, 'Air Conditioner', 'Home Appliance', 'Voltas')
]

cursor.executemany("INSERT INTO Product_Dim VALUES (?, ?, ?, ?);", products)

# Branch Dimension
branches = [
    (101, 'Pune Store', 'Pune', 'West'),
    (102, 'Mumbai Store', 'Mumbai', 'West'),
    (103, 'Delhi Store', 'Delhi', 'North'),
    (104, 'Chennai Store', 'Chennai', 'South'),
    (105, 'Kolkata Store', 'Kolkata', 'East')
]

cursor.executemany("INSERT INTO Branch_Dim VALUES (?, ?, ?, ?);", branches)

# Time Dimension
times = [
    (1001, 12, 'Jan', 'Q1', 2025),
    (1002, 5, 'Feb', 'Q1', 2025),
    (1003, 18, 'Apr', 'Q2', 2025),
    (1004, 22, 'Jul', 'Q3', 2025),
    (1005, 11, 'Oct', 'Q4', 2025)
]

cursor.executemany("INSERT INTO Time_Dim VALUES (?, ?, ?, ?, ?);", times)

# Sales Fact
sales = [
    (1, 1, 101, 1001, 10, 750000),
    (2, 2, 101, 1002, 20, 600000),
    (3, 3, 102, 1003, 5, 250000),
    (4, 1, 103, 1005, 15, 1150000),
    (5, 4, 104, 1004, 7, 350000),
    (6, 5, 105, 1005, 8, 640000),
    (7, 3, 104, 1002, 10, 500000),
    (8, 2, 105, 1004, 18, 540000),
    (9, 1, 102, 1003, 12, 900000),
    (10, 5, 103, 1005, 6, 480000),
    (11, 4, 101, 1001, 9, 450000),
    (12, 2, 104, 1004, 14, 420000)
]

cursor.executemany("INSERT INTO Sales_Fact VALUES (?, ?, ?, ?, ?, ?);", sales)
conn.commit()

# Helper function to display results neatly
def show_query(title, query):
    print(f"\n=== {title} ===")
    df = pd.read_sql_query(query, conn)
    print(df.to_string(index=False))

# ============================================
# 3. OLAP Operations
# ============================================

# (a) Roll-Up: Summarize by Region
query_rollup = """
SELECT Region, SUM(Revenue) AS Total_Revenue
FROM Sales_Fact
JOIN Branch_Dim USING (Branch_ID)
GROUP BY Region;
"""
show_query("ROLL-UP (Total Revenue by Region)", query_rollup)

# (b) Drill-Down: Detailed view by City within Region
query_drilldown = """
SELECT Region, City, SUM(Revenue) AS City_Revenue
FROM Sales_Fact
JOIN Branch_Dim USING (Branch_ID)
GROUP BY Region, City;
"""
show_query("DRILL-DOWN (Revenue by City within Region)", query_drilldown)

# (c) Slice: Data for Quarter Q4
query_slice = """
SELECT Product_Name, SUM(Revenue) AS Q4_Revenue
FROM Sales_Fact
JOIN Product_Dim USING (Product_ID)
JOIN Time_Dim USING (Time_ID)
WHERE Quarter = 'Q4'
GROUP BY Product_Name;
"""
show_query("SLICE (Sales in Q4)", query_slice)

# (d) Dice: Electronics category in Q1 and Q2
query_dice = """
SELECT Region, Product_Name, SUM(Revenue) AS Total_Revenue
FROM Sales_Fact
JOIN Product_Dim USING (Product_ID)
JOIN Branch_Dim USING (Branch_ID)
JOIN Time_Dim USING (Time_ID)
WHERE Category = 'Electronics' AND Quarter IN ('Q1', 'Q2')
GROUP BY Region, Product_Name;
"""
show_query("DICE (Electronics in Q1 and Q2)", query_dice)

# (e) Pivot: Product vs Quarter (rotated view)
query_pivot = """
SELECT 
  Product_Name,
  SUM(CASE WHEN Quarter = 'Q1' THEN Revenue ELSE 0 END) AS Q1,
  SUM(CASE WHEN Quarter = 'Q2' THEN Revenue ELSE 0 END) AS Q2,
  SUM(CASE WHEN Quarter = 'Q3' THEN Revenue ELSE 0 END) AS Q3,
  SUM(CASE WHEN Quarter = 'Q4' THEN Revenue ELSE 0 END) AS Q4
FROM Sales_Fact
JOIN Product_Dim USING (Product_ID)
JOIN Time_Dim USING (Time_ID)
GROUP BY Product_Name;
"""
show_query("PIVOT (Product vs Quarter)", query_pivot)

# Close the connection
conn.close()

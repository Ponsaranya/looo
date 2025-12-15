import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()

# -----------------------------
# Feature Selection
# -----------------------------
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = df.select_dtypes(include=['object']).columns.tolist()

remove_cols = [
    'CustomerID',
    'Loan_type',
    'credit_cardtype',
    'investment_type',
    'savings_plan_type'
]

for col in remove_cols:
    if col in numeric_features:
        numeric_features.remove(col)
    if col in cat_features:
        cat_features.remove(col)

# -----------------------------
# Preprocessing
# -----------------------------
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, cat_features)
    ]
)

# -----------------------------
# Clustering
# -----------------------------
X_features = preprocessor.fit_transform(df[numeric_features + cat_features])
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(X_features)

# -----------------------------
# Recommendation Logic
# -----------------------------
def recommend(customer_id, product_col, top_n=3):
    customer = df[df['CustomerID'] == customer_id]

    if customer.empty:
        return "Customer ID not found."

    cluster_id = int(customer['cluster'].values[0])
    cluster_customers = df[df['cluster'] == cluster_id]

    popular_products = (
        cluster_customers[product_col]
        .value_counts()
        .dropna()
        .head(top_n)
        .index
        .tolist()
    )

    if not popular_products:
        return "No recommendation available."

    return ", ".join(popular_products)

# -----------------------------
# Chat-style Recommendation
# -----------------------------
def get_recommendation(user_input):
    user_input = user_input.lower()

    # Detect product type
    if "loan" in user_input:
        product_col = "Loan_type"
        product_name = "loan"
    elif "credit" in user_input:
        product_col = "credit_cardtype"
        product_name = "credit card"
    elif "investment" in user_input:
        product_col = "investment_type"
        product_name = "investment"
    elif "saving" in user_input:
        product_col = "savings_plan_type"
        product_name = "savings"
    else:
        return "Please mention loan, credit card, investment, or savings."

    # Extract Customer ID
    match = re.search(r'[Cc]\d+', user_input)
    if not match:
        return "Please provide a valid Customer ID."

    customer_id = match.group()

    suggestions = recommend(customer_id, product_col)

    return (
        f"Customer ID : {customer_id}\n"
        f"For your {product_name}, we suggest : {suggestions}"
    )

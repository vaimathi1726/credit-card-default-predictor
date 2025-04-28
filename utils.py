import pandas as pd

import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel


def preprocess_credit_card_data(df):
    """
    Preprocesses the Credit Card Default dataset:
    - Cleans invalid entries in SEX, EDUCATION, and MARRIAGE
    - One-hot encodes categorical variables with 0/1 (not True/False)
    - Standard scales numerical variables
    - Returns full cleaned DataFrame
    """
    df = df.copy()  # Make a safe copy

    # 1. Drop 'ID' column (not useful)
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])

    # 2. Clean invalid entries
    df = df[df['SEX'].isin([1, 2])]
    df = df[df['EDUCATION'].isin([1, 2, 3,4,5,6])]
    df = df[df['MARRIAGE'].isin([1, 2, 3])]

    # 3. Identify features
    target = 'default.payment.next.month'
    feature_cols = [col for col in df.columns if col != target]
    categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
    numerical_cols = [col for col in feature_cols if col not in categorical_cols]

    # 4. One-hot encode categorical columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    # 🔵 Force dummy columns to be integers (0 and 1)
    dummy_cols = [col for col in df.columns if any(prefix in col for prefix in ['SEX_', 'EDUCATION_', 'MARRIAGE_'])]
    df[dummy_cols] = df[dummy_cols].astype(int)

    # 5. Standard scale numerical columns
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # 6. Return the fully cleaned and transformed DataFrame
    return df

def feature_transform(X, method=None, degree=2, n_components=2, gamma=None, X_fit=None):
    """
    Transforms features based on specified method:
    
    Parameters:
    - X: pd.DataFrame or np.array, input features (to transform)
    - method: str, one of ['polynomial', 'pca', 'rbf', None]
    - degree: int, degree for polynomial features (if method='polynomial')
    - n_components: int, number of PCA components (if method='pca')
    - gamma: float, gamma value for RBF kernel (if method='rbf')
    - X_fit: np.array, the reference set for kernels (only used for RBF test transforms)
    
    Returns:
    - X_transformed: transformed feature matrix
    """
    if method == 'polynomial':
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        if X_fit is None:
            X_transformed = poly.fit_transform(X)  # For training
        else:
            X_transformed = poly.transform(X)      # For testing
    
    elif method == 'pca':
        pca = PCA(n_components=n_components)
        if X_fit is None:
            X_transformed = pca.fit_transform(X)  # For training
        else:
            X_transformed = pca.transform(X)      # For testing
    
    elif method == 'rbf':
        if X_fit is None:
            X_transformed = rbf_kernel(X, X, gamma=gamma)  # Train: X vs X
        else:
            X_transformed = rbf_kernel(X, X_fit, gamma=gamma)  # Test: X vs X_train

    else:
        X_transformed = X  # No transformation

    return X_transformed

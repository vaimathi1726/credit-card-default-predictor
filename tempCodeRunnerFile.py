import pandas as pd

import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel

from sklearn import linear_model
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base            import clone

from sklearn.model_selection import ParameterGrid
from sklearn.metrics         import accuracy_score, precision_recall_fscore_support



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




def grid_evaluate(
    estimator,
    param_grid,
    X_train, X_test,
    y_train, y_test
):
    """
    Brute-force “no CV” grid search over both feature transforms and classifier hyper-params.
    param_grid keys can include:
      - 'feature_method': [None, 'polynomial', 'pca', 'rbf']
      - 'degree':         [2,3]       (for polynomial)
      - 'n_components':   [5,10]      (for PCA)
      - 'gamma':          [0.1,0.5]   (for RBF)
      + any estimator params (e.g. 'C', 'penalty', etc.)
    """
    rows = []
    for params in ParameterGrid(param_grid):
        # 1) pull out transform params
        fm    = params.pop('feature_method', None)
        deg   = params.pop('degree',         None)
        ncomp = params.pop('n_components',   None)
        gam   = params.pop('gamma',          None)

        # 2) fit+transform on train, transform on test
        if fm == 'polynomial':
            poly = PolynomialFeatures(degree=deg, include_bias=False)
            X_tr = poly.fit_transform(X_train)
            X_te = poly.transform(X_test)

        elif fm == 'pca':
            pca = PCA(n_components=ncomp)
            X_tr = pca.fit_transform(X_train)
            X_te = pca.transform(X_test)

        elif fm == 'rbf':
            X_tr = rbf_kernel(X_train, X_train, gamma=gam)
            X_te = rbf_kernel(X_test,  X_train, gamma=gam)

        else:
            X_tr, X_te = X_train, X_test

        # 3) train & predict
        clf = clone(estimator).set_params(**params)
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)

        # 4) metrics
        acc  = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary', zero_division=0
        )

        # 5) record
        record = {
            'feature_method': fm,
            'degree':         deg,
            'n_components':   ncomp,
            'gamma':          gam,
            'accuracy':       acc,
            'precision':      prec,
            'recall':         rec,
            'f1_score':       f1,
        }
        record.update(params)  # remaining clf params
        rows.append(record)

    return pd.DataFrame(rows)
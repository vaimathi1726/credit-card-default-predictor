import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, precision_recall_fscore_support



def preprocess_credit_card_data(df):
    df = df.copy()

    #drop id; unnecessary for analysis
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])

    #Clean invalid catagorical entries
    df = df[df['SEX'].isin([1, 2])]
    df = df[df['EDUCATION'].isin([1, 2, 3,4,5,6])]
    df = df[df['MARRIAGE'].isin([1, 2, 3])]

    #Identify features
    target = 'default.payment.next.month'
    categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE']

    #One-hot encode categorical columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    #Force dummy columns to be integers (0 and 1) in case any are in a different form
    dummy_cols = [col for col in df.columns if any(prefix in col for prefix in ['SEX_', 'EDUCATION_', 'MARRIAGE_'])]
    df[dummy_cols] = df[dummy_cols].astype(int)

    return df




def grid_evaluate(
    estimator,
    param_grid,
    X_train, X_test,
    y_train, y_test
):
    """
    #This function replicates the functionality of the grid_search function in sklearn, but allows for more customization on our end
      - 'feature_method': [None, 'polynomial', 'pca', 'rbf']
      - 'degree':         [2,3]       (for polynomial)
      - 'n_components':   [5,10]      (for PCA)
      - 'gamma':          [0.1,0.5]   (for RBF)
      - 'hidden_layer_sizes': [(15,), (20,), (15,15)],
      - 'activation':         ['relu', 'tanh'],
      - 'alpha':              [1e-4, 1e-3], #L2 penalty (weight decay)
      - 'learning_rate_init': [1e-3, 1e-2],
      + any estimator params (e.g. 'C', 'penalty', etc.)
    """
    rows = []
    i=0
    for params in ParameterGrid(param_grid):
        i += 1
        #pull out transform params
        fm = params.pop('feature_method', None)
        deg = params.pop('degree', None)
        ncomp = params.pop('n_components', None)
        gam = params.pop('gamma', None)

        #fit+transform on train, transform on test
        
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

        #train & predict
        clf = clone(estimator).set_params(**params)
        clf.fit(X_tr, y_train)
        y_pred_test = clf.predict(X_te)
        y_pred_train = clf.predict(X_tr)

        #test metrics
        acc_test  = accuracy_score(y_test, y_pred_test)
        prec_test, rec_test, f1_test, _ = precision_recall_fscore_support(
            y_test, y_pred_test, average='binary', zero_division=0
        )

        #train metrics
        acc_train  = accuracy_score(y_train, y_pred_train)
        prec_train, rec_train, f1_train, _ = precision_recall_fscore_support(
            y_train, y_pred_train, average='binary', zero_division=0
        )

        #record
        if fm is None:
            fm = 'Linear'
        if fm != 'polynomial':
            deg = 1
        
        record = {
            'feature_method': fm,
            'degree':         deg,
            'n_components':   ncomp,
            'gamma':          gam,
            'accuracy_test':       acc_test,
            'accuracy_train':      acc_train,
            'precision_test':      prec_test,
            'recall_test':         rec_test,
            'f1_test':       f1_test,
            'f1_train':      f1_train,
            'precision_train':      prec_train,
            'recall_train':         rec_train,
        }
        
        record.update(params)  #remaining clf params
        rows.append(record)
        if(i%5 == 0):
            print(f"Evaluated {i} parameter combinations...")

    return pd.DataFrame(rows)


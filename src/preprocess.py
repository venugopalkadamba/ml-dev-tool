import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class MissingImputer:
    def __init__(self, method: str, column: str):
        self.impute_method = method
        self.column = column
        self.impute_statistic = None
    
    def fit(self, X: pd.DataFrame, y=None):
        if self.impute_method == "mean":
            self.impute_statistic = X[self.column].mean()
        elif self.impute_method == "median":
            self.impute_statistic = X[self.column].median()
        elif self.impute_method == "mode":
            # Use pandas mode for stable behavior; take first mode if multiple
            mode_series = X[self.column].mode(dropna=True)
            self.impute_statistic = mode_series.iloc[0] if not mode_series.empty else None
        else:
            raise ValueError("Unsupported imputation method. Choose 'mean', 'median', or 'mode'.")
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.impute_statistic is None:
            raise ValueError("MissingImputer must be fitted on training data before transform().")
        data_cp = data.copy()
        data_cp[self.column] = data_cp[self.column].fillna(self.impute_statistic)
        return data_cp

class SkewnessCorrector:
    def __init__(self, method: str, column: str):
        self.method = method
        self.column = column
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data_cp = data.copy()

        if self.method == "log":
            # Avoid -inf for non-positive values by shifting
            if (data_cp[self.column] <= 0).any():
                shift = abs(data_cp[self.column].min()) + 1e-6
                data_cp[self.column] = data_cp[self.column] + shift
            data_cp[self.column] = np.log(data_cp[self.column])
        elif self.method == "sqrt":
            # Ensure non-negativity
            if (data_cp[self.column] < 0).any():
                shift = abs(data_cp[self.column].min())
                data_cp[self.column] = data_cp[self.column] + shift
            data_cp[self.column] = np.sqrt(data_cp[self.column])
        elif self.method == "box_cox":
            # Box-Cox requires strictly positive values
            if (data_cp[self.column] <= 0).any():
                shift = abs(data_cp[self.column].min()) + 1e-6
                data_cp[self.column] = data_cp[self.column] + shift
            data_cp[self.column], lambda_opt = stats.boxcox(data_cp[self.column])
        elif self.method == "exp":
            if (data_cp[self.column] < 0).any():
                # Ensure positive values by shifting (if necessary)
                shift = abs(data_cp[self.column].min()) + 1  # Shift to make all values positive
                data_cp[self.column] = data_cp[self.column] + shift
            data_cp[self.column] = np.exp(data_cp[self.column])
        
        return data_cp

class FeatureEncoder:
    def __init__(self, method: str, column: str, **kwargs):
        self.method = method
        self.column = column
        self.args = kwargs
        self.encoder = None
    
    def fit(self, X, y):
        if self.method == 'label':
            self.encoder = LabelEncoder()
            self.encoder.fit(X[self.column])
        elif self.method == 'onehot':
            # Construct without sparse flags for broad compatibility across sklearn versions
            self.encoder = OneHotEncoder(drop=self.args.get('drop', None))
            self.encoder.fit(X[[self.column]])
        else:
            raise ValueError("Unsupported encoding method. Choose 'label' or 'onehot'.")
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data_cp = data.copy()
        
        if self.method == 'label':
            if not self.encoder:
                raise ValueError("Encoder is not fitted. Call fit() first.")
            data_cp[self.column] = self.encoder.transform(data_cp[self.column])
        
        elif self.method == 'onehot':
            if not self.encoder:
                raise ValueError("Encoder is not fitted. Call fit() first.")
            encoded_cols = self.encoder.transform(data_cp[[self.column]])
            # Convert sparse matrix to dense array if needed for DataFrame construction
            try:
                import scipy.sparse as sp
                if sp.issparse(encoded_cols):
                    encoded_cols = encoded_cols.toarray()
            except Exception:
                # If scipy isn't available or any issue occurs, attempt numpy array conversion
                try:
                    encoded_cols = encoded_cols.toarray()
                except Exception:
                    pass
            col_names = self.encoder.get_feature_names_out([self.column])
            encoded_df = pd.DataFrame(encoded_cols, columns=col_names, index=data_cp.index)
            data_cp = data_cp.drop(columns=[self.column]).join(encoded_df)
        
        return data_cp

class Binner:
    def __init__(self, method: str, column: str, n_bins: int = 5, custom_bins: Optional[list] = None, labels: Optional[list] = None):
        self.method = method  # 'quantile' or 'custom'
        self.column = column
        self.n_bins = n_bins
        self.custom_bins = custom_bins
        self.labels = labels
        self.bin_edges_ = None
    
    def fit(self, X: pd.DataFrame, y=None):
        if self.method == 'quantile':
            quantiles = np.linspace(0, 1, self.n_bins + 1)
            self.bin_edges_ = X[self.column].quantile(quantiles).values
            # Ensure strict monotonicity to avoid duplicate edges
            self.bin_edges_ = np.unique(self.bin_edges_)
        elif self.method == 'custom':
            if not self.custom_bins or len(self.custom_bins) < 2:
                raise ValueError("custom_bins must have at least two edges.")
            self.bin_edges_ = np.array(self.custom_bins)
        else:
            raise ValueError("Unsupported binning method. Choose 'quantile' or 'custom'.")
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.bin_edges_ is None:
            raise ValueError("Binner must be fitted before transform().")
        data_cp = data.copy()
        data_cp[self.column] = pd.cut(
            data_cp[self.column],
            bins=self.bin_edges_,
            labels=self.labels,
            include_lowest=True,
            duplicates='drop'
        )
        return data_cp

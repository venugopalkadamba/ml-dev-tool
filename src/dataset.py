import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional
from enum import Enum
from sklearn.datasets import fetch_openml, load_iris
from sklearn.preprocessing import LabelEncoder

class FeatureType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"

class BiasType(Enum):
    REPRESENTATION = "representation"
    MEASUREMENT = "measurement"
    SAMPLING = "sampling"
    LABEL = "label"

class DatasetGenerator:
    def __init__(self):
        self.data = None
        self.protected_attribute = None
        self.label_encoder = LabelEncoder()
    
    def load_benchmark_dataset(self, dataset_name: str) -> pd.DataFrame:
        """
        Load a benchmark dataset (e.g., Adult Income)
        
        Args:
            dataset_name (str): Name of the benchmark dataset
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        if dataset_name.lower() == "adult":
            data = fetch_openml("adult", version=2, as_frame=True)
            self.data = data.data
            self.data["target"] = data.target
            self.data["sex"] = self.label_encoder.fit_transform(self.data["sex"])
            self.protected_attribute = "sex"
            return self.data
        elif dataset_name.lower() == "iris":
            data = load_iris(as_frame=True)
            data_df = data["data"]
            data_df["flower_type"] = data["target"]
            return data_df
        else:
            raise ValueError(f"Dataset {dataset_name} not supported yet")

    def generate_synthetic_dataset(
        self,
        n_samples: int,
        feature_specs: Dict[str, Dict],
        protected_attr_ratio: float = 0.5,
        positive_label_ratio: float = 0.5
    ) -> pd.DataFrame:
        """
        Generate a synthetic dataset with specified characteristics
        
        Args:
            n_samples (int): Number of samples to generate
            feature_specs (Dict): Specification for each feature
                Format: {
                    "feature_name": {
                        "type": FeatureType,
                        "params": Dict with distribution parameters
                    }
                }
            protected_attr_ratio (float): Ratio of samples in protected group 1
            positive_label_ratio (float): Ratio of positive labels
            
        Returns:
            pd.DataFrame: Generated synthetic dataset
        """
        data = {}
        
        # Generate protected attribute (binary)
        protected_attr = np.random.binomial(n=1, p=protected_attr_ratio, size=n_samples)
        data["protected_attribute"] = protected_attr
        
        # Generate features based on specifications
        for feature_name, spec in feature_specs.items():
            if spec["type"] == FeatureType.NUMERIC:
                if spec["params"].get("distribution") == "normal":
                    data[feature_name] = np.random.normal(
                        loc=spec["params"].get("mean", 0),
                        scale=spec["params"].get("std", 1),
                        size=n_samples
                    )
                elif spec["params"].get("distribution") == "uniform":
                    data[feature_name] = np.random.uniform(
                        low=spec["params"].get("low", 0),
                        high=spec["params"].get("high", 1),
                        size=n_samples
                    )
            elif spec["type"] == FeatureType.CATEGORICAL:
                categories = spec["params"].get("categories", [])
                probabilities = spec["params"].get("probabilities", None)
                data[feature_name] = np.random.choice(
                    categories,
                    size=n_samples,
                    p=probabilities
                )
        
        # Generate labels
        labels = np.random.binomial(n=1, p=positive_label_ratio, size=n_samples)
        data["target"] = labels
        
        self.data = pd.DataFrame(data)
        self.protected_attribute = "protected_attribute"
        return self.data

    def load_custom_dataset(
        self,
        dataset: pd.DataFrame,
        protected_attribute: str,
        target_column: str
    ) -> pd.DataFrame:
        """
        Load a custom dataset
        
        Args:
            dataset (pd.DataFrame): Custom dataset
            protected_attribute (str): Name of protected attribute column
            target_column (str): Name of target column
            
        Returns:
            pd.DataFrame: Processed dataset
        """
        self.data = dataset.copy()
        self.protected_attribute = protected_attribute
        if protected_attribute not in self.data.columns:
            raise ValueError(f"Protected attribute {protected_attribute} not found in dataset")
        if target_column not in self.data.columns:
            raise ValueError(f"Target column {target_column} not found in dataset")
        return self.data

class RepresentationBias:
    def __init__(self, protected_attribute: str, underrepresented_group: int, reduction_factor: float):
        self.protected_attribute = protected_attribute
        self.underrepresented_group = underrepresented_group
        self.reduction_factor = reduction_factor
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Inject representation bias by undersampling a protected group
        """
        biased_data = data.copy()
        mask = biased_data[self.protected_attribute] == self.underrepresented_group
        group_indices = biased_data[mask].index
        drop_size = int(len(group_indices) * (1 - self.reduction_factor))
        drop_indices = np.random.choice(group_indices, size=drop_size, replace=False)
        return biased_data.drop(drop_indices)

class MeasurementBias:
    def __init__(self, feature: str, protected_attribute: str, affected_group: int, noise_std: float):
        self.feature = feature
        self.protected_attribute = protected_attribute
        self.affected_group = affected_group
        self.noise_std = noise_std

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Inject measurement bias by adding noise to features for a protected group
        """
        biased_data = data.copy()
        mask = biased_data[self.protected_attribute] == self.affected_group
        noise = np.random.normal(0, self.noise_std, size=mask.sum())
        biased_data.loc[mask, self.feature] += noise
        return biased_data

class SamplingBias:
    def __init__(self, 
        protected_attribute: str,
        target_column: str,
        affected_group: int,
        label_value: int,
        sampling_rate: float
        ):
        self.protected_attribute = protected_attribute
        self.target_column = target_column
        self.affected_group = affected_group
        self.label_value = label_value
        self.sampling_rate = sampling_rate
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Inject sampling bias by undersampling specific label values for a protected group
        """
        biased_data = data.copy()
        mask = (biased_data[self.protected_attribute] == self.affected_group) & (biased_data[self.target_column] == self.label_value)
        group_indices = biased_data[mask].index
        keep_size = int(len(group_indices) * self.sampling_rate)
        keep_indices = np.random.choice(group_indices, size=keep_size, replace=False)
        drop_indices = list(set(group_indices) - set(keep_indices))
        return biased_data.drop(drop_indices)

class LabelBias:
    def __init__(self, protected_attribute: str, target_column: str, affected_group, flip_probability: float):
        self.protected_attribute = protected_attribute
        self.target_column = target_column
        self.affected_group = affected_group
        self.flip_probability = flip_probability
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Inject label bias by randomly flipping labels for a protected group
        """
        biased_data = data.copy()
        mask = biased_data[self.protected_attribute] == self.affected_group
        affected_indices = biased_data[mask].index
        flip_mask = np.random.random(size=len(affected_indices)) < self.flip_probability
        idx_to_flip = affected_indices[flip_mask]
        # Support non-binary labels by sampling from remaining classes
        unique_labels = biased_data[self.target_column].dropna().unique()
        if len(unique_labels) == 2 and set(unique_labels) <= {0, 1}:
            biased_data.loc[idx_to_flip, self.target_column] = 1 - biased_data.loc[idx_to_flip, self.target_column]
        else:
            # For multi-class or non-numeric labels: flip to any other class uniformly
            for idx in idx_to_flip:
                current = biased_data.at[idx, self.target_column]
                choices = [c for c in unique_labels if c != current]
                if len(choices) > 0:
                    biased_data.at[idx, self.target_column] = np.random.choice(choices)
        return biased_data

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Tuple, Union, List, Dict, Any, Optional

class Pipeline:
    def __init__(self, steps: list):
        """
        Initialize a pipeline with a series of steps.
        
        Args:
            steps: List of tuples (name, transformer) where transformer can be
                  a bias injection class, preprocessing class, or sklearn model
        """
        self.steps = steps
        self.fitted_transformers = {}
        self.model = None
        self.target_column = None
        
    def fit(self, data: pd.DataFrame, target_column: str) -> 'Pipeline':
        """
        Fit the pipeline on training data.
        
        Args:
            data: Training data DataFrame
            target_column: Name of the target column
            
        Returns:
            self: The fitted pipeline
        """
        self.target_column = target_column
        X = data.drop(columns=[target_column]) if target_column in data.columns else data
        y = data[target_column] if target_column in data.columns else None
        
        for step_name, transformer in self.steps[:-1]:  # Process all steps except the last one (model)
            if hasattr(transformer, 'fit') and callable(getattr(transformer, 'fit')):
                # For transformers that need fitting (like sklearn preprocessing or custom encoders)
                transformer.fit(X, y)
                self.fitted_transformers[step_name] = transformer
            
            # Apply the transformation
            if hasattr(transformer, 'transform') and callable(getattr(transformer, 'transform')):
                if target_column in data.columns:
                    data = transformer.transform(data)
                    # Re-extract X in case the transformer modified the dataframe structure
                    X = data.drop(columns=[target_column])
                    y = data[target_column]
                else:
                    X = transformer.transform(X)
        
        # Fit the model (last step)
        if len(self.steps) > 0:
            model_name, model = self.steps[-1]
            if hasattr(model, 'fit') and callable(getattr(model, 'fit')):
                self.model = model.fit(X, y)
            else:
                self.model = model
        
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all transformations in the pipeline (except the final model).
        
        Args:
            data: Data to transform
            
        Returns:
            Transformed data
        """
        transformed_data = data.copy()
        
        # Check if target column exists in the input data
        has_target = self.target_column in transformed_data.columns
        
        bias_step_names = {"RepresentationBias", "MeasurementBias", "SamplingBias", "LabelBias"}
        for step_name, transformer in self.steps[:-1]:  # Exclude the final model
            # Skip bias injection steps during transform to avoid changing eval/test rows
            if transformer.__class__.__name__ in bias_step_names:
                continue
            if hasattr(transformer, 'transform') and callable(getattr(transformer, 'transform')):
                transformed_data = transformer.transform(transformed_data)
        
        return transformed_data
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Transform data and make predictions.
        
        Args:
            data: Data to predict on
            
        Returns:
            Model predictions
        """
        if self.model is None:
            raise ValueError("Pipeline has not been fitted with a model.")
            
        # First transform the data through all preprocessing steps
        transformed_data = self.transform(data)
        
        # Extract features and target from the transformed data if available
        if self.target_column and self.target_column in transformed_data.columns:
            X = transformed_data.drop(columns=[self.target_column])
        else:
            X = transformed_data
        
        # Make predictions using the model
        return self.model.predict(X)
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        Transform data and make probability predictions (for classifiers).
        
        Args:
            data: Data to predict on
            
        Returns:
            Probability predictions
        """
        if self.model is None:
            raise ValueError("Pipeline has not been fitted with a model.")
            
        if not hasattr(self.model, 'predict_proba') or not callable(getattr(self.model, 'predict_proba')):
            raise AttributeError("Model does not support probability predictions.")
            
        # Transform the data and extract features
        transformed_data = self.transform(data)
        
        if self.target_column and self.target_column in transformed_data.columns:
            X = transformed_data.drop(columns=[self.target_column])
        else:
            X = transformed_data
        
        # Return probability predictions
        return self.model.predict_proba(X)
    
    def score(self, data: pd.DataFrame, target_column: Optional[str] = None) -> float:
        """
        Calculate the model's score on the provided data.
        
        Args:
            data: Data to score on
            target_column: Name of the target column (if different from fitted target)
            
        Returns:
            Model score
        """
        if self.model is None:
            raise ValueError("Pipeline has not been fitted with a model.")
            
        # Use the provided target column or fall back to the one used during fitting
        target = target_column if target_column is not None else self.target_column
        
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in data.")
            
        # Transform the data
        transformed_data = self.transform(data)
        
        # Extract features and target from the transformed data to preserve alignment
        if target not in transformed_data.columns:
            raise ValueError(f"Target column '{target}' not found after transformations.")
        X = transformed_data.drop(columns=[target])
        y = transformed_data[target]
        
        # Return the model's score
        return self.model.score(X, y)
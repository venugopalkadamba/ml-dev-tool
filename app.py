import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from src.eval import (
    compute_accuracy,
    compute_precision_weighted,
    compute_recall_weighted,
    compute_f1_weighted,
    compute_log_loss,
    compute_roc_auc,
    compute_pr_auc,
    compute_roc_curve,
    compute_pr_curve,
    compute_confusion_matrix,
    generate_classification_report,
    compute_demographic_parity_ratio,
)

# Import your custom modules
# Assuming these are in the same directory
from src.dataset import DatasetGenerator, FeatureType, BiasType, RepresentationBias, MeasurementBias, SamplingBias, LabelBias
from src.preprocess import MissingImputer, SkewnessCorrector, FeatureEncoder, Binner
from src.pipeline import Pipeline
from src.models import WeightedEnsembleClassifier, StackingMetaLearner

# Optional model backends
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

# Define page title and layout
st.set_page_config(page_title="Bias Analysis Pipeline Builder", layout="wide")
st.markdown(
    """
    <style>
    /* Force light theme look for our cards irrespective of user agent preference */
    .step-card { background:#FFFFFF !important; }
    .step-title { color:#111827 !important; }
    .step-meta { color:#374151 !important; opacity:0.80 !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.title("Interactive Bias Analysis Pipeline Builder")
st.markdown("""
This application allows you to:
1. Upload your own dataset or generate synthetic data
2. Create a custom data processing pipeline
3. Apply bias transformations
4. Train and evaluate models
5. Visualize results
""")

# Session state initialization
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'pipeline_steps' not in st.session_state:
    st.session_state.pipeline_steps = []
if 'train_data' not in st.session_state:
    st.session_state.train_data = None
if 'val_data' not in st.session_state:
    st.session_state.val_data = None
if 'test_data' not in st.session_state:
    st.session_state.test_data = None
if 'trained_pipeline' not in st.session_state:
    st.session_state.trained_pipeline = None
if 'feature_specs' not in st.session_state:
    st.session_state.feature_specs = {}
if 'generator_params' not in st.session_state:
    st.session_state.generator_params = {
        'n_samples': 1000,
        'protected_attr_ratio': 0.5,
        'positive_label_ratio': 0.5
    }
if 'split_mode' not in st.session_state:
    st.session_state.split_mode = "Train/Test"
if 'saved_pipelines' not in st.session_state:
    st.session_state.saved_pipelines = []
if 'ensemble_estimators' not in st.session_state:
    st.session_state.ensemble_estimators = []  # list of (name, estimator)
if 'ensemble_weights' not in st.session_state:
    st.session_state.ensemble_weights = []     # list of floats matching estimators
if 'stacking_estimators' not in st.session_state:
    st.session_state.stacking_estimators = []  # list of (name, estimator)
if 'stacking_final' not in st.session_state:
    st.session_state.stacking_final = None
if 'steps_version' not in st.session_state:
    st.session_state.steps_version = 0

# Initialize app_mode to avoid widget default conflict
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = "Data Generation"

# Apply any pending navigation before widgets are created
if 'pending_app_mode' in st.session_state:
    st.session_state.app_mode = st.session_state['pending_app_mode']
    del st.session_state['pending_app_mode']

# Function to add a new feature specification for synthetic data
def add_feature_spec():
    feature_name = st.text_input("Feature Name", key=f"feat_name_{len(st.session_state.feature_specs)}")
    if feature_name:
        feature_type = st.selectbox(
            "Feature Type",
            options=[FeatureType.NUMERIC.value, FeatureType.CATEGORICAL.value],
            key=f"feat_type_{feature_name}"
        )
        
        if feature_type == FeatureType.NUMERIC.value:
            distribution = st.selectbox(
                "Distribution", 
                options=["normal", "uniform"],
                key=f"dist_{feature_name}"
            )
            
            if distribution == "normal":
                mean = st.number_input("Mean", value=0.0, key=f"mean_{feature_name}")
                std = st.number_input("Standard Deviation", value=1.0, min_value=0.1, key=f"std_{feature_name}")
                params = {"distribution": distribution, "mean": mean, "std": std}
            else:  # uniform
                low = st.number_input("Low", value=0.0, key=f"low_{feature_name}")
                high = st.number_input("High", value=1.0, key=f"high_{feature_name}")
                params = {"distribution": distribution, "low": low, "high": high}
                
            return {
                feature_name: {
                    "type": FeatureType.NUMERIC,
                    "params": params
                }
            }
            
        elif feature_type == FeatureType.CATEGORICAL.value:
            categories_str = st.text_input(
                "Categories (comma-separated)", 
                "category1,category2,category3",
                key=f"cat_{feature_name}"
            )
            categories = [c.strip() for c in categories_str.split(",")]
            
            probs_str = st.text_input(
                "Probabilities (comma-separated, must sum to 1)", 
                ",".join(["0.33"] * len(categories)),
                key=f"prob_{feature_name}"
            )
            probabilities = [float(p.strip()) for p in probs_str.split(",")]
            
            # Validate probabilities
            if len(probabilities) != len(categories):
                st.error("Number of probabilities must match number of categories")
                return None
            
            if abs(sum(probabilities) - 1.0) > 0.01:
                st.error("Probabilities must sum to 1")
                return None
                
            return {
                feature_name: {
                    "type": FeatureType.CATEGORICAL,
                    "params": {
                        "categories": categories,
                        "probabilities": probabilities
                    }
                }
            }
    return None

# Function to generate or load dataset
def load_or_generate_data():
    tab_gen, tab_upload, tab_benchmark = st.tabs(["Generate Synthetic Data", "Upload Data", "Use Benchmark Dataset"])
    
    with tab_gen:
        st.subheader("Synthetic Data Generator")
        col1, col2 = st.columns(2)
        with col1:
            n_samples = st.number_input("Number of Samples", min_value=100, value=1000, step=100)
            protected_attr_ratio = st.slider("Protected Attribute Ratio", 0.0, 1.0, 0.5)
        with col2:
            positive_label_ratio = st.slider("Positive Label Ratio", 0.0, 1.0, 0.5)
        st.session_state.generator_params = {
            'n_samples': n_samples,
            'protected_attr_ratio': protected_attr_ratio,
            'positive_label_ratio': positive_label_ratio
        }
        st.subheader("Feature Specifications")
        if st.button("Add Feature"):
            new_spec = add_feature_spec()
            if new_spec:
                st.session_state.feature_specs.update(new_spec)
        if st.session_state.feature_specs:
            st.write("Current Feature Specifications:")
            for name, spec in st.session_state.feature_specs.items():
                st.text(f"{name}: {spec['type'].value}, {spec['params']}")
                if st.button(f"Remove {name}", key=f"remove_{name}"):
                    del st.session_state.feature_specs[name]
                    st.rerun()
        if st.button("Generate Dataset"):
            if len(st.session_state.feature_specs) > 0:
                data_generator = DatasetGenerator()
                try:
                    df = data_generator.generate_synthetic_dataset(
                        n_samples=n_samples,
                        feature_specs=st.session_state.feature_specs,
                        protected_attr_ratio=protected_attr_ratio,
                        positive_label_ratio=positive_label_ratio
                    )
                    st.session_state.dataset = df
                    st.success("Dataset generated successfully!")
                    st.dataframe(df.head())
                except Exception as e:
                    st.error(f"Error generating dataset: {str(e)}")
            else:
                st.error("Please add at least one feature specification")
    
    with tab_upload:
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.dataset = df
                st.success("Dataset uploaded successfully!")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    with tab_benchmark:
        dataset_name = st.selectbox("Select Benchmark Dataset", ["adult", "iris"])
        if st.button("Load Dataset"):
            data_generator = DatasetGenerator()
            try:
                df = data_generator.load_benchmark_dataset(dataset_name)
                st.session_state.dataset = df
                st.success(f"{dataset_name.capitalize()} dataset loaded successfully!")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"Error loading benchmark dataset: {str(e)}")
    
    if st.session_state.dataset is not None:
        if st.button("Proceed to Pipeline Configuration"):
            st.session_state['pending_app_mode'] = "Pipeline Configuration"
            st.rerun()

# Function to configure and build pipeline
def configure_pipeline():
    st.subheader("Configure Pipeline")
    
    # Target column selection
    if st.session_state.dataset is not None:
        all_cols = st.session_state.dataset.columns.tolist()
        default_target_index = all_cols.index('target') if 'target' in all_cols else len(all_cols)-1
        target_col = st.selectbox(
            "Select Target Column", 
            options=all_cols,
            index=default_target_index
        )
        
        left, right = st.columns([2, 1])
        non_target_columns = [c for c in st.session_state.dataset.columns if c != target_col]
        with left:
            tab_split, tab_bias, tab_pre, tab_feat, tab_model = st.tabs(["Split", "Bias", "Preprocess", "Features", "Model"])
            with tab_split:
                st.session_state.split_mode = st.radio("Split Mode", ["Train/Test", "Train/Val/Test"], horizontal=True)
                if st.session_state.split_mode == "Train/Test":
                    test_ratio = st.slider("Test Ratio", 0.1, 0.5, 0.2)
                    if st.button("Split Dataset", key="split_tt"):
                        try:
                            train_data, test_data = train_test_split(
                    st.session_state.dataset, 
                                test_size=test_ratio,
                                random_state=42
                            )
                            st.session_state.train_data = train_data
                            st.session_state.val_data = None
                            st.session_state.test_data = test_data
                            st.session_state.target_column = target_col
                            st.success(f"Data split into {len(train_data)} train and {len(test_data)} test samples")
                        except Exception as e:
                            st.error(f"Error splitting dataset: {str(e)}")
                else:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        val_ratio = st.slider("Validation Ratio", 0.05, 0.5, 0.2)
                    with col_b:
                        test_ratio = st.slider("Test Ratio", 0.05, 0.5, 0.2)
                    if val_ratio + test_ratio >= 0.9:
                        st.warning("Validation + Test ratio is very high; ensure enough training data remains.")
                    if st.button("Split Dataset", key="split_tvt"):
                        try:
                            temp_train, test_data = train_test_split(
                                st.session_state.dataset,
                                test_size=test_ratio,
                                random_state=42
                            )
                            # Adjust val size relative to remaining
                            relative_val = val_ratio / (1.0 - test_ratio)
                            train_data, val_data = train_test_split(
                                            temp_train,
                                            test_size=relative_val,
                                random_state=42
                            )
                            st.session_state.train_data = train_data
                            st.session_state.val_data = val_data
                            st.session_state.test_data = test_data
                            st.session_state.target_column = target_col
                            st.success(f"Data split into {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples")
                        except Exception as e:
                            st.error(f"Error splitting dataset: {str(e)}")
        
            with tab_bias:
                bias_types = [
                    "Representation Bias", 
                    "Measurement Bias", 
                    "Sampling Bias", 
                    "Label Bias"
                ]
                bias_type = st.selectbox("Bias Type", bias_types)
                
                if bias_type == "Representation Bias":
                    protected_attr = st.selectbox(
                        "Protected Attribute", 
                        options=non_target_columns
                    )
                    rep_groups = st.session_state.dataset[protected_attr].dropna().unique().tolist()
                    underrep_group = st.selectbox("Underrepresented Group", options=rep_groups, key=f"rep_under_{protected_attr}")
                    reduction = st.slider("Reduction Factor", 0.1, 1.0, 0.5)
                    
                    if st.button("Add Representation Bias Step"):
                        step_name = f"rep_bias_{len(st.session_state.pipeline_steps)}"
                        step = (
                            step_name, 
                            RepresentationBias(
                                protected_attribute=protected_attr,
                                underrepresented_group=underrep_group,
                                reduction_factor=reduction
                            )
                        )
                        st.session_state.pipeline_steps.append(step)
                        st.success("Representation Bias step added to pipeline")
                        st.session_state.steps_version += 1
                        st.rerun()
                
                elif bias_type == "Measurement Bias":
                    feature = st.selectbox(
                        "Feature to Add Noise", 
                        options=non_target_columns
                    )
                    protected_attr = st.selectbox(
                        "Protected Attribute", 
                        options=non_target_columns
                    )
                    meas_groups = st.session_state.dataset[protected_attr].dropna().unique().tolist()
                    affected_group = st.selectbox("Affected Group", options=meas_groups, key=f"meas_aff_{protected_attr}")
                    noise_std = st.slider("Noise Standard Deviation", 0.1, 5.0, 1.0)
                    
                    if st.button("Add Measurement Bias Step"):
                        step_name = f"meas_bias_{len(st.session_state.pipeline_steps)}"
                        step = (
                            step_name, 
                            MeasurementBias(
                                feature=feature,
                                protected_attribute=protected_attr,
                                affected_group=affected_group,
                                noise_std=noise_std
                            )
                        )
                        st.session_state.pipeline_steps.append(step)
                        st.success("Measurement Bias step added to pipeline")
                        st.session_state.steps_version += 1
                        st.rerun()
                
                elif bias_type == "Sampling Bias":
                    protected_attr = st.selectbox(
                        "Protected Attribute", 
                        options=non_target_columns
                    )
                    target_column = st.selectbox(
                        "Target Column", 
                        options=st.session_state.dataset.columns.tolist(),
                        index=st.session_state.dataset.columns.get_loc(target_col) if target_col in st.session_state.dataset.columns else len(st.session_state.dataset.columns)-1
                    )
                    samp_groups = st.session_state.dataset[protected_attr].dropna().unique().tolist()
                    affected_group = st.selectbox("Affected Group", options=samp_groups, key=f"samp_aff_{protected_attr}")
                    label_vals = st.session_state.dataset[target_column].dropna().unique().tolist()
                    label_value = st.selectbox("Label Value", options=label_vals, key=f"samp_label_{target_column}")
                    sampling_rate = st.slider("Sampling Rate", 0.1, 1.0, 0.5)
                    
                    if st.button("Add Sampling Bias Step"):
                        step_name = f"samp_bias_{len(st.session_state.pipeline_steps)}"
                        step = (
                            step_name, 
                            SamplingBias(
                                protected_attribute=protected_attr,
                                target_column=target_column,
                                affected_group=affected_group,
                                label_value=label_value,
                                sampling_rate=sampling_rate
                            )
                        )
                        st.session_state.pipeline_steps.append(step)
                        st.success("Sampling Bias step added to pipeline")
                        st.session_state.steps_version += 1
                        st.rerun()
                
                elif bias_type == "Label Bias":
                    protected_attr = st.selectbox(
                        "Protected Attribute", 
                        options=non_target_columns
                    )
                    target_column = st.selectbox(
                        "Target Column", 
                        options=st.session_state.dataset.columns.tolist(),
                        index=st.session_state.dataset.columns.get_loc(target_col) if target_col in st.session_state.dataset.columns else len(st.session_state.dataset.columns)-1
                    )
                    label_groups = st.session_state.dataset[protected_attr].dropna().unique().tolist()
                    affected_group = st.selectbox("Affected Group", options=label_groups, key=f"label_aff_{protected_attr}")
                    flip_prob = st.slider("Label Flip Probability", 0.0, 1.0, 0.2)
                    
                    if st.button("Add Label Bias Step"):
                        step_name = f"label_bias_{len(st.session_state.pipeline_steps)}"
                        step = (
                            step_name, 
                            LabelBias(
                                protected_attribute=protected_attr,
                                target_column=target_column,
                                affected_group=affected_group,
                                flip_probability=flip_prob
                            )
                        )
                        st.session_state.pipeline_steps.append(step)
                        st.success("Label Bias step added to pipeline")
                        st.session_state.steps_version += 1
                        st.rerun()
            
            with tab_pre:
                preproc_types = ["Missing Value Imputation", "Skewness Correction", "Binning"]
                preproc_type = st.selectbox("Preprocessing Type", preproc_types)
                
                if preproc_type == "Missing Value Imputation":
                    column = st.selectbox(
                        "Column to Impute", 
                        options=non_target_columns
                    )
                    method = st.selectbox("Imputation Method", ["mean", "median", "mode"])
                    
                    if st.button("Add Imputation Step"):
                        step_name = f"imputer_{len(st.session_state.pipeline_steps)}"
                        step = (step_name, MissingImputer(method=method, column=column))
                        st.session_state.pipeline_steps.append(step)
                        st.success("Missing Value Imputation step added to pipeline")
                        st.session_state.steps_version += 1
                        st.rerun()
                
                elif preproc_type == "Skewness Correction":
                    column = st.selectbox(
                        "Column to Transform", 
                        options=non_target_columns
                    )
                    method = st.selectbox("Transformation Method", ["log", "sqrt", "box_cox", "exp"])
                    
                    if st.button("Add Skewness Correction Step"):
                        step_name = f"skew_{len(st.session_state.pipeline_steps)}"
                        step = (step_name, SkewnessCorrector(method=method, column=column))
                        st.session_state.pipeline_steps.append(step)
                        st.success("Skewness Correction step added to pipeline")
                        st.session_state.steps_version += 1
                        st.rerun()
            
                elif preproc_type == "Binning":
                    column = st.selectbox(
                        "Column to Bin",
                        options=non_target_columns
                    )
                    method = st.selectbox("Binning Method", ["quantile", "custom"])
                    if method == "quantile":
                        n_bins = st.slider("Number of Bins", 2, 20, 5)
                        if st.button("Add Binning Step", key="add_bin_quant"):
                            step_name = f"bin_{len(st.session_state.pipeline_steps)}"
                            step = (step_name, Binner(method=method, column=column, n_bins=n_bins))
                            st.session_state.pipeline_steps.append(step)
                            st.success("Binning step added to pipeline")
                            st.session_state.steps_version += 1
                            st.rerun()
                    else:
                        bins_str = st.text_input("Custom bin edges (comma-separated)", "0,10,20,50,100")
                        labels_str = st.text_input("Labels (optional, comma-separated)", "")
                        custom_bins = [float(x.strip()) for x in bins_str.split(",") if x.strip() != ""]
                        labels = [s.strip() for s in labels_str.split(",") if s.strip() != ""] if labels_str else None
                        if st.button("Add Binning Step", key="add_bin_custom"):
                            if len(custom_bins) < 2:
                                st.error("Please provide at least two bin edges.")
                            else:
                                step_name = f"bin_{len(st.session_state.pipeline_steps)}"
                                step = (step_name, Binner(method=method, column=column, custom_bins=custom_bins, labels=labels))
                                st.session_state.pipeline_steps.append(step)
                                st.success("Binning step added to pipeline")
                                st.session_state.steps_version += 1
                                st.rerun()
            
            with tab_feat:
                column = st.selectbox(
                    "Column to Encode", 
                    options=non_target_columns
                )
                method = st.selectbox("Encoding Method", ["label", "onehot"])
                
                if st.button("Add Encoding Step"):
                    step_name = f"encoder_{len(st.session_state.pipeline_steps)}"
                    step = (step_name, FeatureEncoder(method=method, column=column))
                    st.session_state.pipeline_steps.append(step)
                    st.success("Feature Encoding step added to pipeline")
                    st.session_state.steps_version += 1
                    st.rerun()
            
            with tab_model:
                model_types = [
                    "Logistic Regression", 
                    "Random Forest", 
                    "Support Vector Machine", 
                    "Gradient Boosting",
                    "LightGBM" if LIGHTGBM_AVAILABLE else "LightGBM (unavailable)",
                    "CatBoost" if CATBOOST_AVAILABLE else "CatBoost (unavailable)",
                    "Weighted Ensemble",
                    "Meta Learner"
                ]
                model_type = st.selectbox("Model Type", model_types)
                
                if model_type == "Logistic Regression":
                    C = st.slider("Regularization (C)", 0.01, 10.0, 1.0)
                    max_iter = st.slider("Max Iterations", 100, 1000, 100, 100)
                    
                    if st.button("Add Logistic Regression Model"):
                        step_name = "model"
                        step = (step_name, LogisticRegression(C=C, max_iter=max_iter, random_state=42))
                        st.session_state.pipeline_steps.append(step)
                        st.success("Logistic Regression model added to pipeline")
                        st.session_state.steps_version += 1
                        st.rerun()
                
                elif model_type == "Random Forest":
                    n_estimators = st.slider("Number of Trees", 10, 200, 100, 10)
                    max_depth = st.slider("Max Tree Depth", 1, 32, 5)
                    
                    if st.button("Add Random Forest Model"):
                        step_name = "model"
                        step = (
                            step_name, 
                            RandomForestClassifier(
                                n_estimators=n_estimators, 
                                max_depth=max_depth, 
                                random_state=42
                            )
                        )
                        st.session_state.pipeline_steps.append(step)
                        st.success("Random Forest model added to pipeline")
                        st.session_state.steps_version += 1
                        st.rerun()
                
                elif model_type == "Support Vector Machine":
                    kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
                    C = st.slider("Regularization (C)", 0.01, 10.0, 1.0)
                    
                    if st.button("Add SVM Model"):
                        step_name = "model"
                        step = (step_name, SVC(kernel=kernel, C=C, probability=True, random_state=42))
                        st.session_state.pipeline_steps.append(step)
                        st.success("SVM model added to pipeline")
                        st.session_state.steps_version += 1
                        st.rerun()
                
                elif model_type == "Gradient Boosting":
                    n_estimators = st.slider("Number of Estimators", 50, 500, 100, 50)
                    learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
                    
                    if st.button("Add Gradient Boosting Model"):
                        step_name = "model"
                        step = (
                            step_name, 
                            GradientBoostingClassifier(
                                n_estimators=n_estimators, 
                                learning_rate=learning_rate, 
                                random_state=42
                            )
                        )
                        st.session_state.pipeline_steps.append(step)
                        st.success("Gradient Boosting model added to pipeline")
                        st.session_state.steps_version += 1
                        st.rerun()
        
                elif model_type.startswith("LightGBM"):
                    if not LIGHTGBM_AVAILABLE:
                        st.error("LightGBM is not installed. Please install dependencies.")
                    else:
                        n_estimators = st.slider("Number of Estimators", 50, 1000, 200, 50)
                        num_leaves = st.slider("Num Leaves", 8, 256, 31)
                        learning_rate = st.slider("Learning Rate", 0.001, 0.5, 0.1)
                        if st.button("Add LightGBM Model"):
                            step_name = "model"
                            step = (
                                step_name,
                                LGBMClassifier(
                                    n_estimators=n_estimators,
                                    num_leaves=num_leaves,
                                    learning_rate=learning_rate,
                                    random_state=42
                                )
                            )
                            st.session_state.pipeline_steps.append(step)
                            st.success("LightGBM model added to pipeline")
                            st.session_state.steps_version += 1
                            st.rerun()

                elif model_type.startswith("CatBoost"):
                    if not CATBOOST_AVAILABLE:
                        st.error("CatBoost is not installed. Please install dependencies.")
                    else:
                        n_estimators = st.slider("Number of Estimators", 50, 1000, 200, 50)
                        depth = st.slider("Depth", 2, 10, 6)
                        learning_rate = st.slider("Learning Rate", 0.001, 0.5, 0.1)
                        if st.button("Add CatBoost Model"):
                            step_name = "model"
                            step = (
                                step_name,
                                CatBoostClassifier(
                                    iterations=n_estimators,
                                    depth=depth,
                                    learning_rate=learning_rate,
                                    loss_function='Logloss',
                                    verbose=False,
                                    random_state=42
                                )
                            )
                            st.session_state.pipeline_steps.append(step)
                            st.success("CatBoost model added to pipeline")
                            st.session_state.steps_version += 1
                            st.rerun()

                elif model_type == "Weighted Ensemble":
                    st.write("Build a weighted ensemble by adding base learners with weights.")
                    base_choice = st.selectbox("Base Model", [
                        "Logistic Regression", "Random Forest", "Support Vector Machine", "Gradient Boosting"
                    ] + (["LightGBM"] if LIGHTGBM_AVAILABLE else []) + (["CatBoost"] if CATBOOST_AVAILABLE else []))
                    weight = st.slider("Weight", 0.0, 5.0, 1.0, 0.1)
                    # Hyperparams per base
                    estimator = None
                    if base_choice == "Logistic Regression":
                        C = st.slider("C (LR)", 0.01, 10.0, 1.0, 0.01)
                        estimator = LogisticRegression(C=C, max_iter=500, random_state=42)
                    elif base_choice == "Random Forest":
                        n_estimators = st.slider("n_estimators (RF)", 10, 300, 100, 10)
                        max_depth = st.slider("max_depth (RF)", 1, 50, 10)
                        estimator = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                    elif base_choice == "Support Vector Machine":
                        kernel = st.selectbox("kernel (SVM)", ["linear", "rbf", "poly"])
                        C = st.slider("C (SVM)", 0.01, 10.0, 1.0, 0.01)
                        estimator = SVC(kernel=kernel, C=C, probability=True, random_state=42)
                    elif base_choice == "Gradient Boosting":
                        n_estimators = st.slider("n_estimators (GB)", 50, 500, 100, 50)
                        learning_rate = st.slider("learning_rate (GB)", 0.01, 0.5, 0.1, 0.01)
                        estimator = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
                    elif base_choice == "LightGBM" and LIGHTGBM_AVAILABLE:
                        n_estimators = st.slider("n_estimators (LGBM)", 50, 1000, 200, 50)
                        num_leaves = st.slider("num_leaves (LGBM)", 8, 256, 31)
                        learning_rate = st.slider("learning_rate (LGBM)", 0.001, 0.5, 0.1)
                        estimator = LGBMClassifier(n_estimators=n_estimators, num_leaves=num_leaves, learning_rate=learning_rate, random_state=42)
                    elif base_choice == "CatBoost" and CATBOOST_AVAILABLE:
                        n_estimators = st.slider("iterations (CatBoost)", 50, 1000, 200, 50)
                        depth = st.slider("depth (CatBoost)", 2, 10, 6)
                        learning_rate = st.slider("learning_rate (CatBoost)", 0.001, 0.5, 0.1)
                        estimator = CatBoostClassifier(iterations=n_estimators, depth=depth, learning_rate=learning_rate, loss_function='Logloss', verbose=False, random_state=42)

                    if st.button("Add Base to Ensemble"):
                        if estimator is None:
                            st.error("Please configure a valid base estimator.")
                        else:
                            name = f"ens_{base_choice.lower().replace(' ', '_')}_{len(st.session_state.ensemble_estimators)}"
                            st.session_state.ensemble_estimators.append((name, estimator))
                            st.session_state.ensemble_weights.append(weight)
                            st.success(f"Added base learner '{name}' with weight {weight}")

                    if st.session_state.ensemble_estimators:
                        st.write("Current base learners:")
                        for (name, _), w in zip(st.session_state.ensemble_estimators, st.session_state.ensemble_weights):
                            st.text(f"- {name} (weight={w})")
                        cols = st.columns(2)
                        with cols[0]:
                            if st.button("Clear Ensemble Build", key="ens_clear"):
                                st.session_state.ensemble_estimators = []
                                st.session_state.ensemble_weights = []
                        with cols[1]:
                            if st.button("Add Weighted Ensemble Model", key="ens_add_model"):
                                step_name = "model"
                                model = WeightedEnsembleClassifier(
                                    estimators=st.session_state.ensemble_estimators,
                                    weights=st.session_state.ensemble_weights
                                )
                                st.session_state.pipeline_steps.append((step_name, model))
                                st.success("Weighted Ensemble model added to pipeline")
                                st.session_state.ensemble_estimators = []
                                st.session_state.ensemble_weights = []
                                st.session_state.steps_version += 1
                                st.rerun()

                elif model_type == "Meta Learner":
                    st.write("Build a stacking meta-learner with base learners and a final estimator.")
                    base_choice = st.selectbox("Base Learner", [
                        "Logistic Regression", "Random Forest", "Support Vector Machine", "Gradient Boosting"
                    ] + (["LightGBM"] if LIGHTGBM_AVAILABLE else []) + (["CatBoost"] if CATBOOST_AVAILABLE else []))
                    estimator = None
                    if base_choice == "Logistic Regression":
                        C = st.slider("C (LR)", 0.01, 10.0, 1.0, 0.01)
                        estimator = LogisticRegression(C=C, max_iter=500, random_state=42)
                    elif base_choice == "Random Forest":
                        n_estimators = st.slider("n_estimators (RF)", 10, 300, 100, 10)
                        max_depth = st.slider("max_depth (RF)", 1, 50, 10)
                        estimator = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                    elif base_choice == "Support Vector Machine":
                        kernel = st.selectbox("kernel (SVM)", ["linear", "rbf", "poly"])
                        C = st.slider("C (SVM)", 0.01, 10.0, 1.0, 0.01)
                        estimator = SVC(kernel=kernel, C=C, probability=True, random_state=42)
                    elif base_choice == "Gradient Boosting":
                        n_estimators = st.slider("n_estimators (GB)", 50, 500, 100, 50)
                        learning_rate = st.slider("learning_rate (GB)", 0.01, 0.5, 0.1, 0.01)
                        estimator = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
                    elif base_choice == "LightGBM" and LIGHTGBM_AVAILABLE:
                        n_estimators = st.slider("n_estimators (LGBM)", 50, 1000, 200, 50)
                        num_leaves = st.slider("num_leaves (LGBM)", 8, 256, 31)
                        learning_rate = st.slider("learning_rate (LGBM)", 0.001, 0.5, 0.1)
                        estimator = LGBMClassifier(n_estimators=n_estimators, num_leaves=num_leaves, learning_rate=learning_rate, random_state=42)
                    elif base_choice == "CatBoost" and CATBOOST_AVAILABLE:
                        n_estimators = st.slider("iterations (CatBoost)", 50, 1000, 200, 50)
                        depth = st.slider("depth (CatBoost)", 2, 10, 6)
                        learning_rate = st.slider("learning_rate (CatBoost)", 0.001, 0.5, 0.1)
                        estimator = CatBoostClassifier(iterations=n_estimators, depth=depth, learning_rate=learning_rate, loss_function='Logloss', verbose=False, random_state=42)

                    if st.button("Add Base Learner"):
                        if estimator is None:
                            st.error("Please configure a valid base estimator.")
                        else:
                            name = f"stack_{base_choice.lower().replace(' ', '_')}_{len(st.session_state.stacking_estimators)}"
                            st.session_state.stacking_estimators.append((name, estimator))
                            st.success(f"Added base learner '{name}'")

                    st.write("Final estimator")
                    final_choice = st.selectbox("Final Estimator", [
                        "Logistic Regression", "Random Forest", "Gradient Boosting"
                    ] + (["LightGBM"] if LIGHTGBM_AVAILABLE else []) + (["CatBoost"] if CATBOOST_AVAILABLE else []))
                    final_est = None
                    if final_choice == "Logistic Regression":
                        C = st.slider("C (final LR)", 0.01, 10.0, 1.0, 0.01)
                        final_est = LogisticRegression(C=C, max_iter=500, random_state=42)
                    elif final_choice == "Random Forest":
                        n_estimators = st.slider("n_estimators (final RF)", 10, 300, 100, 10)
                        max_depth = st.slider("max_depth (final RF)", 1, 50, 10)
                        final_est = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                    elif final_choice == "Gradient Boosting":
                        n_estimators = st.slider("n_estimators (final GB)", 50, 500, 100, 50)
                        learning_rate = st.slider("learning_rate (final GB)", 0.01, 0.5, 0.1, 0.01)
                        final_est = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
                    elif final_choice == "LightGBM" and LIGHTGBM_AVAILABLE:
                        n_estimators = st.slider("n_estimators (final LGBM)", 50, 1000, 200, 50)
                        num_leaves = st.slider("num_leaves (final LGBM)", 8, 256, 31)
                        learning_rate = st.slider("learning_rate (final LGBM)", 0.001, 0.5, 0.1)
                        final_est = LGBMClassifier(n_estimators=n_estimators, num_leaves=num_leaves, learning_rate=learning_rate, random_state=42)
                    elif final_choice == "CatBoost" and CATBOOST_AVAILABLE:
                        n_estimators = st.slider("iterations (final CatBoost)", 50, 1000, 200, 50)
                        depth = st.slider("depth (final CatBoost)", 2, 10, 6)
                        learning_rate = st.slider("learning_rate (final CatBoost)", 0.001, 0.5, 0.1)
                        final_est = CatBoostClassifier(iterations=n_estimators, depth=depth, learning_rate=learning_rate, loss_function='Logloss', verbose=False, random_state=42)

                    passthrough = st.checkbox("Passthrough original features", value=False)
                    cv = st.slider("CV folds", 2, 10, 5)

                    cols = st.columns(2)
                    with cols[0]:
                        if st.button("Clear Meta Learner Build", key="stack_clear"):
                            st.session_state.stacking_estimators = []
                            st.session_state.stacking_final = None
                    with cols[1]:
                        if st.button("Add Stacking Meta Learner Model", key="stack_add_model"):
                            if not st.session_state.stacking_estimators or final_est is None:
                                st.error("Please add at least one base learner and select a final estimator.")
                            else:
                                step_name = "model"
                                model = StackingMetaLearner(
                                    estimators=st.session_state.stacking_estimators,
                                    final_estimator=final_est,
                                    passthrough=passthrough,
                                    cv=cv
                                )
                                st.session_state.pipeline_steps.append((step_name, model))
                                st.success("Stacking Meta Learner model added to pipeline")
                                st.session_state.stacking_estimators = []
                                st.session_state.stacking_final = None
                                st.session_state.steps_version += 1
                                st.rerun()
        
        with right:
            st.subheader("Current Pipeline Steps")
            if st.session_state.pipeline_steps:
                # Card-style summary of steps
                st.markdown(
                    """
                    <style>
                    .step-card {border:1px solid rgba(128,128,128,0.20);border-radius:8px;padding:12px 14px;margin-bottom:10px;background:var(--secondary-background-color, var(--background-color));box-shadow:0 1px 2px rgba(0,0,0,0.04);} 
                    /* Default to light theme-friendly colors */
                    .step-title {font-weight:600;margin-bottom:4px;color:#111827;} /* gray-900 */
                    .step-meta {font-size:13px;color:#374151;opacity:0.80;} /* gray-700 */
                    .badge {display:inline-block;background:var(--primary-color, #4C78FF);color:#ffffff;border-radius:6px;padding:2px 8px;margin-right:8px;font-size:12px;}

                    /* Prefer explicit overrides to avoid inheriting Streamlit theme vars */
                    @media (prefers-color-scheme: dark) {
                      .step-card {border-color: rgba(255,255,255,0.18); box-shadow: 0 1px 2px rgba(0,0,0,0.2);} 
                      .step-title {color:#F9FAFB;} /* near-white */
                      .step-meta {color:#E5E7EB; opacity:0.90;} /* gray-200 */
                    }
                    @media (prefers-color-scheme: light) {
                      .step-card {border-color: rgba(0,0,0,0.08);} 
                      .step-title {color:#111827;} /* gray-900 */
                      .step-meta {color:#374151; opacity:0.80;} /* gray-700 */
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
            for i, (name, step) in enumerate(st.session_state.pipeline_steps):
                step_type = type(step).__name__
                col_applied = getattr(step, 'column', None)
                # Derive a human-readable method for preprocessing/encoding steps
                method_val = None
                if hasattr(step, 'impute_method'):
                    method_val = getattr(step, 'impute_method', None)
                elif hasattr(step, 'method'):
                    method_val = getattr(step, 'method', None)
                    # For binner, add bin count/context if available
                    if step_type == 'Binner' and method_val == 'quantile':
                        n_bins = getattr(step, 'n_bins', None)
                        if n_bins is not None:
                            method_val = f"quantile (bins={n_bins})"
                    elif step_type == 'Binner' and method_val == 'custom':
                        custom_bins = getattr(step, 'custom_bins', None)
                        if custom_bins is not None and hasattr(custom_bins, '__len__') and len(custom_bins) >= 2:
                            method_val = f"custom (bins={len(custom_bins)-1})"
                left_col, right_col = st.columns([5, 1])
                with left_col:
                    meta_parts = []
                    if step_type.endswith('Bias'):
                        if step_type == 'RepresentationBias':
                            prot = getattr(step, 'protected_attribute', None)
                            grp = getattr(step, 'underrepresented_group', None)
                            red = getattr(step, 'reduction_factor', None)
                            if prot is not None:
                                meta_parts.append(f"Protected: {prot}")
                            if grp is not None:
                                meta_parts.append(f"Group: {grp}")
                            if red is not None:
                                meta_parts.append(f"Reduction: {red}")
                        elif step_type == 'MeasurementBias':
                            feat = getattr(step, 'feature', None)
                            prot = getattr(step, 'protected_attribute', None)
                            grp = getattr(step, 'affected_group', None)
                            noise = getattr(step, 'noise_std', None)
                            if feat is not None:
                                meta_parts.append(f"Feature: {feat}")
                            if prot is not None:
                                meta_parts.append(f"Protected: {prot}")
                            if grp is not None:
                                meta_parts.append(f"Group: {grp}")
                            if noise is not None:
                                meta_parts.append(f"Noise: {noise}")
                        elif step_type == 'SamplingBias':
                            prot = getattr(step, 'protected_attribute', None)
                            tgt = getattr(step, 'target_column', None)
                            grp = getattr(step, 'affected_group', None)
                            lbl = getattr(step, 'label_value', None)
                            rate = getattr(step, 'sampling_rate', None)
                            if prot is not None:
                                meta_parts.append(f"Protected: {prot}")
                            if tgt is not None:
                                meta_parts.append(f"Target: {tgt}")
                            if grp is not None:
                                meta_parts.append(f"Group: {grp}")
                            if lbl is not None:
                                meta_parts.append(f"Label: {lbl}")
                            if rate is not None:
                                meta_parts.append(f"Rate: {rate}")
                        elif step_type == 'LabelBias':
                            prot = getattr(step, 'protected_attribute', None)
                            tgt = getattr(step, 'target_column', None)
                            grp = getattr(step, 'affected_group', None)
                            flip = getattr(step, 'flip_probability', None)
                            if prot is not None:
                                meta_parts.append(f"Protected: {prot}")
                            if tgt is not None:
                                meta_parts.append(f"Target: {tgt}")
                            if grp is not None:
                                meta_parts.append(f"Group: {grp}")
                            if flip is not None:
                                meta_parts.append(f"Flip: {flip}")
                    else:
                        if col_applied is not None:
                            meta_parts.append(f"Column: {col_applied}")
                        if method_val is not None:
                            meta_parts.append(f"Method: {method_val}")

                    meta_html_items = ''.join([f"<span>• {p}</span>" for p in meta_parts])
                    html = (
                        f"<div class='step-card'>"
                        f"<div class='step-title'>#{i+1} {name} <span class='badge' style='margin-left:8px;'>{step_type}</span></div>"
                        f"<div class='step-meta'>"
                        f"{meta_html_items}"
                        f"</div>"
                        f"</div>"
                    )
                    st.markdown(html, unsafe_allow_html=True)
                with right_col:
                    if st.button("❌", key=f"remove_step_{i}"):
                        st.session_state.pipeline_steps.pop(i)
                        st.rerun()
            if len(st.session_state.pipeline_steps) == 0:
                st.info("No steps added yet.")
            
            # Clear all steps button
            if st.button("Clear All Steps", key="clear_all_steps"):
                st.session_state.pipeline_steps = []
                st.rerun()
            
            # Train pipeline button
            if st.button("Train Pipeline", key="train_pipeline"):
                if st.session_state.train_data is not None:
                    if st.session_state.target_column not in st.session_state.train_data.columns:
                        st.error(f"Selected target column '{st.session_state.target_column}' not found in training data. Available columns: {list(st.session_state.train_data.columns)}")
                    else:
                        # Validate that no step will remove or overwrite the target column
                        invalid_steps = []
                        for step_name, step in st.session_state.pipeline_steps[:-1]:
                            # If a step targets the same column as the selected target, block (encoders/binning/etc.)
                            step_col = getattr(step, 'column', None)
                            if step_col == st.session_state.target_column:
                                invalid_steps.append(f"{step_name} ({type(step).__name__})")
                        if invalid_steps:
                            st.error("These steps operate on the target column and would break training: " + ", ".join(invalid_steps) + ". Please remove or reconfigure them to use a feature column.")
                        else:
                            try:
                                pipeline = Pipeline(steps=st.session_state.pipeline_steps)
                                pipeline.fit(st.session_state.train_data, st.session_state.target_column)
                                st.session_state.trained_pipeline = pipeline
                                st.success("Pipeline trained successfully!")
                            except Exception as e:
                                st.error(f"Error training pipeline: {str(e)}")
                else:
                    st.error("Please split your dataset first")
            
            if st.session_state.trained_pipeline is not None:
                if st.button("Proceed to Results Visualization", key="go_results"):
                    st.session_state['pending_app_mode'] = "Results Visualization"
                    st.rerun()

# Function to visualize pipeline results
def visualize_results():
    st.subheader("Pipeline Visualization")
    
    if st.session_state.trained_pipeline is None:
        st.warning("Please train a pipeline first")
        return
    
    # Draw pipeline diagram (Graphviz)
    if st.session_state.pipeline_steps:
        def categorize_and_color(step_obj):
            cls = type(step_obj).__name__
            if cls.endswith('Bias'):
                return 'Bias', '#fde68a'
            if cls in ['MissingImputer', 'SkewnessCorrector', 'Binner']:
                return 'Preprocess', '#bfdbfe'
            if cls in ['FeatureEncoder']:
                return 'Feature', '#c7f9cc'
            return 'Model', '#e9d5ff'

        dot_lines = [
            'digraph G {',
            'rankdir=LR;',
            'node [shape=box, style=filled, fontname="Helvetica"];'
        ]

        # Nodes
        for i, (name, step) in enumerate(st.session_state.pipeline_steps):
            category, color = categorize_and_color(step)
            col_applied = getattr(step, 'column', None)
            method_val = None
            if hasattr(step, 'impute_method'):
                method_val = getattr(step, 'impute_method', None)
            elif hasattr(step, 'method'):
                method_val = getattr(step, 'method', None)
                if type(step).__name__ == 'Binner' and method_val == 'quantile':
                    n_bins = getattr(step, 'n_bins', None)
                    if n_bins is not None:
                        method_val = f"quantile (bins={n_bins})"
                elif type(step).__name__ == 'Binner' and method_val == 'custom':
                    custom_bins = getattr(step, 'custom_bins', None)
                    if custom_bins is not None and hasattr(custom_bins, '__len__') and len(custom_bins) >= 2:
                        method_val = f"custom (bins={len(custom_bins)-1})"
            label_lines = [name, f"[{category}]"]
            if col_applied is not None:
                label_lines.append(f"col: {col_applied}")
            if method_val is not None:
                label_lines.append(f"method: {method_val}")
            label = '\n'.join(label_lines).replace('"', '\"')
            dot_lines.append(f's{i} [label="{label}", fillcolor="{color}"];')

        # Edges
        for i in range(len(st.session_state.pipeline_steps) - 1):
            dot_lines.append(f's{i} -> s{i+1};')

        dot_lines.append('}')
        dot_src = '\n'.join(dot_lines)

        st.graphviz_chart(dot_src, use_container_width=True)
        
        # Legend
        st.markdown(
            """
            <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:6px;">
              <span style="display:inline-flex;align-items:center;gap:6px;"><span style="width:12px;height:12px;background:#bfdbfe;display:inline-block;border-radius:3px;"></span>Preprocess</span>
              <span style="display:inline-flex;align-items:center;gap:6px;"><span style="width:12px;height:12px;background:#c7f9cc;display:inline-block;border-radius:3px;"></span>Feature</span>
              <span style="display:inline-flex;align-items:center;gap:6px;"><span style="width:12px;height:12px;background:#fde68a;display:inline-block;border-radius:3px;"></span>Bias</span>
              <span style="display:inline-flex;align-items:center;gap:6px;"><span style="width:12px;height:12px;background:#e9d5ff;display:inline-block;border-radius:3px;"></span>Model</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Model Evaluation
    if st.session_state.trained_pipeline is not None and (st.session_state.val_data is not None or st.session_state.test_data is not None):
        st.subheader("Model Evaluation")

        # Metric selection
        metric_options = [
            "Accuracy",
            "Precision (weighted)",
            "Recall (weighted)",
            "F1 (weighted)",
            "ROC AUC",
            "PR AUC",
            "Log Loss",
            "Confusion Matrix",
            "ROC Curve",
            "PR Curve",
            "Classification Report",
            "Disparity (Demographic Parity Ratio)",
        ]
        default_metrics = ["Accuracy", "F1 (weighted)", "ROC AUC", "Confusion Matrix"]
        selected_metrics = st.multiselect("Select metrics to compute", metric_options, default=default_metrics)
        
        try:
            eval_choice = "Validation" if st.session_state.val_data is not None else "Test"
            if st.session_state.val_data is not None and st.session_state.test_data is not None:
                eval_choice = st.radio("Evaluate on", ["Validation", "Test"], horizontal=True)
            eval_df = st.session_state.val_data if eval_choice == "Validation" else st.session_state.test_data
            # Align labels with transformed data in case steps drop rows
            transformed_eval = st.session_state.trained_pipeline.transform(eval_df)
            if st.session_state.target_column not in transformed_eval.columns:
                st.error(f"Target column '{st.session_state.target_column}' missing after transformations.")
                return
            X_eval = transformed_eval.drop(columns=[st.session_state.target_column])
            y_true = transformed_eval[st.session_state.target_column]
            y_pred = st.session_state.trained_pipeline.model.predict(X_eval)

            # Attempt probabilities if any prob-based metrics are selected
            needs_proba = any(m in selected_metrics for m in ["ROC AUC", "PR AUC", "Log Loss", "ROC Curve", "PR Curve"]) 
            y_proba = None
            if needs_proba:
                try:
                    y_proba = st.session_state.trained_pipeline.model.predict_proba(X_eval)[:, 1]
                except Exception:
                    y_proba = None

            # If using probabilities and target labels are non-binary strings, binarize y_true
            y_true_for_prob = y_true
            pos_label_name = None
            if y_proba is not None:
                unique_vals = pd.Series(y_true).dropna().unique().tolist()
                if not set(unique_vals) <= {0, 1, -1} and hasattr(st.session_state.trained_pipeline.model, 'classes_') and len(st.session_state.trained_pipeline.model.classes_) == 2:
                    pos_label_name = st.session_state.trained_pipeline.model.classes_[1]
                    y_true_for_prob = (y_true == pos_label_name).astype(int)

            # Scalar metrics (compute once per selection)
            cols_top = st.columns(3)
            col_idx = 0
            summary_vals = {}
            if "Accuracy" in selected_metrics:
                acc = compute_accuracy(y_true, y_pred)
                summary_vals["accuracy"] = acc
                cols_top[col_idx % 3].metric("Accuracy", f"{acc:.4f}")
                col_idx += 1
            if "Precision (weighted)" in selected_metrics:
                prec = compute_precision_weighted(y_true, y_pred)
                summary_vals["precision_weighted"] = prec
                cols_top[col_idx % 3].metric("Precision (w)", f"{prec:.4f}")
                col_idx += 1
            if "Recall (weighted)" in selected_metrics:
                rec = compute_recall_weighted(y_true, y_pred)
                summary_vals["recall_weighted"] = rec
                cols_top[col_idx % 3].metric("Recall (w)", f"{rec:.4f}")
                col_idx += 1
            if "F1 (weighted)" in selected_metrics:
                f1w = compute_f1_weighted(y_true, y_pred)
                summary_vals["f1_weighted"] = f1w
                cols_top[col_idx % 3].metric("F1 (weighted)", f"{f1w:.4f}")
                col_idx += 1
            if "Log Loss" in selected_metrics and y_proba is not None:
                ll = compute_log_loss(y_true_for_prob, y_proba)
                summary_vals["log_loss"] = ll
                cols_top[col_idx % 3].metric("Log Loss", f"{ll:.4f}")
                col_idx += 1
            if "ROC AUC" in selected_metrics and y_proba is not None:
                rocauc = compute_roc_auc(y_true_for_prob, y_proba)
                summary_vals["roc_auc"] = rocauc
                cols_top[col_idx % 3].metric("ROC AUC", f"{rocauc:.4f}")
                col_idx += 1
            if "PR AUC" in selected_metrics and y_proba is not None:
                prauc = compute_pr_auc(y_true_for_prob, y_proba)
                summary_vals["pr_auc"] = prauc
                cols_top[col_idx % 3].metric("PR AUC", f"{prauc:.4f}")
                col_idx += 1

            # Plots
            plot_cols = st.columns(2)
            if "Confusion Matrix" in selected_metrics:
                cm = compute_confusion_matrix(y_true, y_pred)
                with plot_cols[0]:
                    st.write("Confusion Matrix")
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    st.pyplot(fig)
            if "ROC Curve" in selected_metrics and y_proba is not None:
                fpr, tpr, _, auc_val = compute_roc_curve(y_true_for_prob, y_proba)
                with plot_cols[1]:
                    st.write("ROC Curve")
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.plot(fpr, tpr, lw=2, label=f'AUC = {auc_val:.2f}')
                    ax.plot([0, 1], [0, 1], 'k--', lw=1)
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.legend(loc="lower right")
                    st.pyplot(fig)
            if "PR Curve" in selected_metrics and y_proba is not None:
                precision, recall, _, ap = compute_pr_curve(y_true_for_prob, y_proba)
                with plot_cols[1]:
                    st.write("Precision-Recall Curve")
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.plot(recall, precision, lw=2, label=f'AP = {ap:.2f}')
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('Recall')
                    ax.set_ylabel('Precision')
                    ax.legend(loc="lower left")
                    st.pyplot(fig)

            # Classification report
            if "Classification Report" in selected_metrics:
                rpt = generate_classification_report(y_true, y_pred)
                rpt_df = pd.DataFrame(rpt).transpose()
                st.write("Classification Report")
                st.dataframe(rpt_df)

            # Fairness metrics
            fairness_context = None
            if "Disparity (Demographic Parity Ratio)" in selected_metrics:
                # Prefer binary columns from transformed_eval to ensure alignment
                binary_candidates = [c for c in transformed_eval.columns if transformed_eval[c].nunique(dropna=True) == 2]
                # Fallback to original eval df if none found (we'll align by index later)
                if not binary_candidates:
                    binary_candidates = [c for c in eval_df.columns if eval_df[c].nunique(dropna=True) == 2]

                if not binary_candidates:
                    st.info("No binary column found for fairness evaluation.")
                else:
                    default_choice = None
                    for cand in ["protected_attribute", "sex"]:
                        if cand in binary_candidates:
                            default_choice = cand
                            break
                    if default_choice is None:
                        default_choice = binary_candidates[0]

                    prot_col = st.selectbox(
                        "Protected attribute column",
                        options=binary_candidates,
                        index=binary_candidates.index(default_choice) if default_choice in binary_candidates else 0,
                        key="protected_attr_choice_eval"
                    )
                    # Optional: allow probability thresholding for fairness if probabilities available
                    y_pred_for_fair = y_pred
                    if y_proba is not None:
                        use_thresh = st.checkbox("Use probability threshold for fairness", value=False, key="fair_use_thresh")
                        if use_thresh:
                            thr = st.slider("Probability threshold (for fairness)", 0.0, 1.0, 0.5, 0.01, key="fair_threshold")
                            y_pred_for_fair = (y_proba >= thr).astype(int)

                    # Get aligned protected attribute series
                    if prot_col in transformed_eval.columns:
                        prot_series = transformed_eval[prot_col]
                    else:
                        # Align by index to transformed rows
                        prot_series = eval_df.loc[transformed_eval.index, prot_col]
                    dpr = compute_demographic_parity_ratio(y_pred_for_fair, prot_series)
                    # Derive group stats for context and handle undefined denominators
                    series = prot_series.copy()
                    if series.dtype == bool:
                        series = series.astype(int)
                    uniques = pd.Series(series.unique()).dropna().tolist()
                    if len(uniques) == 2 and not set(uniques) <= {0, 1}:
                        mapping = {uniques[0]: 0, uniques[1]: 1}
                        series = series.map(mapping)
                    mask0 = series == 0
                    mask1 = series == 1
                    n0 = int(mask0.sum())
                    n1 = int(mask1.sum())
                    p_y1_g0 = float((y_pred_for_fair[mask0] == 1).mean()) if mask0.any() else float("nan")
                    p_y1_g1 = float((y_pred_for_fair[mask1] == 1).mean()) if mask1.any() else float("nan")

                    # Display metric with safe fallback
                    if not np.isfinite(dpr):
                        label = "Demographic Parity Ratio"
                        st.metric(label, "undefined")
                        st.caption("Undefined because group A=0 has zero predicted positives or is absent.")
                        if (not np.isnan(p_y1_g0) and p_y1_g0 == 0.0) and (not np.isnan(p_y1_g1) and p_y1_g1 == 0.0) and y_proba is not None:
                            st.info("Both groups have zero predicted positives at the current threshold. Try lowering the threshold or selecting a different split/column.")
                    else:
                        st.metric("Demographic Parity Ratio", f"{dpr:.4f}")
                        st.caption("Values < 0.8 or > 1.25 may indicate disparity")

                    # Show compact group diagnostics below
                    st.write(
                        f"A=0: n={n0}, P(ŷ=1|A=0)={(p_y1_g0 if not np.isnan(p_y1_g0) else float('nan')):.4f}  |  "
                        f"A=1: n={n1}, P(ŷ=1|A=1)={(p_y1_g1 if not np.isnan(p_y1_g1) else float('nan')):.4f}"
                    )

                    # Add DPR into summary metrics (for comparison table)
                    summary_vals["demographic_parity_ratio"] = float("nan") if not np.isfinite(dpr) else dpr
                    fairness_context = {
                        "protected_column": prot_col,
                        "use_threshold": bool(use_thresh) if y_proba is not None else False,
                        "threshold": float(thr) if (y_proba is not None and use_thresh) else None,
                        "group0_n": n0,
                        "group1_n": n1,
                        "p_y1_g0": None if np.isnan(p_y1_g0) else float(p_y1_g0),
                        "p_y1_g1": None if np.isnan(p_y1_g1) else float(p_y1_g1),
                        "dpr": None if not np.isfinite(dpr) else float(dpr),
                    }

            # Save pipeline for comparison
            st.subheader("Save Pipeline for Comparison")
            default_name = f"pipeline_{len(st.session_state.saved_pipelines)+1}"
            pipeline_name = st.text_input("Name", value=default_name)
            if st.button("Save Current Pipeline", key="save_pipeline"):
                # Serialize pipeline configuration
                steps_info = []
                for step_name_i, step_obj in st.session_state.pipeline_steps:
                    step_type_i = type(step_obj).__name__
                    info = {"name": step_name_i, "type": step_type_i}
                    if step_type_i == 'RepresentationBias':
                        info.update({
                            "protected_attribute": getattr(step_obj, 'protected_attribute', None),
                            "underrepresented_group": getattr(step_obj, 'underrepresented_group', None),
                            "reduction_factor": getattr(step_obj, 'reduction_factor', None),
                        })
                    elif step_type_i == 'MeasurementBias':
                        info.update({
                            "feature": getattr(step_obj, 'feature', None),
                            "protected_attribute": getattr(step_obj, 'protected_attribute', None),
                            "affected_group": getattr(step_obj, 'affected_group', None),
                            "noise_std": getattr(step_obj, 'noise_std', None),
                        })
                    elif step_type_i == 'SamplingBias':
                        info.update({
                            "protected_attribute": getattr(step_obj, 'protected_attribute', None),
                            "target_column": getattr(step_obj, 'target_column', None),
                            "affected_group": getattr(step_obj, 'affected_group', None),
                            "label_value": getattr(step_obj, 'label_value', None),
                            "sampling_rate": getattr(step_obj, 'sampling_rate', None),
                        })
                    elif step_type_i == 'LabelBias':
                        info.update({
                            "protected_attribute": getattr(step_obj, 'protected_attribute', None),
                            "target_column": getattr(step_obj, 'target_column', None),
                            "affected_group": getattr(step_obj, 'affected_group', None),
                            "flip_probability": getattr(step_obj, 'flip_probability', None),
                        })
                    elif step_type_i == 'MissingImputer':
                        info.update({"method": getattr(step_obj, 'impute_method', None), "column": getattr(step_obj, 'column', None)})
                    elif step_type_i == 'SkewnessCorrector':
                        info.update({"method": getattr(step_obj, 'method', None), "column": getattr(step_obj, 'column', None)})
                    elif step_type_i == 'Binner':
                        info.update({
                            "method": getattr(step_obj, 'method', None),
                            "column": getattr(step_obj, 'column', None),
                            "n_bins": getattr(step_obj, 'n_bins', None),
                            "custom_bins": getattr(step_obj, 'custom_bins', None),
                            "labels": getattr(step_obj, 'labels', None),
                        })
                    elif step_type_i == 'FeatureEncoder':
                        info.update({"method": getattr(step_obj, 'method', None), "column": getattr(step_obj, 'column', None)})
                    else:
                        # Likely the model step
                        if hasattr(step_obj, 'get_params') and callable(getattr(step_obj, 'get_params')):
                            try:
                                info.update({"params": step_obj.get_params()})
                            except Exception:
                                pass
                    steps_info.append(info)

                st.session_state.saved_pipelines.append({
                    "name": pipeline_name,
                    "metrics": summary_vals,
                    "selected_metrics": selected_metrics,
                    "eval_set": eval_choice,
                    "fairness": fairness_context,
                    "pipeline": {"steps": steps_info}
                })
                st.success(f"Saved '{pipeline_name}' for comparison")
            if st.button("Go to Pipeline Comparison", key="go_compare"):
                st.session_state['pending_app_mode'] = "Pipeline Comparison"
                st.rerun()
                
        except Exception as e:
            st.error(f"Error evaluating model: {str(e)}")

# Main app layout
def compare_pipelines():
    st.subheader("Pipeline Comparison")
    if not st.session_state.saved_pipelines:
        st.info("No saved pipelines yet. Evaluate a trained pipeline and click 'Save Current Pipeline'.")
        return
    # Build comparison table with all saved scalar metrics
    rows = []
    all_metric_keys = set()
    for item in st.session_state.saved_pipelines:
        metric_dict = item.get("metrics", {}) or {}
        all_metric_keys.update(metric_dict.keys())
    all_metric_keys = sorted(list(all_metric_keys))

    for item in st.session_state.saved_pipelines:
        row = {"name": item.get("name", ""), "eval_set": item.get("eval_set", "")}
        metric_dict = item.get("metrics", {}) or {}
        for k in all_metric_keys:
            row[k] = metric_dict.get(k, None)
        rows.append(row)
    df = pd.DataFrame(rows)
    st.dataframe(df)

    names = [item.get("name", f"pipeline_{i+1}") for i, item in enumerate(st.session_state.saved_pipelines)]

    # Side-by-side comparison view
    st.markdown("---")
    st.subheader("Side-by-side Pipeline Details")
    max_cols = len(st.session_state.saved_pipelines)
    if max_cols > 0:
        # Avoid rendering a slider with equal min/max which can cause a RangeError in the frontend
        if max_cols > 1:
            cols_count = st.slider("Number of columns", 1, max_cols, min(3, max_cols), key="compare_cols")
        else:
            cols_count = 1
            st.info("Only one pipeline saved. Showing a single column.")
        selected_names = st.multiselect(
            "Pipelines to display side-by-side",
            options=names,
            default=names[:cols_count],
            key="compare_sel_names"
        )
        if len(selected_names) > cols_count:
            selected_names = selected_names[:cols_count]
        if selected_names:
            cols = st.columns(len(selected_names))
            for idx, nm in enumerate(selected_names):
                item = st.session_state.saved_pipelines[names.index(nm)]
                with cols[idx]:
                    st.markdown(f"### {nm}")
                    st.write("Eval set:", item.get("eval_set", ""))
                    # Metrics table for this pipeline
                    metrics_dict = item.get("metrics", {}) or {}
                    if metrics_dict:
                        mdf = pd.DataFrame([metrics_dict]).T
                        mdf.columns = ["value"]
                        st.dataframe(mdf)
                    # Selected metrics list
                    sel_metrics = item.get("selected_metrics")
                    # if sel_metrics:
                    #     st.write("Selected metrics:", sel_metrics)
                    # Fairness context
                    fair = item.get("fairness")
                    if fair:
                        with st.expander("Fairness context", expanded=False):
                            st.json(fair)
                    # Pipeline steps/config
                    with st.expander("Pipeline configuration", expanded=False):
                        st.json(item.get("pipeline", {}))

    colx, coly = st.columns(2)
    with colx:
        if st.button("Clear Saved Pipelines", key="compare_clear"):
            st.session_state.saved_pipelines = []
            st.success("Cleared saved pipelines")
    with coly:
        st.download_button("Download CSV", data=df.to_csv(index=False), file_name="pipeline_comparison.csv", mime="text/csv", key="compare_download")

def sidebar_progress():
    st.sidebar.title("Navigation")
    has_dataset = st.session_state.get('dataset') is not None
    has_split = st.session_state.get('train_data') is not None and (st.session_state.get('test_data') is not None)
    has_model_step = any(name == 'model' for name, _ in st.session_state.get('pipeline_steps', []))
    has_trained = st.session_state.get('trained_pipeline') is not None
    st.sidebar.markdown(("✅" if has_dataset else "⬜") + " 1. Data")
    st.sidebar.markdown(("✅" if has_split else "⬜") + " 2. Split")
    st.sidebar.markdown(("✅" if has_model_step else "⬜") + " 3. Build")
    st.sidebar.markdown(("✅" if has_trained else "⬜") + " 4. Train & Evaluate")
    st.sidebar.markdown("⬜ 5. Compare" if not st.session_state.get('saved_pipelines') else f"✅ 5. Compare ({len(st.session_state['saved_pipelines'])})")
    if st.sidebar.button("Reset App"):
        keys = list(st.session_state.keys())
        for k in keys:
            del st.session_state[k]
        st.rerun()
    app_mode = st.sidebar.radio("Choose the app mode", ["Data Generation", "Pipeline Configuration", "Results Visualization", "Pipeline Comparison"], key='app_mode')
    return app_mode

app_mode = sidebar_progress()

if app_mode == "Data Generation":
    load_or_generate_data()
elif app_mode == "Pipeline Configuration":
    if st.session_state.dataset is None:
        st.warning("Please generate or upload a dataset first")
    else:
        configure_pipeline()
elif app_mode == "Results Visualization":
    if st.session_state.trained_pipeline is None:
        st.warning("Please configure and train a pipeline first")
    else:
        visualize_results()
elif app_mode == "Pipeline Comparison":
    compare_pipelines()

# Show dataset preview in sidebar
if st.session_state.dataset is not None:
    with st.sidebar.expander("Dataset Preview"):
        st.dataframe(st.session_state.dataset.head())
        st.text(f"Shape: {st.session_state.dataset.shape}")
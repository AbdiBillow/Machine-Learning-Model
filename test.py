import pytest
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError
from io import StringIO
from unittest.mock import patch
from sklearn.linear_model import LinearRegression

# Sample test data
TEST_CSV = """Category,Numeric1,Year,Price
A,10,2020,100
B,,2021,200
C,30,,300
A,40,2023,400"""

def test_data_loading_and_preprocessing():
    """Test data loading and preprocessing pipeline construction"""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    
    # Mock uploaded CSV file
    uploaded_file = StringIO(TEST_CSV)
    df = pd.read_csv(uploaded_file)
    
    # Test data loading
    assert not df.empty, "Data should be loaded successfully"
    assert set(df.columns) == {'Category', 'Numeric1', 'Year', 'Price'}, "Columns should match"

    # Test automatic feature detection
    numeric_features = df.select_dtypes(include=['number']).columns.tolist()
    categorical_features = df.select_dtypes(exclude=['number']).columns.tolist()
    
    assert 'Numeric1' in numeric_features, "Numeric feature should be detected"
    assert 'Category' in categorical_features, "Categorical feature should be detected"

def test_missing_value_handling():
    """Test missing value imputation logic"""
    uploaded_file = StringIO(TEST_CSV)
    df = pd.read_csv(uploaded_file)
    
    # Apply app's missing value handling logic
    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    
    cat_cols = df.select_dtypes(exclude=['number']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode().iloc[0])

    # Check missing values
    assert not df['Numeric1'].isnull().any(), "Numeric missing values should be filled"
    assert not df['Year'].isnull().any(), "Year missing values should be filled"

def test_preprocessing_pipeline():
    """Test the complete preprocessing and model pipeline"""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    
    uploaded_file = StringIO(TEST_CSV)
    df = pd.read_csv(uploaded_file)
    
    # Selected features and target
    selected_features = ['Category', 'Numeric1', 'Year']
    target = 'Price'
    
    # Handle missing values as per app logic
    num_cols = ['Numeric1', 'Year']
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    df[target] = df[target].fillna(df[target].mean())
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), ['Category']),
            ('num', StandardScaler(), num_cols)
        ]
    )
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    # Test pipeline construction
    assert isinstance(model.named_steps['preprocessor'], ColumnTransformer), "Correct preprocessor"
    assert isinstance(model.named_steps['regressor'], LinearRegression), "Correct model"

    # Test model training
    X = df[selected_features]
    y = df[target]
    model.fit(X, y)
    
    assert model.named_steps['regressor'].coef_ is not None, "Model should be trained"

def test_prediction():
    """Test model prediction functionality"""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    
    # Create trained model
    uploaded_file = StringIO(TEST_CSV)
    df = pd.read_csv(uploaded_file)
    
    # Preprocess data
    num_cols = ['Numeric1', 'Year']
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    df['Price'] = df['Price'].fillna(df['Price'].mean())
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Category']),
            ('num', 'passthrough', num_cols)
        ]
    )
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    model.fit(df[['Category', 'Numeric1', 'Year']], df['Price'])
    
    # Create test input
    test_input = pd.DataFrame([{
        'Category': 'A',
        'Numeric1': 25,
        'Year': 2024
    }])
    
    # Make prediction
    prediction = model.predict(test_input)
    assert isinstance(prediction[0], float), "Should return numeric prediction"
    assert prediction[0] > 0, "Prediction should be positive"

def test_error_handling():
    """Test error handling for invalid inputs"""
    from unittest.mock import MagicMock
    
    # Mock Streamlit components
    mock_st = MagicMock()
    mock_st.session_state = {}
    mock_st.error = MagicMock()
    
    # Test empty feature selection
    with patch('streamlit.multiselect', return_value=[]):
        mock_st.button.return_value = True
        # Should call st.error("❌ Please select at least one feature.")
        assert mock_st.error.called

    # Test non-numeric target
    invalid_data = StringIO("Category,Price\nA,High\nB,Low")
    df = pd.read_csv(invalid_data)
    with patch('pandas.api.types.is_numeric_dtype', return_value=False):
        mock_st.selectbox.return_value = 'Price'
        # Should call st.error("❌ Target column must be numeric for regression.")
        assert mock_st.error.called

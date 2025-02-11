# 🔍 Data Cleaning Review: Missing Data, Label Encoding, and Class Removal

Reviews of **potentially problematic code** related to:
- **Missing Data Handling**
- **Label Encoding & Categorical Variables**
- **Data Filtering & Class Removal**


---

## 📌 1. Missing Data Handling

### **Potentially Problematic Code**
```python
# Determine columns to drop based on missing values
columns_to_drop = []
for col in glioma_data.columns:
    total_rows = len(glioma_data)
    empty_or_nan_count = glioma_data[col].apply(lambda x: pd.isna(x) or x == '' or x == 'NaN' or x == '[Not Available]').sum()
    
    if empty_or_nan_count >= 50:  # Arbitrary threshold
        columns_to_drop.append(col)

# Drop identified columns
glioma_data_filtered = glioma_data.drop(columns=columns_to_drop)

# Drop rows with any missing values
glioma_data_filtered = glioma_data_filtered.dropna()
```

### ❓ **Review Questions**
1. **Why was a hard threshold of 50 missing values chosen?**  
   - Would a **percentage-based threshold** (e.g., dropping columns with >30% missing) be better?

2. **Is dropping all rows with NaN values appropriate?**  
   - Would **imputation (mean/median/knn)** be a better?


---

## 📌 2. Label Encoding & Categorical Variables

### **Potentially Problematic Code**
```python
from sklearn.preprocessing import LabelEncoder

# Encode the target column
target_col = 'classification.2021_simplified.labels'
le = LabelEncoder()
glioma_data_filtered[target_col] = le.fit_transform(glioma_data_filtered[target_col])

# Encode categorical variable
encoded_data, label_encoders = encode_categorical_columns(glioma_data_filtered, ['gender'])
```

### ❓ **Review Questions**
1. **Is Label Encoding appropriate for `'classification.2021_simplified.labels'`?**  
   - Should **One-Hot Encoding (OHE)** be used if the classes are **non-ordinal**?

2. **Why was gender encoded with `LabelEncoder()` instead of `OneHotEncoder()`?**  
   - Does numerical representation of gender make sense, or should it be one-hot encoded?

3. **Do we need to handle unknown categories when encoding?**  
   - Should we account for **new categories** appearing in test data?

---

## 📌 3. Data Filtering & Class Removal

### **Potentially Problematic Code**
```python
# Remove a specific class from classification labels
cleaned_glioma_data = encoded_data_frame[encoded_data_frame['classification.2021_simplified.labels'] != 3]

# Drop additional categorical columns
columns_to_drop = ['TCGA-histological.type', 'classification.2021_complete.labels', 'Patient_ID', 
                   'vital_status', 'tumor_status', 'histological_type', 'OS', 'OS.time', 
                   'DSS', 'DSS.time', 'PFI', 'PFI.time']
filtered_glioma_data = cleaned_glioma_data.drop(columns=columns_to_drop)
```

### ❓ **Review Questions**
1. **Why are we removing class `'3'` from `'classification.2021_simplified.labels'`?**  
   - Are we just removing the unknown here?

2. **Are we introducing bias by removing an entire class?**  
   - Will removing class 3 affect **model generalization**?




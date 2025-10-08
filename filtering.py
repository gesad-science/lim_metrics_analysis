import pandas as pd
import ast
import json
import math
from collections import Counter
import os
import traceback

names = [
    'flan_default', 'flan_treatment_mode', 'gemma3_default', 
    'gemma3_treatment_mode', 'llama3_default', 'llama3_treatment_mode', 
    'mistral_default', 'mistral_treatment_mode', 'qwen3_default', 
    'qwen3_treatment_mode', 'phi3_default', 'phi3_treatment_mode'
    ]

def parse_cell(val):
    if pd.isna(val) or not str(val).strip():
        return None
    s = str(val).strip().replace('\\"', "'")
    
    for parser in [json.loads, ast.literal_eval]:
        try: 
            return parser(s)
        except: 
            pass
    
    s_clean = s
    if len(s) >= 6 and ((s[:3] == '"""' and s[-3:] == '"""') or 
                        (s[:3] == "'''" and s[-3:] == "'''")):
        s_clean = s[3:-3]

    elif len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or 
                          (s[0] == "'" and s[-1] == "'")):
        s_clean = s[1:-1]
    
    s_clean = s_clean.replace('""', '"')
    
    for parser in [json.loads, ast.literal_eval]:
        try: 
            return parser(s_clean)
        except: 
            pass
    
    return s_clean

def normalize_value(x):
    if x is None: return None
    if isinstance(x, str): return x.strip().lower()
    if isinstance(x, float) and x.is_integer():return int(x)
    return x        

def normalize_structure(d):
    if d is None: return None
    if isinstance(d, dict): return {normalize_value(k): 
                                    normalize_structure(v) for k, v in d.items()}
    if isinstance(d, list): return [normalize_structure(x) for x in d]
    return normalize_value(d)

def normalize_dataset_values(df):
    df_normalized = df.copy()

    simple_cols = [c for c in ['expected_intent', 'expected_class', 'processed_intent', 
                               'processed_class'] if c in df_normalized.columns]
    for col in simple_cols:
        df_normalized[col] = df_normalized[col].apply(
                lambda x: normalize_value(x) if pd.notna(x) else x
            )
    
    complex_cols = [c for c in ["expected_attributes", "expected_filter_attributes", 
                            "processed_attributes", "processed_filter_attributes"]
                            if c in df_normalized.columns]
    for col in complex_cols:
        df_normalized[col] = df_normalized[col].apply(
            lambda x: normalize_structure(parse_cell(x)) if pd.notna(x) else x
        )

    return df_normalized

def clean_dataset(df):
    df_clean = df.copy()
    
    attr_cols = [c for c in ["expected_attributes", "expected_filter_attributes", 
                            "processed_attributes", "processed_filter_attributes"] 
                if c in df_clean.columns]
    
    for col in attr_cols:
        df_clean[col] = df_clean[col].apply(lambda x: x.replace('\\"', "'") if 
                                            isinstance(x, str) else x)
    
    critical_cols = [c for c in ['expected_intent', 'expected_class'] if 
                     c in df_clean.columns]
    for col in critical_cols:
        df_clean = df_clean[df_clean[col].notna()]
    
    string_cols = [ c for c in ['expected_intent', 'expected_class',
                                 'processed_intent', 'processed_class'] 
                                 if c in df_clean.columns]
    for col in string_cols:
        df_clean[col] = df_clean[col].apply(lambda x: x.strip() if 
                                            isinstance(x, str) else x)
    
    for col in attr_cols:
        df_clean[col] = df_clean[col].apply(lambda x: parse_cell(x) if 
                                            isinstance(x, str) else x)

    df_clean = normalize_dataset_values(df_clean)
    return df_clean

def main():
    for name in names:
        try:
            df_bronze = pd.read_csv(f'datasets/bronze/{name}.csv', dtype=str)
            df_silver = clean_dataset(df_bronze)
            df_silver.to_csv(f'datasets/silver/{name}_cleaned.csv', index=False)
        except Exception as e:
            print(f"Error processing {name}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
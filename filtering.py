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
    'qwen3_treatment_mode'
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
    if len(s) >= 6 and ((s[:3] == '"""' and s[-3:] == '"""') or (s[:3] == "'''" and s[-3:] == "'''")):
        s_clean = s[3:-3]
    elif len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
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
    if isinstance(d, dict): return {normalize_value(k): normalize_structure(v) for k, v in d.items()}
    if isinstance(d, list): return [normalize_structure(x) for x in d]
    return normalize_value(d)

def compare_values(proc, exp):
    proc_empty = not proc or (isinstance(proc, str) and not proc.strip())
    exp_empty = not exp or (isinstance(exp, str) and not exp.strip())

    if exp_empty and proc_empty: return "TN"
    if not exp_empty and not proc_empty: return "TP" if proc == exp else "FP"
    if not exp_empty and proc_empty: return "FN"
    return "FP"

def calculate_metrics(counter):
    TP, FP, FN = counter.get("TP", 0), counter.get("FP", 0), counter.get("FN", 0)
    precision = TP / (TP + FP) if (FP + TP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1":f1}

def evaluate(df):
    pairs = [
        ("processed_intent", "expected_intent"),
        ("processed_class", "expected_class"), 
        ("processed_attributes", "expected_attributes"),
        ("processed_filter_attributes", "expected_filter_attributes"),
        ]
    
    counts = {p[0]: Counter() for p in pairs}
    global_counter = Counter()

    for _, row in df.iterrows():
        for proc_col, exp_col in pairs:
            try:
                proc_val = normalize_structure(parse_cell(row.get(proc_col)))
                exp_val = normalize_structure(parse_cell(row.get(exp_col)))
                res = compare_values(proc_val, exp_val)
                counts[proc_col][res] += 1
                global_counter[res] += 1
            except Exception as e:
                counts[proc_col]["FP"] += 1
                global_counter["FP"] += 1
                continue

    return counts, global_counter

def clean_dataset(df):
    df_clean = df.copy()
    
    attr_cols = [c for c in ["expected_attributes", "expected_filter_attributes", 
                            "processed_attributes", "processed_filter_attributes"] 
                if c in df_clean.columns]
    
    for col in attr_cols:
        df_clean[col] = df_clean[col].apply(lambda x: x.replace('\\"', "'") if isinstance(x, str) else x)
    
    critical_cols = [c for c in ['expected_intent', 'expected_class'] if c in df_clean.columns]
    for col in critical_cols:
        df_clean = df_clean[df_clean[col].notna()]
    
    string_cols = [ c for c in ['expected_intent', 'expected_class',
                                 'processed_intent', 'processed_class'] if c in df_clean.columns]
    for col in string_cols:
        df_clean[col] = df_clean[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    
    for col in attr_cols:
        df_clean[col] = df_clean[col].apply(lambda x: parse_cell(x) if isinstance(x, str) else x)
    
    return df_clean

def save_all_metrics(all_metrics):
    detailed_rows = []
    global_rows = []

    for name, (counts, global_counter) in all_metrics.items():
        for col, counter in counts.items():
            metrics = calculate_metrics(counter)
            detailed_rows.append({
                'model': name, 'column': col, **metrics, 'TP': counter['TP'],
                'FP': counter['FP'], 'FN': counter['FN'], 'TN': counter['TN']
            })

        global_metrics = calculate_metrics(global_counter)
        global_rows.append({
            'model': name, **global_metrics, 'TP': global_counter['TP'],
            'FP': global_counter['FP'], 'FN': global_counter['FN'], 'TN': global_counter['TN']
        })

    pd.DataFrame(detailed_rows).to_csv('results/all_detailed_metrics.csv', index=False)
    pd.DataFrame(global_rows).to_csv('results/all_global_metrics.csv', index=False)

def main():
    all_metrics = {}

    for name in names:
        try:
            df_bronze = pd.read_csv(f'datasets/bronze/{name}.csv', dtype=str)
            df_silver = clean_dataset(df_bronze)
            df_silver.to_csv(f'datasets/silver/{name}_cleaned.csv', index=False)
            counts, global_counter = evaluate(df_silver)
            all_metrics[name] = (counts, global_counter)
        except Exception as e:
            print(f"Error processing {name}: {e}")
            traceback.print_exc()
    
    if all_metrics:
        save_all_metrics(all_metrics)
    else:
        print("no models proecessed successfully")

if __name__ == "__main__":
    main()

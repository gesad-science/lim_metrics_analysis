import pandas as pd
import numpy as np
from collections import Counter
import traceback

names = [
    'flan_default', 'flan_treatment_mode', 'gemma3_default', 
    'gemma3_treatment_mode', 'llama3_default', 'llama3_treatment_mode', 
    'mistral_default', 'mistral_treatment_mode', 'qwen3_default', 
    'qwen3_treatment_mode', 'phi3_default', 'phi3_treatment_mode'
]

def compare_values(proc, exp):
    empty_marks = {'none', 'null', '{}', '[]', 'nan', ''}
    proc_str, exp_str = str(proc).strip().lower(), str(exp).strip().lower()
    proc_empty = not proc_str or proc_str in empty_marks
    exp_empty = not exp_str or exp_str in empty_marks
    
    if exp_empty and proc_empty: return "TN"
    if not exp_empty and not proc_empty: return "TP" if proc_str == exp_str else "FP"
    return "FN" if not exp_empty and proc_empty else "FP"

def calculate_metrics(counter):
    TP, FP, FN, TN = counter.get("TP", 0), counter.get("FP", 0), counter.get("FN", 0), counter.get("TN", 0)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    accuracy = TP / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}

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
                res = compare_values(row.get(proc_col), row.get(exp_col))
                counts[proc_col][res] += 1
                global_counter[res] += 1
            except Exception:
                counts[proc_col]["FP"] += 1
                global_counter["FP"] += 1
    return counts, global_counter

def _aggregate_metrics(all_metrics):
    detailed_config, global_config = {}, {}
    
    for name, (counts, global_counter) in all_metrics.items():
        config = "treatment_mode" if "treatment_mode" in name else "default"
        
        for col, counter in counts.items():
            key = (config, col)
            if key not in detailed_config:
                detailed_config[key] = Counter({'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0})
            detailed_config[key].update(counter)
        
        if config not in global_config:
            global_config[config] = Counter({'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0})
        global_config[config].update(global_counter)
    
    return detailed_config, global_config

def _create_dataframe(data, is_global=False):
    rows = []
    for key, counter in data.items():
        metrics = calculate_metrics(counter)
        counts_dict = {}
        for count_type in ['TP', 'FP', 'FN', 'TN']:
            counts_dict[count_type] = counter.get(count_type, 0)
        row = {
            "configuration": key[0] if not is_global else key,
            **metrics,
            **counts_dict
        }
        if not is_global: 
            row["column"] = key[1]
        rows.append(row)
    df = pd.DataFrame(rows)

    if not is_global:
        column_order = ['configuration', 'column', 'TP', 'FP', 'FN', 'TN', 
                        'precision', 'recall', 'f1', 'accuracy']
    else:
        column_order = ['configuration', 'TP', 'FP', 'FN', 'TN', 
                        'precision', 'recall', 'f1', 'accuracy']

    existing_columns = [col for col in column_order if col in df.columns]
    return df.reindex(columns=existing_columns)

def metrics_by_config(all_metrics):
    detailed_config, global_config = _aggregate_metrics(all_metrics)
    
    detaileds = _create_dataframe(detailed_config)
    globals_df = _create_dataframe(global_config, is_global=True)
    
    detaileds.to_csv('results/all_detailed_metrics.csv', index=False)
    globals_df.to_csv('results/all_global_metrics.csv', index=False)
    
    metrics_by_model(all_metrics)
    return detaileds, globals_df

def metrics_by_model(all_metrics):
    detailed_rows, global_rows = [], []
    
    for name, (counts, global_counter) in all_metrics.items():
        config = 'treatment_mode' if 'treatment_mode' in name else 'default'
        
        for col, counter in counts.items():
            complete_counter = Counter({'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0})
            complete_counter.update(counter)

            detailed_rows.append({
                'model': name, 'configuration': config, 'column': col,
                **calculate_metrics(counter), **dict(complete_counter)
            })
        
        complete_global_counter = Counter({'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0})
        complete_global_counter.update(global_counter)

        global_rows.append({
            'model': name, 'configuration': config,
            **calculate_metrics(global_counter), **dict(complete_global_counter)
        })
    
    pd.DataFrame(detailed_rows).to_csv('results/all_detailed_metrics_by_model.csv', index=False)
    pd.DataFrame(global_rows).to_csv('results/all_global_metrics_by_model.csv', index=False)

def relative_improvement(df):
    defaults = df[df['configuration'] == 'default']
    treatments = df[df['configuration'] == 'treatment_mode']
    
    if defaults.empty or treatments.empty:
        print("Missing default or treatment_mode configurations")
        return pd.DataFrame()
    
    is_detailed = 'column' in df.columns
    
    if is_detailed:
        merged = pd.merge(defaults, treatments, on='column', suffixes=('_default', '_treatment'))
        improvements = treatments.copy()
        
        for col in ['precision', 'recall', 'f1', 'accuracy']:
            for idx, row in merged.iterrows():
                default_val = row[f'{col}_default']
                treatment_val = row[f'{col}_treatment']
                improvement_pct = (treatment_val - default_val) / default_val * 100 if default_val != 0 else np.nan
                
                mask = improvements['column'] == row['column']
                improvements.loc[mask, f'{col}_improvement_pct'] = improvement_pct
    else:
        improvements = treatments.copy()
        for col in ['precision', 'recall', 'f1', 'accuracy']:
            default_val = defaults[col].iloc[0] if len(defaults) > 0 else 0
            treatment_val = treatments[col].iloc[0] if len(treatments) > 0 else 0
            improvement_pct = (treatment_val - default_val) / default_val * 100 if default_val != 0 else np.nan
            improvements[f'{col}_improvement_pct'] = improvement_pct
    
    output_cols = ['configuration']
    if is_detailed:
        output_cols.append('column')
    
    improvement_cols = [f'{col}_improvement_pct' for col in ['precision', 'recall', 'f1', 'accuracy']]
    output_cols.extend(improvement_cols)
    
    available_cols = [col for col in output_cols if col in improvements.columns]
    result = improvements[available_cols]
    
    return result.drop('configuration', axis=1, errors='ignore')

def error_reduction(df):
    defaults = df[df['configuration'] == 'default']
    treatments = df[df['configuration'] == 'treatment_mode']
    
    if defaults.empty or treatments.empty:
        print("Missing default or treatment_mode configurations")
        return pd.DataFrame()
    
    is_detailed = 'column' in df.columns
    
    if is_detailed:
        merged = pd.merge(defaults, treatments, on='column', suffixes=('_default', '_treatment'))
        error_data = treatments.copy()
        
        for idx, row in merged.iterrows():
            error_default = row['FP_default'] + row['FN_default']
            error_treatment = row['FP_treatment'] + row['FN_treatment']
            err_pct = (error_default - error_treatment) / error_default * 100 if error_default != 0 else np.nan
            
            mask = error_data['column'] == row['column']
            error_data.loc[mask, 'error_reduction_rate_pct'] = err_pct
            error_data.loc[mask, 'error_default'] = error_default
            error_data.loc[mask, 'error_treatment'] = error_treatment
    else:
        error_data = treatments.copy()
        fp_default = defaults['FP'].iloc[0] if len(defaults) > 0 else 0
        fn_default = defaults['FN'].iloc[0] if len(defaults) > 0 else 0
        fp_treatment = treatments['FP'].iloc[0] if len(treatments) > 0 else 0
        fn_treatment = treatments['FN'].iloc[0] if len(treatments) > 0 else 0
        
        error_default = fp_default + fn_default
        error_treatment = fp_treatment + fn_treatment
        err_pct = (error_default - error_treatment) / error_default * 100 if error_default != 0 else np.nan
        
        error_data['error_reduction_rate_pct'] = err_pct
        error_data['error_default'] = error_default
        error_data['error_treatment'] = error_treatment
    
    output_cols = ['configuration']
    if is_detailed:
        output_cols.append('column')
    
    output_cols.extend(['error_reduction_rate_pct', 'error_default', 'error_treatment'])
    
    available_cols = [col for col in output_cols if col in error_data.columns]
    result = error_data[available_cols]
    
    return result.drop('configuration', axis=1, errors='ignore')

def improvement_analysis(df_details, df_global):
    detailed_improvements = relative_improvement(df_details)
    global_improvements = relative_improvement(df_global)
    
    if not detailed_improvements.empty:
        detailed_improvements.to_csv("results/detailed_improvement.csv", index=False)
    if not global_improvements.empty:
        global_improvements.to_csv("results/global_improvement.csv", index=False)

def error_analysis(df_details, df_global):
    detailed_errors = error_reduction(df_details)
    global_errors = error_reduction(df_global)
    
    if not detailed_errors.empty:
        detailed_errors.to_csv("results/detailed_errors.csv", index=False)
    if not global_errors.empty:
        global_errors.to_csv("results/global_errors.csv", index=False)

def main():
    all_metrics = {}
    
    for name in names:
        try:
            df_silver = pd.read_csv(f'datasets/silver/{name}_cleaned.csv', dtype=str)
            all_metrics[name] = evaluate(df_silver)
        except Exception as e:
            print(f"Error processing {name}: {e}")
            traceback.print_exc()
    
    if all_metrics:
        detaileds, globals_df = metrics_by_config(all_metrics)
        improvement_analysis(detaileds, globals_df)
        error_analysis(detaileds, globals_df)
    else:
        print("No models processed successfully")

if __name__ == "__main__":
    main()
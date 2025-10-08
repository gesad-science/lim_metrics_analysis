import pandas as pd
import ast
import json
import math
from collections import Counter
import os
import traceback
import numpy as np

names = [
    'flan_default', 'flan_treatment_mode', 'gemma3_default', 
    'gemma3_treatment_mode', 'llama3_default', 'llama3_treatment_mode', 
    'mistral_default', 'mistral_treatment_mode', 'qwen3_default', 
    'qwen3_treatment_mode'
    ]

def compare_values(proc, exp):
    empty_marks = ['none', 'null', '{}', '[]', 'nan']
    exp = str(exp)
    proc = str(proc)
    proc_empty = not proc or not proc.strip() or proc.lower() in empty_marks
    exp_empty = not exp or not exp.strip() or exp.lower() in empty_marks

    if exp_empty and proc_empty: return "TN"
    if not exp_empty and not proc_empty: return "TP" if proc == exp else "FP"
    if not exp_empty and proc_empty: return "FN"
    return "FP"

def calculate_metrics(counter):
    TP, FP, FN = counter.get("TP", 0), counter.get("FP", 0), counter.get("FN", 0)
    precision = TP / (TP + FP) if (FP + TP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1":f1, "accuracy": accuracy}

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
                proc_val = row.get(proc_col)
                exp_val = row.get(exp_col)
                res = compare_values(proc_val, exp_val)
                counts[proc_col][res] += 1
                global_counter[res] += 1
            except Exception as e:
                counts[proc_col]["FP"] += 1
                global_counter["FP"] += 1
                continue

    return counts, global_counter

def metrics_by_config(all_metrics):
    detailed_rows = []
    global_rows = []
    detailed_config_metrics = {}
    global_config_metrics = {}

    for name, (counts, global_counter) in all_metrics.items():
        config = "treatment_mode" if "treatment_mode" in name else "default"

        for col, counter in counts.items():
            key = (config, col)
            if key not in detailed_config_metrics:
                detailed_config_metrics[key] = Counter()

            detailed_config_metrics[key]['TP'] += counter['TP']
            detailed_config_metrics[key]['FP'] += counter['FP']
            detailed_config_metrics[key]['FN'] += counter['FN']
            detailed_config_metrics[key]['TN'] += counter['TN']

        if config not in global_config_metrics:
            global_config_metrics[config] = Counter()

        global_config_metrics[config]['TP'] += global_counter['TP']
        global_config_metrics[config]['FP'] += global_counter['FP']
        global_config_metrics[config]['FN'] += global_counter['FN']
        global_config_metrics[config]['TN'] += global_counter['TN']

    for (config, col), counter in detailed_config_metrics.items():
        metrics = calculate_metrics(counter)
        detailed_rows.append({
            "configuration": config,
            "column": col,
            "precision": metrics['precision'],
            "recall": metrics['recall'],
            "f1": metrics['f1'],
            "accuracy": metrics['accuracy'],
            "TP": counter['TP'],
            "FP": counter['FP'],
            "FN": counter['FN'],
            "TN": counter['TN'],
        })

    for config, counter in global_config_metrics.items():
        metrics = calculate_metrics(counter)
        global_rows.append({
            "configuration": config,
            "precision": metrics['precision'],
            "recall": metrics['recall'],
            "f1": metrics['f1'],
            "accuracy": metrics['accuracy'],
            "TP": counter['TP'],
            "FP": counter['FP'],
            "FN": counter['FN'],
            "TN": counter['TN'],
        })

    detaileds = pd.DataFrame(detailed_rows)
    globals = pd.DataFrame(global_rows)

    detaileds.to_csv('results/all_detailed_metrics.csv', index=False)
    globals.to_csv('results/all_global_metrics.csv', index=False)

    metrics_by_model(all_metrics) #opc se quiser salvar métricas detalhadas tb por modelo
    return detaileds, globals

def metrics_by_model(all_metrics):
    """Função auxiliar para salvar também as métricas por modelo (opcional)"""
    detailed_rows = []
    global_rows = []

    for name, (counts, global_counter) in all_metrics.items():
        # Determinar a configuração
        config = 'treatment_mode' if 'treatment_mode' in name else 'default'
        
        for col, counter in counts.items():
            metrics = calculate_metrics(counter)
            detailed_rows.append({
                'model': name,
                'configuration': config,
                'column': col, 
                **metrics, 
                'TP': counter['TP'],
                'FP': counter['FP'], 
                'FN': counter['FN'], 
                'TN': counter['TN']
            })

        global_metrics = calculate_metrics(global_counter)
        global_rows.append({
            'model': name,
            'configuration': config,
            **global_metrics, 
            'TP': global_counter['TP'],
            'FP': global_counter['FP'], 
            'FN': global_counter['FN'], 
            'TN': global_counter['TN']
        })

    pd.DataFrame(detailed_rows).to_csv('results/all_detailed_metrics_by_model.csv', index=False)
    pd.DataFrame(global_rows).to_csv('results/all_global_metrics_by_model.csv', index=False)

def relative_improvement(df):
    defaults = df[df['configuration'] == 'default']
    treatments = df[df['configuration'] == 'treatment_mode']

    if defaults.empty or treatments.empty:
        print("missing default or treatment_mode configurations")
        return pd.DataFrame()
    
    if 'column' in df.columns:
        merged = pd.merge(
            defaults,
            treatments,
            on='column',
            suffixes=('_default', '_treatment')
        )

        improvements = treatments.copy()

        for col in ['precision', 'recall', 'f1', 'accuracy']:
            if f'{col}_default' in merged.columns and f'{col}_treatment' in merged.columns:
                for idx, row in merged.iterrows():
                    default_val = row[f'{col}_default']
                    treatment_val = row[f'{col}_treatment']

                    if default_val != 0 and not pd.isna(default_val):
                        improvement_pct = (treatment_val - default_val) / default_val * 100
                    else:
                        improvement_pct = np.nan

                    mask = improvements['column'] == row['column']
                    improvements.loc[mask, f'{col}_improvement_pct'] = improvement_pct
                    improvements.loc[mask, f'{col}_default'] = default_val
                    improvements.loc[mask, f'{col}_treatment'] = treatment_val
    
    else:
        improvements = treatments.copy()
        for col in ['precision', 'recall', 'f1', 'accuracy']:
            if col in defaults.columns and col in treatments.columns:
                default_val = defaults[col].iloc[0] if len(defaults) > 0 else 0
                treatment_val = treatments[col].iloc[0] if len(treatments) > 0 else 0

                if default_val != 0 and not pd.isna(default_val):
                    improvement_pct = (treatment_val - default_val) / default_val * 100
                else:
                    improvement_pct = np.nan

                improvements[f'{col}_improvement_pct'] = improvement_pct
                improvements[f'{col}_default'] = default_val
                improvements[f'{col}_treatment'] = treatment_val
    
    if 'column' in improvements.columns:
        cols_to_show = ['configuration', 'column']
    else:
        cols_to_show = ['configuration']
    
    for col in ['precision', 'recall', 'f1', 'accuracy']:
        if f'{col}_improvement_pct' in improvements.columns:
            cols_to_show.append(f'{col}_improvement_pct')
    
    available_cols = [col for col in cols_to_show if col in improvements.columns]
    if available_cols:
        print(improvements[available_cols])
    else:
        print(improvements)
    
    final = improvements[available_cols]
    return final.drop('configuration', axis=1)


def improvement_analysis(df_details, df_global):
    detailed_improvements = relative_improvement(df_details)
    global_improvements = relative_improvement(df_global)
    
    if not detailed_improvements.empty:
        detailed_improvements.to_csv("results/detailed_improvement.csv", index=False)
    
    if not global_improvements.empty:
        global_improvements.to_csv("results/global_improvement.csv", index=False)

def main():
    all_metrics = {}

    for name in names:
        try:
            df_silver = pd.read_csv(f'datasets/silver/{name}_cleaned.csv', dtype=str)
            counts, global_counter = evaluate(df_silver)
            all_metrics[name] = (counts, global_counter)
        except Exception as e:
            print(f"Error processing {name}: {e}")
            traceback.print_exc()
    
    if all_metrics:
        detaileds, globals = metrics_by_config(all_metrics)
        improvement_analysis(detaileds, globals)
    else:
        print("no models proecessed successfully")

if __name__ == "__main__":
    main()

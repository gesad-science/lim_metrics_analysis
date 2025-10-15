import pandas as pd

models = ['flan', 'gemma3', 'llama3', 'mistral', 'phi3', 'qwen3']
configurations = ['default', 'treatment_mode']

def performance_parity(df):
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    results = pd.DataFrame(columns=['configuration', 'metric', 'performance_parity'])
    defaults = df[df['configuration'] == 'default']
    treatments = df[df['configuration'] != 'default']
    for metric in metrics:
        min_val = defaults[metric].min()
        max_val = defaults[metric].max()
        pp = 1 - (max_val - min_val) / max_val if max_val != 0 else 0.0
        results = pd.concat([results, pd.DataFrame({'configuration': "default", 'metric': metric, 'performance_parity': pp}, index=[0])], ignore_index=True)
    for metric in metrics:
        min_val = treatments[metric].min()
        max_val = treatments[metric].max()
        pp = 1 - (max_val - min_val) / max_val if max_val != 0 else 0.0
        results = pd.concat([results, pd.DataFrame({'configuration': "treatment_mode", 'metric': metric, 'performance_parity': pp}, index=[0])], ignore_index=True)
    return results

def prediction_stability_index(df):
    pass 

def main():
    # Execução do performance_parity
    df = pd.read_csv("results/all_global_metrics_by_model.csv")
    pp_result = performance_parity(df)
    pp_result.to_csv("results/performance_parity.csv", index=False)

    

if __name__ == "__main__":
    main()
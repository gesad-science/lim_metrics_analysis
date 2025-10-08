import pandas as pd

# Função principal
def main():
    df_metrics = pd.read_csv('results/all_detailed_metrics.csv')
    df_global = pd.read_csv('results/all_global_metrics.csv')

    versions_default = ["flan_default", "gemma3_default", "llama3_default", "mistral_default", "qwen3_default"]
    versions_treated = ["flan_treatment_mode", "gemma3_treatment_mode", "llama3_treatment_mode", "mistral_treatment_mode", "qwen3_treatment_mode"]
    columns = ["processed_intent", "processed_class", "processed_attributes", "processed_filter_attributes"]

    for column in columns:
        linhas = []
        for version in versions_default:
            linha = []

# Execução principal
if __name__ == "__main__":
    main()

from scipy import stats
import pandas as pd
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='calculate the correlation coefficient between human evaluation and different BLEU variants')
    parser.add_argument("--format", action='store_true', help='if you want to a formatted table', required = False)
    args = parser.parse_args()

    comparision_metrics = pd.read_csv("BLEUs_results.csv")
    human_result = pd.read_csv("human_annotation.csv")
    human_result["AVG"] = (human_result["Author1"]+human_result["Author2"]+human_result["Author3"])/3*25
    
    results = pd.DataFrame(columns=['B-Moses', 'B-Norm', 'B-CC'], index=["Pearson","Spearman","Kendall"])
    for bleu_var in ['B-Moses', 'B-Norm', 'B-CC']:
        results[bleu_var]["Pearson"] = stats.pearsonr(human_result["AVG"],comparision_metrics[bleu_var])[0]
    for bleu_var in ['B-Moses', 'B-Norm', 'B-CC']:
        results[bleu_var]["Spearman"] = stats.spearmanr(human_result["AVG"],comparision_metrics[bleu_var])[0]
    for bleu_var in ['B-Moses', 'B-Norm', 'B-CC']:
        results[bleu_var]["Kendall"] = stats.kendalltau(human_result["AVG"],comparision_metrics[bleu_var])[0]
    
    if args.format:
        pd.options.display.float_format = '{:.4f}'.format
        print(results)
    else:
        for bleu_var in ['B-Moses', 'B-Norm', 'B-CC']:
            print(bleu_var)
            print("\tPearsonResult(correlation={}, pvalue={})".format(results[bleu_var]["Pearson"],stats.pearsonr(human_result["AVG"],comparision_metrics[bleu_var])[1]))
            print("\t{}".format(stats.spearmanr(human_result["AVG"],comparision_metrics[bleu_var])))
            print("\t{}".format(stats.kendalltau(human_result["AVG"],comparision_metrics[bleu_var])))

        
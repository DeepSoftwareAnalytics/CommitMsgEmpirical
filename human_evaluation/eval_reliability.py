from scipy import stats
import pandas as pd
import krippendorff

if __name__ == '__main__':
    human_label = pd.read_csv('human_annotation.csv')
    human_label = human_label[["Author1","Author2","Author3"]].T.values.tolist()
    print("Krippendorff's alpha for ordinal metric: {}".format(krippendorff.alpha(reliability_data=human_label,level_of_measurement='ordinal')))
    print("Kendall’s tau (Author#0 and Author#1): {}".format(stats.kendalltau(human_label[0],human_label[1])))
    print("Kendall’s tau (Author#0 and Author#2): {}".format(stats.kendalltau(human_label[0],human_label[2])))
    print("Kendall’s tau (Author#1 and Author#2): {}".format(stats.kendalltau(human_label[1],human_label[2])))
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
    # Content adequacy #1,Conciseness #1,Expressiveness #1
    human_label = pd.read_csv('human_annotation_emse_enhanced.csv')
    human_label = human_label[["Content adequacy #1","Content adequacy #2","Content adequacy #3"]].T.values.tolist()
    print("Krippendorff's alpha for ordinal metric: {}".format(krippendorff.alpha(reliability_data=human_label,level_of_measurement='ordinal')))
    print("Kendall’s tau (Author#0 and Author#1): {}".format(stats.kendalltau(human_label[0],human_label[1])))
    print("Kendall’s tau (Author#0 and Author#2): {}".format(stats.kendalltau(human_label[0],human_label[2])))
    print("Kendall’s tau (Author#1 and Author#2): {}".format(stats.kendalltau(human_label[1],human_label[2])))
    
    human_label = pd.read_csv('human_annotation_emse_enhanced.csv')
    human_label = human_label[["Conciseness #1","Conciseness #2","Conciseness #3"]].T.values.tolist()
    print("Krippendorff's alpha for ordinal metric: {}".format(krippendorff.alpha(reliability_data=human_label,level_of_measurement='ordinal')))
    print("Kendall’s tau (Author#0 and Author#1): {}".format(stats.kendalltau(human_label[0],human_label[1])))
    print("Kendall’s tau (Author#0 and Author#2): {}".format(stats.kendalltau(human_label[0],human_label[2])))
    print("Kendall’s tau (Author#1 and Author#2): {}".format(stats.kendalltau(human_label[1],human_label[2])))
    
    human_label = pd.read_csv('human_annotation_emse_enhanced.csv')
    human_label = human_label[["Expressiveness #1","Expressiveness #2","Expressiveness #3"]].T.values.tolist()
    print("Krippendorff's alpha for ordinal metric: {}".format(krippendorff.alpha(reliability_data=human_label,level_of_measurement='ordinal')))
    print("Kendall’s tau (Author#0 and Author#1): {}".format(stats.kendalltau(human_label[0],human_label[1])))
    print("Kendall’s tau (Author#0 and Author#2): {}".format(stats.kendalltau(human_label[0],human_label[2])))
    print("Kendall’s tau (Author#1 and Author#2): {}".format(stats.kendalltau(human_label[1],human_label[2])))
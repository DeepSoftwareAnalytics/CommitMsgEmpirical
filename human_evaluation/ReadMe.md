# RQ1 Human Evaluation Results



## Get Krippendorff's alpha and Kendall’s tau

```sh
python eval_reliability.py
```

Krippendorff's alpha for ordinal metric: 0.8551221815781381 

Kendall’s tau (Author#0 and Author#1): KendalltauResult(correlation=0.8251680898238802 **> 0.8** , pvalue=5.804877252677888e-23)

Kendall’s tau (Author#0 and Author#2): KendalltauResult(correlation=0.9007287753910033 **> 0.8** , pvalue=1.994301058255853e-27)

Kendall’s tau (Author#1 and Author#2): KendalltauResult(correlation=0.8466989681122703 **> 0.8** , pvalue=3.1986477435326564e-24)


## Get Correlation

```sh
python eval_correlation.py --format
```

|       | B-Moses | B-Norm | B-CC  |
|----------|---------|--------|--------|
| Pearson  | 0.2444  | 0.6966 | 0.5715 |
| Spearman | 0.1970  | 0.6229 | 0.5461 |
| Kendall  | 0.1698  | 0.4677 | 0.4037 |


```sh
python eval_correlation.py
```

B-Moses

        PearsonResult(correlation=0.24435772972227326, pvalue=0.014280876238425467)
        
        SpearmanrResult(correlation=0.19697704093225188, pvalue=0.049495829147941255)
        
        KendalltauResult(correlation=0.1697675971999534, pvalue=0.04930553082987851)
        
B-Norm

        PearsonResult(correlation=0.6965603253000296, pvalue=8.476334482364091e-16)
        
        SpearmanrResult(correlation=0.6228542206450789, pvalue=4.538124117661111e-12)
        
        KendalltauResult(correlation=0.46767293985242286, pvalue=7.230184262098552e-11)
        
B-CC

        PearsonResult(correlation=0.5715389625437758, pvalue=5.287140230888454e-10)
        
        SpearmanrResult(correlation=0.546091524849012, pvalue=4.190203785059649e-09)
        
        KendalltauResult(correlation=0.40371805391990023, pvalue=1.822821167326917e-08)
        

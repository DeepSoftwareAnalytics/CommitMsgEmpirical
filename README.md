## Environment

This is the replication package for "On the Evaluation of Commit Message Generation Models: An Experimental Study".

```sh
conda create -n MCMD python=3.8 numpy=1.19.2
conda activate MCMD
pip install nltk==3.6.2 scipy==1.5.2 pandas==1.1.3 krippendorff==0.4.0 scikit-learn==0.24.1 sumeval==0.2.2
```

## Experimental Models

[CommitGen][CommitGen], [NMT][NMT], [CoDiSum][CoDiSum], [PtrGNCMsg][PtrGNCMsg], [NNGen][NNGen].


## Experimental Datasets

- Existing Datasets: [CommitGen<sub>data</sub>][CommitGen_data], [NNGen<sub>data</sub>][NNGen_data], [CoDiSum<sub>data</sub>][CoDiSum_data]

- Our Dataset: [MCMD][MCMD]

More info about our dataset can be found [here](dataset/ReadMe.md).

## Evaluation Metrics

- [B-Moses](metrics/B-Moses.perl)
- [B-Norm](metrics/B-Norm.py)
- [B-CC](metrics/B-CC.py)
- [Rouge](metrics/Rouge.py)
- [Meteor](metrics/Meteor.py)

Usage demo about the metrics can be found [here](metrics/ReadMe.md).


## Research Questions

### (RQ1) How do different BLEU variants affect the evaluation of commit message generation?


See RQ1 results [here](human_evaluation/ReadMe.md).


### (RQ2) How good are the existing models and datasets?

RQ2 results: [RQ2.ipynb](research_questions/RQ2.ipynb)


### (RQ3) Why do we need a new dataset MCMD for evaluating commit message generation?

RQ3 results: [RQ3.ipynb](research_questions/RQ3.ipynb)


### (RQ4) What is the impact of different dataset splitting strategies?

RQ4 results: [RQ4.ipynb](research_questions/RQ4.ipynb)


## Possible Ways to Improve

Evaluation results of our improvements to NNGen can be found in [nngen_improvement.ipynb](discussion/nngen_improvement.ipynb)


[CommitGen]: https://sjiang1.github.io/commitgen/ "S. Jiang, A. Armaly, and C. McMillan, “Automatically generating commit messages from diffs using neural machine translation,” in ASE, 2017."

[NMT]: https://github.com/epochx/commitgen "P. Loyola, E. Marrese-Taylor, and Y. Matsuo, “A neural architecture for generating natural language descriptions from source code changes,” in ACL. Association for Computational Linguistics, 2017, pp. 287–292."

[CoDiSum]: https://github.com/SoftWiser-group/CoDiSum "S. Xu, Y. Yao, F. Xu, T. Gu, H. Tong, and J. Lu, “Commit message generation for source code changes,” in IJCAI. ijcai.org, 2019, pp. 3975–3981."

[PtrGNCMsg]: https://zenodo.org/record/2542706#.XECK8C277BJ "Q. Liu, Z. Liu, H. Zhu, H. Fan, B. Du, and Y. Qian, “Generating commit messages from diffs using pointer-generator network,” in MSR. IEEE / ACM, 2019, pp. 299–309."

[NNGen]: https://github.com/Tbabm/nngen "Z. Liu, X. Xia, A. E. Hassan, D. Lo, Z. Xing, and X. Wang, “Neuralmachine-translation-based commit message generation: how far are we?” in ASE. ACM, 2018, pp. 373–384."

[CommitGen_data]: https://dl.boxcloud.com/zip_download/zip_download?ProgressReportingKey=0199BB09C96974426BFE1A61F051C6D3&d=43152170708&ZipFileName=NMT_DataSet.zip&Timestamp=1620278305&SharedLink=https%3A%2F%2Fnotredame.box.com%2Fs%2Fwghwpw46x41nu6iulm6qi8j42finuxni&HMAC2=4ae81054b14a58386038b28bd5d03083f3db5f124f03cf44e73f3c02ae7a3496 "CommitGen_data download link"

[NNGen_data]: https://github.com/Tbabm/nngen/tree/master/data "NNGen_data download link"

[CoDiSum_data]: https://github.com/SoftWiser-group/CoDiSum/blob/master/data4CopynetV3.zip "CoDiSum_data download link"

[MCMD]: https://doi.org/10.5281/zenodo.5025758 "Multi-programming-language Commit Message Dataset"

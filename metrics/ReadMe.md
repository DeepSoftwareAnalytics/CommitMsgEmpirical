# Metrics

There are 3 BLEU variants B-Moses, B-Norm, B-CC commonly used in commit message generation.

`example.ref.msg` is an example reference file,
`example.gen.msg` is an example generation file.

Commands below are supposed to used under the folder `metrics`

## Environment

### Conda

```sh
conda create -n MCMD python=3.8 numpy=1.19.2 -y
conda activate MCMD
pip install nltk==3.6.2 scipy==1.5.2 pandas==1.1.3 krippendorff==0.4.0 scikit-learn==0.24.1 sumeval==0.2.2 sacrebleu==1.5.1
```
### Docker

```sh
docker pull itaowei/commit_msg_empirical
```

## B-Moses


```sh
./B-Moses.perl example.ref.msg < example.gen.msg
```

```sh
BLEU = 16.41, 27.6/16.8/13.4/11.7 (BP=1.000, ratio=1.004, hyp_len=17546, ref_len=17469)
It is in-advisable to publish scores from multi-bleu.perl.  The scores depend on your tokenizer, which is unlikely to be reproducible from your paper or consistent across research groups.  Instead you should detokenize then use mteval-v14.pl, which has a standard tokenization.  Scores from multi-bleu.perl can still be used for internal purposes when you have a consistent tokenizer.

```

## B-Norm


```sh
python B-Norm.py example.ref.msg < example.gen.msg
```

```sh
23.07485193493874
```


## B-CC


```sh
python B-CC.py --ref_path example.ref.msg --gen_path example.gen.msg
```

```sh
16.76651267396287
```

## Rouge

```sh
python Rouge.py --ref_path example.ref.msg --gen_path example.gen.msg
```

```sh
{'ROUGE-1': 27.431629823503034, 'ROUGE-2': 15.733902586866064, 'ROUGE-L': 26.996306706020356}
```


## Meteor


```sh
python Meteor.py --ref_path example.ref.msg --gen_path example.gen.msg
```

```sh
26.558283553998972
```

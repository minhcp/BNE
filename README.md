# Pretrained Biomedical Name Encoder

This repository contains the pre-trained biomedical name encoders proposed in the paper "Robust Representation Learning of Biomedical Names", ACL 2019.

# Dependencies:
  * Python 3
  * [Spacy](https://spacy.io/usage) and the ['en_core_web_sm' model](https://spacy.io/usage/models)
  * [TensorFlow](https://www.tensorflow.org/install)

The encoder is tested with the following dependency settings:
  * Python 3.6.5
  * tensorflow==1.12.0
  * tensorflow-gpu==1.12.0
  * numpy==1.16.1
  * spacy==2.0.12
  * tqdm==4.31.1

# Two pretrained encoders:
models/BNE_SGw and models/Emb_SGsc

# Usage
1. Download and extract two pretrained word embedding files: [Emb_SGw.txt and Emb_SGsc.txt](https://bit.ly/2LnM5E7). These embeddings are pretrained on a PubMed abstracts using skip-gram models.
2. Create a file names.txt that stores the input names, one name on each line.
3. Run either of these commands. You will need to update the paths accordingly.
```
python bne.py --model models/BNE_SGw --fe e:/Emb_SGw.txt --fi names.txt --fo output_BNE_SGw.txt
python bne.py --model models/BNE_SGsc --fe e:/Emb_SGsc.txt --fi names.txt --fo output_BNE_SGsc.txt
```

# Pre-calculated UMLS name embeddings
We calculate embeddings of 2.2 million concept names collected from UMLS. Specifically, these names are specified by the 'STR' column in MRCONSO.RRF file, UMLS 2018AA-full dump. We consider the names in 4 popular biomedical vocabularies OMIM, SNOMEDCT_US, MSH, and ICD10.

Download these name embeddings using this link [UMLS_output_BNE_SGw.txt](https://bit.ly/2Gg0Qo9)







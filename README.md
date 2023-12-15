# CrossFeat
CrossFeat: A transformer-based cross-feature learning model for predicting drug side effects frequency
DOI : 

# Requirements
- Python == 3.6.12
- PyTorch == 1.2
- Numpy == 1.19.2
- Scikit-learn == 0.24.0

# Files
- data
   - sample_5fold_idx.pkl  This file includes drug and side effect indices in the PS1 and PS2 datasets for training, validation, and test set per fold. Additionally, it incorporates frequency values for drug-side effect pairs.
   - PS3.pkl  This file contains drug and side effect indices of PS3 dataset, along with corresponding frequency values for drug-side effect pairs.

   The four input files below were preprocessed following the method by Zhao et al. (DOI: https://doi.org/10.1093/bib/bbab449)
   - drug_word.pkl  The drug Mol2vec word vector matrix.
   - drug_fingerprint_similarity.pkl  The Jaccard score similarity matrix of drug fingerprint.
   - side_sem.pkl  The side effect semantic similarity matrix.
   - side_word.pkl  The side effect GloVe word vector matrix.
     
- main
   - main.py  This Python file allows for training and testing the CrossFeat using a five-fold cross-validation setup.
   - network_main.py  This file contains each network comprising CrossFeat.
   - utils.py  This file contains the utilities for CrossFeat.
 
# Contact
If you have any questions or suggestions regarding CrossFeat, please feel free to contact Bin Baek at baekbini@gm.gist.ac.kr.


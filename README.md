# CrossFeat
CrossFeat: A transformer-based cross-feature learning model for predicting drug side effects frequency
![Figure1](/main/Figure1.JPG)
DOI : https://doi.org/10.1186/s12859-024-05915-2

# Requirements
- Python == 3.6.12
- PyTorch == 1.2
- Numpy == 1.19.2
- Scikit-learn == 0.24.0

# Files
- data
   - sample_5fold_idx.pkl:  This file includes drug and side effect indices in the PS1 and PS2 datasets for training, validation, and test set per fold. Additionally, it incorporates frequency values for drug-side effect pairs.
   - PS3.pkl:  This file contains drug and side effect indices of PS3 dataset, along with corresponding frequency values for drug-side effect pairs.
   - drug_id.csv: .csv file with drug names
   - side_id.csv: .csv file with side effect names

   The four input files below were preprocessed following the method by Zhao et al. (DOI: https://doi.org/10.1093/bib/bbab449)
   - drug_word.pkl:  The drug Mol2vec word vector matrix.
   - drug_fingerprint_similarity.pkl:  The Jaccard score similarity matrix of drug fingerprint.
   - side_sem.pkl:  The side effect semantic similarity matrix.
   - side_word.pkl:  The side effect GloVe word vector matrix.
     
- main
   - main.py:  This Python file allows for training and testing the CrossFeat using a five-fold cross-validation setup.
   - network_main.py:  This file contains each network comprising CrossFeat.
   - utils.py:  This file contains the utilities for CrossFeat.
 
- FAERS dataset:  Collectively, these files provide diverse input data for training and evaluating models using the FAERS dataset from the 4th quarter of 2012 to the 2nd quarter of 2023.
   - SEname.tsv:  Contains the names of side effects.
   - age_sample_5fold_idx_VL.zip:  File with sample indices divided into training, validation, and test folds for 5-fold cross-validation, specifically related to patient age.
   - drug_fingerprint1024_jaccard.pkl:  Holds Jaccard similarity values for drug fingerprints.
   - drug_mol2vec300d.csv:  Includes 300-dimensional mol2vec vectors for drugs.
   - drugname.csv:  Contains the names of drugs.
   - gender_age_sample_5fold_idx_VL.zip:  Similar to the age file, it also includes sex information for 5-fold cross-validation.
   - gender_sample_5fold_idx_VL.zip:  Sample index file considering sex for 5-fold cross-validation.
   - se_DAG_sim.csv:  File representing the similarity between side effects using a Directed Acyclic Graph (DAG) approach.
   - se_glove_wordvector.csv:  Holds word vectors for side effects obtained using the GloVe algorithm.
   
 
# Contact
If you have any questions or suggestions regarding CrossFeat, please feel free to contact Bin Baek at baekbini@gm.gist.ac.kr.


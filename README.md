### **<u>TEMC-Cas: Accurate Cas Protein Classification via Combined Contrastive Learning and Protein Language Models</u>**

This project focuses on the development and application of Cas protein classification models, providing standardized datasets of Cas proteins and reproducible model training code. It supports classification tasks for general Cas proteins and specific subtypes (e.g., Cas12), aiming to facilitate research in CRISPR-Cas system bioinformatics.

##### Directory Structure

|    File    | Description                                                  |
| :--------: | :----------------------------------------------------------- |
| ./cas_all/ | Contains all collected Cas protein sequences (FASTA format), serving as the foundational dataset for model training and multi-type Cas protein analysis. |
|  ./cas12/  | Contains curated Cas12 subtype sequences (FASTA format), specialized for Cas12-specific classification tasks and performance validation. |
| ./model.py | Core script implementing model definition, training, and inference logic:- Defines neural network architectures for Cas protein sequence classification;- Includes data loading, training loop, loss calculation, and evaluation modules;- Supports dataset switching between cas_all and cas12. |

##### Model Training

1）Environment Preparation

```xml
Python 3.8+
torch==2.0.0 
biopython==1.81 
scikit-learn==1.3.0 
pandas==2.0.3
```

2）Data Preparation

./cas_all/ and ./cas12/ contain valid FASTA files.

Each sequence in FASTA files must follow the standard format (header line starting with >, followed by amino acid sequences):

```
>CasProtein_ID_Species
MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGNFGADAQGAMNKALELFRKDIAAKYKELGYQG
```

3）Run Training

```python
python ./model.py
```

Evaluation Metrics

* Precision: Measures the proportion of true positives among all predicted positives (avoids false positives).
* Recall: Measures the proportion of true positives among all actual positives (avoids false negatives).
* F1-Score: Harmonic mean of Precision and Recall (balances the two metrics).
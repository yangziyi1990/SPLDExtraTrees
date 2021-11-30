# SPLDExtraTrees: Robust machine learning approach for predicting kinase inhibitor resistance

This is the repository for the SPLDExtraTrees project.

communication E-mail: yangziyi091100@163.com

Please 🌟star🌟 the repo if you like our work, thank you.

## Abstract
Drug resistance is a major threat to the global health and a significant concern throughout the clinical treatment of diseases and drug development. The mutation in proteins that is related to drug binding is a common cause for adaptive drug resistance. Therefore, quantitative estimations of how mutations would affect the interaction between a drug and the target protein would be of vital significance for the drug development and the clinical practice. Computational methods that rely on molecular dynamics simulations, Rosetta protocols, as well as machine learning methods have been proven to be capable of predicting ligand affinity changes upon protein mutation. However, the severely limited sample size and heavy noise induced overfitting and generalization issues have impeded wide adoption of machine learning for studying drug resistance. In this paper, we propose a robust machine learning method, termed SPLDExtraTrees, which can accurately predict ligand binding affinity changes upon protein mutation and identify resistance-causing mutations. Especially, the proposed method ranks training data following a specific scheme that starts with easy-to-learn samples and gradually incorporates harder and diverse samples into the training, and then iterates between sample weight recalculations and model updates. In addition, we calculate additional physics-based structural features to provide the machine learning model with the valuable domain knowledge on proteins for this data-limited predictive tasks. The experiments substantiate the capability of the proposed method for predicting kinase inhibitor resistance under three scenarios, and achieves predictive accuracy comparable to that of molecular dynamics and Rosetta methods with much less computational costs.

If you find this code useful in your research then please cite:
```bash
@article{
  title={SPLDExtraTrees: Robust machine learning approach for predicting kinase inhibitor resistance},
  author={Zi-Yi Yang, Zhao-Feng Ye, Yi-Jia Xiao, and Chang-Yu Hsieh},
  year={2021},
}
```

![](https://github.com/yangziyi1990/SPLDExtraTrees/blob/main/Figure1_pipeline.png)

## Installation

Follow the steps in the defined order to avoid conflicts.

1. The models.zip file will have to be downloaded and unzipping.

```bash
git clone https://github.com/yangziyi1990/SPLDExtraTrees.git
```

2. Create an environment:

```bash
conda env create -f requirements_env.yml

conda activate SPLDExtraTrees
```



## Usage

In this repo, we compare the proposed method (i.e., SPLDExtraTrees) with other two machine learning methods, ExtraTrees and SPLExtraTrees in three scenarios. For the first scenario, we trained the machine learning methods on the Platinum dataset and tested them on the TKI dataset to evaluate the model's extrapolating capability. In the second scenario, a small part of the TKI dataset along with the Platinum dataset was used to train the models, and the rest of the TKI dataset was used for testing. For the third scenario, the machine learning methods were trained and tested on the TKI dataset such that we could evaluate the interpolative capability of the model.

We provide some notebooks called `S*.ipynb` which contains the analysis performed in the manuscript for anyone intersted and who wants to reproduce our results. The analysis was made in python. Input data for the machine learning methods is provided in the file `Data`.

`Results_State.ipynb` can plot scatter plots of the experimental versus calculated $\Delta\Delta$G values.














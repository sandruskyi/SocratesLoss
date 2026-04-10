# Socrates Loss: Unifying Confidence Calibration and Classification by Leveraging the Unknown
Deep neural networks, despite their high accuracy, often exhibit poor confidence calibration, limiting their reliability in high-stakes applications. Current ad-hoc confidence calibration methods attempt to fix this during training but face a fundamental trade-off: two-phase training methods achieve strong classification performance at the cost of training instability and poorer confidence calibration, while single-loss methods are stable but underperform in classification. This paper addresses and mitigates this stability-performance trade-off. We propose Socrates Loss, a novel, unified loss function that explicitly leverages uncertainty by incorporating an auxiliary unknown class, whose predictions directly influence the loss function and a dynamic uncertainty penalty. This unified objective allows the model to be optimized for both classification and confidence calibration simultaneously, without the instability of complex, scheduled losses. We provide theoretical guarantees that our method regularizes the model to prevent miscalibration and overfitting. Across four benchmark datasets and multiple architectures, our comprehensive experiments demonstrate that Socrates Loss consistently improves training stability while achieving more favorable accuracy-calibration trade-off, often converging faster than existing methods. 

This repository contains the PyTorch implementation of
- Gómez-Gálvez, S., Olenyi, T., Dobbie, G., & Taskova, K. (2026). Socrates Loss: Unifying Confidence Calibration and Classification by Leveraging the Unknown. Transactions on Machine Learning Research. https://openreview.net/forum?id=DONqw1KhHq

## Requirements

- Python >= 3.6
- PyTorch >= 1.0
- CUDA
- Numpy
- argparse
- os
- psutil
- pandas
- sklearn
- scipy
- matplotlib
- plotly
- PIL
- datetime
- time
- tqdm
- seaborn
- h5py
- opencv-python
- Pillow
- tensorboard
- kaleido
- transformers



## Usage
### Training and evaluating Calibrated Classifiers based on Socrates Loss
The `main.py` contains training and evaluation functions in standard training setting. 

### Integrating Socrates Loss in your training: 
If you want to use Socrates Loss in your own training, simply add an extra class to your model (classes = num_classes + 1) and copy and follow the loss function code found in ./losses/Socrates.py

To replicate Socrates values:

```
--old 0 --version 1 --pretrain 0 --dynamic --version_SAT_original --version_FOCALinGT --version_FOCALinSAT --version_changingWithIdk

Rest of hyperparameter tuning check Appendix F - Model Reproducibility: --arch --sat-momentum --gamma-focal-loss --alpha-focal-loss
```
examples in ./sh_examples folder.



#### Runnable scripts
- Training and evaluation using the default parameters
  
  We provide training scripts in the sh_examples directory with format `.sh`. Example:
  ```bash
  $ bash run_cifar10_vgg_Socrates.sh
  ```


## Reference and citation
For technical details, please check:

@article{gomezgalvez2026socrates,
  title={Socrates Loss: Unifying Confidence Calibration and Classification by Leveraging the Unknown},
  author={G{\'o}mez-G{\'a}lvez, Sandra and Olenyi, Tobias and Dobbie, Gillian and Ta{\v{s}}kova, Katerina},
  journal={Transactions on Machine Learning Research},
  year={2026},
  url={https://openreview.net/forum?id=DONqw1KhHq}
}

## Acknowledgement
This code is based on:
- [Self Adaptative Training](https://github.com/LayneH/SAT-selective-cls)
- [Deep Gambler](https://github.com/Z-T-WANG/NIPS2019DeepGamblers)
- [pytorch-classification](https://github.com/bearpaw/pytorch-classification)
- [Towards better selective classification](https://github.com/BorealisAI/towards-better-sel-cls)

We thank the authors for sharing their code.

## Contact
If you have any question about this code, feel free to open an issue or contact: Sandra Gómez-Gálvez sgom490@aucklanduni.ac.nz  or Katerina Taškova  katerina.taskova@auckland.ac.nz 

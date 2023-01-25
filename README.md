# Using gaze for behavioural biometrics

Identification system based on eye movements modeled as an Ornstein-Uhlenbeck process.

## Installation

Install required libraries with Anaconda:

```bash
conda create --name gazeID -c conda-forge --file requirements.txt
conda activate gazeID
```
Install [NSLR-HMM](https://gitlab.com/nslr/nslr-hmm)

```bash
python -m pip install git+https://gitlab.com/nslr/nslr
```

### Features extraction
Extract Ornstein-Uhlenbeck features from [FIFA-DB dataset](https://www.morancerf.com/publications) (`datasets/FIFA`) launching the module `extract_OU_params.py`, results will be saved in `features/FIFA_OU_posterior_VI`.


### Train and test
Module `event_kfold.py` exploits SVMs for classification on the features extracted as an Ornstein-Uhlenbeck process via a Nested cross-validation procedure.


## Reference

If you use this code, please cite the paper:

```
@article{damelio2023using,
AUTHOR = {D’Amelio, Alessandro and Patania, Sabrina and Bursic, Sathya and Cuculo, Vittorio and Boccignone, Giuseppe},
TITLE = {Using Gaze for Behavioural Biometrics},
JOURNAL = {Sensors},
VOLUME = {23},
YEAR = {2023},
NUMBER = {3},
ARTICLE-NUMBER = {1262},
URL = {https://www.mdpi.com/1424-8220/23/3/1262},
ISSN = {1424-8220},
DOI = {10.3390/s23031262}
}
```

```
@article{d2021gazing,
  title={Gazing at Social Interactions Between Foraging and Decision Theory},
  author={D'Amelio, Alessandro and Boccignone, Giuseppe},
  journal={Frontiers in neurorobotics},
  volume={15},
  pages={31},
  year={2021},
  publisher={Frontiers}
}
```

```
@article{boccignone2020gaze,
  title={On gaze deployment to audio-visual cues of social interactions},
  author={Boccignone, Giuseppe and Cuculo, Vittorio and D’Amelio, Alessandro and Grossi, Giuliano and Lanzarotti, Raffaella},
  journal={IEEE Access},
  volume={8},
  pages={161630--161654},
  year={2020},
  publisher={IEEE}
}
```

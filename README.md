# spec17-ml

Comparative Analysis of Machine Learning Models for Performance Prediction of  SPEC CPU2017 Benchmarks

## Installation

- ### Install Python3.8 venv

  `sudo apt-get install python3.8-venv`

- ### Create a clean virtual environment and activate

  `python -m venv ./venv`

  `source venv/bin/activate`

  for some packages you may need to install to install the python dev:

  `sudo apt-get install python3.8-dev`

- ### Install project dependencies

  `pip install -r requirements.txt`

## Structure

| Resource | Description |
| --- | --- |
| `spec/data` | contains the input data and a notebook to download the data from [SPEC CPU2017 published results](https://www.spec.org/cpu2017/results/cpu2017.html) |
| `spec/predict` | contains Python scripts to store parameters, prepare data, select features, create models, and visualise results |
| `spec/regress_01_explore.ipynb` | Jupyter notebook to load, clean and transform data |
| `spec/regress_02_select.ipynb` | Jupyter notebook to select features |
| `spec/regress_03_evaluate.ipynb` | Jupyter notebook to select models and evaluate them |
| `spec/regress_related_work.ipynb` | Jupyter notebook to compare the results with related work |


## Publications

- A. Tousi and M. Luján, "Comparative Analysis of Machine Learning Models for Performance Prediction of the SPEC Benchmarks," in _IEEE Access_, vol. 10, pp. 11994-12011, 2022, doi: 10.1109/ACCESS.2022.3142240 [[IEEE Open Access 2022](https://ieeexplore.ieee.org/document/9676614)]. If you use `spec17-ml` for your research, please cite this paper.

This work was supported by EPSRC grants EP/T026995/1 EnnCore and EP/N035127/1 Lambda.

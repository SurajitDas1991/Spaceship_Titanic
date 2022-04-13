# Spaceship-Titanic

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/surajitdas1991/spaceship_titanic/main/src/dashboard.py)

This repository consists of files required to deploy a Machine Learning Web App created with streamlit.

This is an app to predict if passengers have been transported based on the available data. Data has been taken from a competition https://www.kaggle.com/competitions/spaceship-titanic/data

The entire code is developed in modular fashion. Idea has been to focus on more on trying out feature engineering and utilizing different machine learning algorithms.

![alt text](https://github.com/SurajitDas1991/Spaceship_Titanic/blob/main/assets/01.jpg?raw=true)
![alt text](https://github.com/SurajitDas1991/Spaceship_Titanic/blob/main/assets/02.jpg?raw=true)
![alt text](https://github.com/SurajitDas1991/Spaceship_Titanic/blob/main/assets/03.jpg?raw=true)


## STEPS -

### STEP 01- Clone the repository

### STEP 02- Create a conda environment after opening the repository in VSCODE

```bash
conda create --prefix ./env python=3.8 -y
```

```bash
conda activate ./env
```
OR
```bash
source activate ./env
```

### STEP 03- install the requirements
```bash
pip install -r requirements.txt
```
### STEP 05- commit and push the changes to the remote repository

### STEP 06- streamlit run .\src\dashboard.py

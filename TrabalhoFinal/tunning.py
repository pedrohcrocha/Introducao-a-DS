from os import chdir
import pandas as pd
import numpy as np
from datetime import datetime

chdir('/home/pedro/Área de Trabalho/PPGE/Introdução a Data Science/TrabalhoFinal')

# =============================================================================
# Data Prep
# =============================================================================
df = pd.read_csv('nascidos_vivos_2017-2014.csv', sep=',')
df['PARTO'] = np.where(df['PARTO'] == 2, 1, 0)

df = df[['ANO','CODESTAB','IDADEMAE', 'QTDPARTNOR', 'QTDPARTCES',
        'MESPRENAT', 'CONSPRENAT','SEMAGESTAC', 'PARTO']]

df = df.dropna()

# Filtrando para apenas um estabelecimento
df['CODESTAB'].value_counts()
"""
2399644 = Maternidade Cândida Vargas
2707527 = Maternidade Frei Damião
2400324 = Hospital Edson Ramalho
3039269 = Clim Hospital e Maternidade
3056724 = Unimed
2400243 = HU
2399989 = Hospital HapClinica
3001377 = Hospital Joao Paulo II
5654319 = HU Jaguaribe
2399814 = Hospital Exercito
"""
df = df.query('CODESTAB == 2399644 | CODESTAB == 2707527')
del df['CODESTAB']

# Reorganizando as informações não dadas
df['MESPRENAT'].value_counts()
df = df.query('MESPRENAT != 99'
df = df.query('MESPRENAT != 10')

## Transformando a data de nascimento em timestamp
#df['DTNASC'] = [str(int(i)) for i in df['DTNASC']]
#df['DTNASC'] = ['0'+i if len(i) == 7 else i for i in df['DTNASC']]
#df['DTNASC'] = [datetime.strptime(i, "%d%m%Y").timestamp() for i in df['DTNASC']]
#
## Transformando a hora do nascimento em contagem de segundos a partir de 00:00h
#hora_padrao = '0000'
#df['HORANASC'] = [str(int(i)) for i in df['HORANASC']]
#df['HORANASC'] = ['0'+i if len(i) == 3 else i for i in df['HORANASC']]
#df['HORANASC'] = ['00'+i if len(i) == 2 else i for i in df['HORANASC']]
#df['HORANASC'] = ['000'+i if len(i) == 1 else i for i in df['HORANASC']]
#df['HORANASC'] = [(datetime.strptime(i,'%H%M') - datetime.strptime(hora_padrao,'%H%M')).seconds for i in df['HORANASC']]

# Normalizando
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
df[['IDADEMAE', 'DTNASC', 'HORANASC', 'SEMAGESTAC']] = scaler.fit_transform(df[['IDADEMAE', 'DTNASC', 'HORANASC','SEMAGESTAC']])
df[['QTDFILVIVO','QTDPARTNOR', 'QTDPARTCES','CONSPRENAT']] = scaler.fit_transform(df[['QTDFILVIVO','QTDPARTNOR', 'QTDPARTCES', 'CONSPRENAT']])

# Transformando variáveis binárias
df = pd.get_dummies(df, prefix=['ESTCIVMAE_','ESCMAE_','GRAVIDEZ_','GESTACAO_' ,'CONSULTAS_', 'MESPRENAT_'],columns=['ESTCIVMAE','ESCMAE', 'GRAVIDEZ','GESTACAO','CONSULTAS', 'MESPRENAT'])

# Conjunto de Dados para ML 
y = df['PARTO']
del df['PARTO']
X = df[::]

# Train - Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 33)
   
# Melhor Resultados:
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_curve, auc

def ClassificationScore(y_test, y_pred):
    print('Precision',precision_score(y_test, y_pred))
    print('Recall',recall_score(y_test, y_pred))
    print('F1 score',f1_score(y_test, y_pred))
    print('Acurácia',accuracy_score(y_test, y_pred))
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)    
    print(f"Roc-Curve:{roc_auc:.3f}")

def OverallClassificationScore(y_test, y_pred):
    precisao = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    acuracia = accuracy_score(y_test, y_pred)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)  
    weighted_classificator_mean = ((precisao+(2*recall)+f1+acuracia+roc_auc)/6)*100
    print(f"{weighted_classificator_mean:.4f} %")
    
# =============================================================================
# Estratégia 1 - Árvores de Decisão - Fácil de explicar
# =============================================================================
from sklearn.tree import DecisionTreeClassifier

# Usando busca-aleatória para definir os hiper-parâmetros
dt = DecisionTreeClassifier(max_depth=11,random_state=42)
dt.fit(X_train, y_train)
y_dt_pred = dt.predict(X_test)
ClassificationScore(y_test, y_dt_pred)
OverallClassificationScore(y_test, y_dt_pred)


# Melhor Modelo de Decision Tree
dt = DecisionTreeClassifier(max_depth=12,
                                max_features=27,
                                min_samples_leaf=0.0027,
                                min_samples_split=0.0001,
                                min_impurity_decrease= 0.000169,
                                random_state=42)
dt.fit(X_train, y_train)
ClassificationScore(y_test, dt.predict(X_test))
OverallClassificationScore(y_test, dt.predict(X_test))

# Plotando a árvore
from sklearn import tree
from sklearn.tree import export_graphviz

export_graphviz(dt, out_file='arvore_cls.dot', 
                feature_names = X.columns,
                class_names = 'PARTO',
                rounded = True, proportion = False, 
                precision = 2, filled = True)

from subprocess import call
call(['dot', '-Tpng', 'arvore_cls.dot', '-o', 'arvore_cls.png', '-Gdpi=600'])

import matplotlib.pyplot as plt
plt.figure(figsize = (14, 18))
plt.imshow(plt.imread('arvore_cls.png'))
plt.axis('off');
plt.show();

# =============================================================================
# Estratégia 2 - Random Forest
# =============================================================================
from sklearn.ensemble import RandomForestClassifier

# Random Forest
rf = RandomForestClassifier(n_estimators=100,n_jobs=-1)
rf.fit(X_train, y_train)
ClassificationScore(y_test, rf.predict(X_test))

# Precisamos ajustar os hiperparâmetros
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

param_grid = {'max_depth':[7,8,9,10,11,12,13],
              'min_samples_split':[0.05, 0.06, 0.07],
              'min_samples_leaf': [0.0001,0.0005,0.00075,0.001],
              'n_estimators':[14, 16, 18, 20, 22],
              'max_features':[9, 10, 11],
              'bootstrap':[True, False]}

gs = RandomizedSearchCV(rf, param_distributions=param_grid, cv=5, verbose=2, n_jobs=-1)
gs.fit(X_train, y_train)
gsRF = gs.predict(X_test)
ClassificationScore(y_test, gsRF)
gs.best_params_

# 1 Fit
rf = RandomForestClassifier(max_depth=13, max_features=10, min_samples_leaf=0.0005,
                            min_samples_split=0.0005, n_estimators=23, bootstrap=False,
                            n_jobs=-1, random_state=10)
rf.fit(X_train, y_train)
ClassificationScore(y_test, rf.predict(X_test))
OverallClassificationScore(y_test, rf.predict(X_test))

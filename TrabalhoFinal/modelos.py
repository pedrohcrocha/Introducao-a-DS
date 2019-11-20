"""
OBSERVAÇÃO:
Para testar diferentes especificações dos modelos, basta comentar uma das linhas
e rodar novamente.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Função que mostra as métricas do modelo 
def ClassificationScore(y_test, y_pred):
    print(f"Precision = {precision_score(y_test, y_pred)*100:.4f} %")
    print(f"Recall = {recall_score(y_test, y_pred)*100:.4f} %")
    print(f"F1 score = {f1_score(y_test, y_pred)*100:.4f} %")
    print(f"Acurácia = {accuracy_score(y_test, y_pred)*100:.4f} %")

# Importando a base de dados
df = pd.read_csv('nascidos_vivos_2017-2014.csv', sep=',')

# Transformando a coluna PARTO em binária
df['PARTO'] = np.where(df['PARTO'] == 2, 1, 0)

# Avaliando as variáveis numéricas
df['IDADEMAE'].value_counts(sort=False)
df['QTDPARTNOR'].value_counts(ascending=True)
df['QTDPARTCES'].value_counts(ascending=True)

# Filtrando para gestantes acima 18 anos (OPCIONAL)
df = df.query('IDADEMAE >= 18')

# Selecionando as colunas que desejamos trabalhar e retirando as NA's
df.columns
df = df[['ANO','CODESTAB','IDADEMAE','QTDPARTNOR', 'QTDPARTCES', 'CONSPRENAT','GRAVIDEZ',
         'TPROBSON', 'KOTELCHUCK', 'MESPRENAT',
         'PARTO']]
df = df.dropna()

# Filtrando para apenas um estabelecimento
df['CODESTAB'].value_counts()
df = df.query('CODESTAB == 2399644')
#df = df.query('CODESTAB == 2707527')
#df = df.query('CODESTAB == 2400324')
#df = df.query('CODESTAB == 3039269')
#df = df.query('CODESTAB == 3056724')
del df['CODESTAB']

# Testando os diferentes tipos de normalização de variáveis
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler(feature_range=(0,1))
stdscaler = StandardScaler()
#df[['IDADEMAE','QTDPARTNOR', 'QTDPARTCES','CONSPRENAT']] = scaler.fit_transform(df[['IDADEMAE', 'QTDPARTNOR', 'QTDPARTCES','CONSPRENAT']])
#df[['IDADEMAE','QTDPARTNOR', 'QTDPARTCES','CONSPRENAT']] = stdscaler.fit_transform(df[['IDADEMAE', 'QTDPARTNOR', 'QTDPARTCES','CONSPRENAT']])

# Retirando as informações da coluna MESPRENAT irrelevantes para a análise
df = df.query('MESPRENAT != 99')
df = df.query('MESPRENAT != 10')
del df['MESPRENAT']

# Criando dummies para variáveis categóricas
#df = pd.get_dummies(df, prefix=['MESPRENAT_'],columns=['MESPRENAT'])
df = pd.get_dummies(df, prefix=['GRAVIDEZ_'],columns=['GRAVIDEZ'])
#del df['GRAVIDEZ']
df = pd.get_dummies(df, prefix=['TPROBSON_'],columns=['TPROBSON'])
#del df['TPROBSON']
df = pd.get_dummies(df, prefix=['KOTELCHUCK_'],columns=['KOTELCHUCK'])
#del df['KOTELCHUCK']

# Filtrando a base do ano de 2017 - Será nossa base de teste
df_teste = df.query('ANO == 2017')
y_test = df_teste['PARTO']
del df_teste['PARTO']
del df_teste['ANO']

# Filtrando a base de treino
X_test = df_teste[::] 
df = df.query('ANO < 2017')
y = df['PARTO']
del df['PARTO']
del df['ANO']
X = df[::]

# =============================================================================
# Algortimos de classificação
# =============================================================================
# Decision Tree
dt = DecisionTreeClassifier(max_depth=4,
                            max_features=12,
                            random_state=8)
dt.fit(X, y)
print(i,'dt =',ClassificationScore(y_test, dt.predict(X_test)))

# Random Forest
rf = RandomForestClassifier(max_depth=10,
                            max_features=7,
                            n_estimators=181,
                            bootstrap=True,
                            n_jobs=-1,
                            random_state=10)
rf.fit(X,y)
print('rf = ',ClassificationScore(y_test, rf.predict(X_test)))

## Multi-Layer Perceptron
#nn = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10,10), random_state=42)
#nn.fit(X, y)                         
#print('Neural Networks =', nn.score(X_test, y_test)*100,'%')
#ClassificationScore(y_test, nn.predict(X_test))

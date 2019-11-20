
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# Base
df = pd.read_csv('nascidos_vivos_2017-2014.csv', sep=',')

# Transformando a coluna PARTO em binária
df['PARTO'] = np.where(df['PARTO'] == 2, 1, 0)

# Colunas
df = df[['ANO','CODESTAB','IDADEMAE','QTDPARTNOR', 'QTDPARTCES', 'CONSPRENAT','GRAVIDEZ',
         'TPROBSON', 'KOTELCHUCK', 'MESPRENAT',
         'PARTO']]

# Maternidades Cândida Vargas e Frei Damião
df = df.query('CODESTAB == 2399644 | CODESTAB == 2707527')
df = df.query('MESPRENAT != 99')
df = df.query('MESPRENAT != 10')
df = df.query('CONSPRENAT != 99')
df = df.dropna()

# Bases para cada uma das maternidades
df_candida = df.query('CODESTAB == 2399644')
df_frei = df.query('CODESTAB == 2707527')

# Variáveis categóricas
df_cand_cat = df_candida[['TPROBSON','KOTELCHUCK', 'MESPRENAT']]
df_frei_cat = df_frei[['TPROBSON','KOTELCHUCK', 'MESPRENAT']]

# Histogramas - Selecionar tudo até plt.show() e rodar
fig = plt.figure(figsize=(18, 10))
plt.subplot(1,3,1)
plt.hist([df_cand_cat['TPROBSON'], df_frei_cat['TPROBSON']], 
         label = ['Cândida Vargas','Frei Damião'],
         alpha=0.7, rwidth=0.85)
plt.legend(loc='upper right')
plt.grid(axis='y', alpha=0.75)
plt.ylabel('Observações')
plt.title('Histograma - Tipo de Robson')

plt.subplot(1,3,2)
plt.hist([df_cand_cat['KOTELCHUCK'], df_frei_cat['KOTELCHUCK']],
         label = ['Cândida Vargas','Frei Damião'],
         alpha=0.7, rwidth=0.85)
plt.legend(loc='upper right')
plt.grid(axis='y', alpha=0.75)
plt.ylabel('Observações')
plt.title('Histograma - Índice Kotelchuck')

plt.subplot(1,3,3)
plt.hist([df_cand_cat['MESPRENAT'], df_frei_cat['MESPRENAT']],
         label = ['Cândida Vargas','Frei Damião'],
         alpha=0.7, rwidth=0.85)
plt.legend(loc='upper right')
plt.grid(axis='y', alpha=0.75)
plt.ylabel('Observações')
plt.title('Histograma - Mês do 1º Pré-Natal')
plt.show()

# Variáveis numércias
df_cand_num = df_candida[['IDADEMAE', 'QTDPARTNOR', 'QTDPARTCES', 'CONSPRENAT']]
df_frei_num = df_frei[['IDADEMAE', 'QTDPARTNOR', 'QTDPARTCES', 'CONSPRENAT']]

# Tabela estatística
df_cand_num.describe()
df_frei_num.describe()

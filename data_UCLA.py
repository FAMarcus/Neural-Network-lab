import numpy as np
import pandas as pd

admissions = pd.read_csv('data/UCLA_Admission.csv')
# mesmo que student_data.csv

# One-hot coding: Make dummy variables for rank
data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis=1) # Adiciona à tabela de dados 4 colunas para os ranks
data = data.drop('rank', axis=1) # elimina da tabela a coluna 'rank'

#Standarize by max values (Original do curso) 
data['gre'] = data['gre']/800
data['gpa'] = data['gpa']/4
    
# Split off random 10% of the data for testing
np.random.seed(42)
sample = np.random.choice(data.index, size=int(len(data)*0.85), replace=False)
#data, test_data = data.ix[sample], data.drop(sample)  #.ix is deprecated.
data, test_data = data.loc[sample], data.drop(sample)

# Split into features and targets
features, targets = data.drop('admit', axis=1), data['admit']
features_test, targets_test = test_data.drop('admit', axis=1), test_data['admit']

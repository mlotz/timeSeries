import numpy as np
import scipy
import matplotlib
import pandas as pd
import sklearn
import tensorflow
import keras
import matplotlib.pyplot as plt
from datetime import datetime


#########Wczytanie danych############
dataset = pd.read_csv('dataset_PSE (1).csv', delimiter = ',')

########Wizualizacja surowych probek###########

#Vds= dataset.iloc[:,3] # ZAP, raw, unstamped.
#visualize ZAP
#plt.figure()
#Vds.plot()
#plt.show()
#plt.savefig('RAW_Unstamped_ZAP.pdf')

############Sprawdzenie i naprawa danych#############

#identyfikacja wadliwych rekordow
#dataset['Godzina'].map(int)

dataset = dataset[dataset['Godzina'] != '2A']
#odrzucono 2 rekordow
#sprawdzenie danych
HCheck = dataset[dataset['Godzina'].map(int).isin([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])]
#print(HCheck)
#19669 rekordow



#dataset['DH'] = dataset['Data'].map(str)+ dataset['Godzina']

#db_ds = dataset[dataset['Godzina'] == '2A']
#print(db_ds)
#print(dataset)
#db_ds = dataset[dataset['Data'] == 20171029]
#print(db_ds)
#print(dataset)


##########Konwersja etykiet probek#############


timeStamps = pd.to_datetime(dataset.Data,format='%Y%m%d') + dataset.Godzina.astype('timedelta64[h]')
#print(timeStamps)

dataset['timeStamp'] = timeStamps


dataset = dataset.drop(['Godzina','Data'], 1)
dataset = dataset.reindex(columns = ['timeStamp', 'ZAP'])


#Indeksowanie po datach
dataset.set_index('timeStamp', inplace=True)


#Reindeksowanie do pelnej dziedziny dat.
timeIndexRange = pd.date_range('2016-01-01 3:00', '2018-04-01', freq='H')
dataset.index = pd.DatetimeIndex(dataset.index)
#dataset = dataset.reindex(timeIndexRange, fill_value=0)
dataset = dataset.reindex(timeIndexRange, fill_value=float('NaN'))
#print(dataset)

##########sprawdzenie struktury brakujacych danych########
#missingVals = dataset[dataset['ZAP'].isnull()]
#print(missingVals)


############uzupelniam NaN przez mean#############
dataset = dataset.interpolate(method='linear')


##########Wizualizacja interpolacji brakujacych danych #######

#print (dataset.loc((dataset['index'] > pd.to_datetime('2016-03-03 13:00:00')) & (dataset['index'] <= pd.to_datetime('2016-03-04 14:00:00'))))
anomallyIndexRange = pd.date_range('2016-03-03 1:00', '2016-03-04 23:00', freq='H')
#print (AnomallyIndexRange)
anomally = dataset.ix[anomallyIndexRange]
plt.figure()
anomally.plot()
#plt.show()
plt.savefig('anomally_linear_ZAP.pdf')
#######Wizualizacja oetykietowanych probek########
#weekly_summary = pd.DataFrame()
#weekly_summary['ZAP'] = dataset['ZAP'].resample('W').mean()



#plt.figure()
#dataset.plot()
#plt.show()
#plt.savefig('filled_with_mean_ZAP.pdf')


#
#print(dataset.iloc[:,3])

#print (dataset.iloc[:,1])
#dataset['Godzina'] = dataset['Godzina'].resample(freq='H', periods=24)
#dataset['DH'] = dataset['Data'].map(str)+ dataset['Godzina']
#print(dataset)






#print(dataset)

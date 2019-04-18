import pandas as pd
import math
import matplotlib.pyplot as plt

#########Wczytanie danych############
dataset = pd.read_csv('dataset_PSE (1).csv', delimiter=',')
#print(dataset)
series = pd.Series(dataset['ZAP'])
#print(series)
Adatas = dataset[dataset['Godzina'] == '2A']
dataset = dataset[dataset['Godzina'] != '2A']
#print(Adatas)
timeStamps = pd.to_datetime(dataset.Data,format='%Y%m%d') + dataset.Godzina.astype('timedelta64[h]')
#print(timeStamps)

dataset['timeStamp'] = timeStamps
dataset = dataset.drop(['Godzina','Data'], 1)
dataset = dataset.reindex(columns = ['timeStamp', 'ZAP'])
dataset.set_index('timeStamp', inplace=True)
timeIndexRange = pd.date_range('2016-01-01 3:00', '2018-04-01', freq='H')
dataset.index = pd.DatetimeIndex(dataset.index)
dataset = dataset.reindex(timeIndexRange, fill_value=float('NaN'))



#print (pd.to_datetime('20170326' ,format='%Y%m%d') + pd.to_timedelta(arg=2, unit='h'))
#print (pd.to_datetime('20180325' ,format='%Y%m%d') + pd.to_timedelta(arg=2, unit='h'))

dataset = dataset.drop(pd.to_datetime('20170326' ,format='%Y%m%d') + pd.to_timedelta(arg=2, unit='h'))
dataset = dataset.drop(pd.to_datetime('20180325' ,format='%Y%m%d') + pd.to_timedelta(arg=2, unit='h'))

#print(dataset[dataset['ZAP'].isnull()])

#fix stuff
for i, j in dataset[dataset['ZAP'].isnull()].iterrows():
    #print(i, j)
    #print(i+pd.to_timedelta(arg=24, unit='h'))
    #print(dataset.loc[i]['ZAP'],math.isnan(dataset.loc[i]['ZAP']))
    forward_24h_isnan = math.isnan(dataset.loc[i + pd.to_timedelta(arg=24, unit='h')]['ZAP'])
    backward_24h_isnan = math.isnan(dataset.loc[i + pd.to_timedelta(arg=-24, unit='h')]['ZAP'])
    forward_24h_val = dataset.loc[i+pd.to_timedelta(arg=24, unit='h')]['ZAP']
    backward_24h_val = dataset.loc[i + pd.to_timedelta(arg=-24, unit='h')]['ZAP']
    #print(i, backward_24h_val, backward_24h_isnan, forward_24h_val, forward_24h_isnan)

    if not forward_24h_isnan and not backward_24h_isnan:
        dataset.loc[i]['ZAP'] = (backward_24h_val + forward_24h_val)/2
    if not forward_24h_isnan:
        pass
        dataset.loc[i]['ZAP'] = forward_24h_val
    if not backward_24h_isnan:
        pass
        dataset.loc[i]['ZAP'] = backward_24h_val
    if forward_24h_isnan and backward_24h_isnan:
        pass
        dataset.loc[i]['ZAP'] = dataset['ZAP'].mean()


TSanomallyIndexRange = pd.date_range('2016-03-03 1:00', '2016-03-04 23:00', freq='H')
TSanomally = dataset.ix[anomallyIndexRange]
plt.figure()
TSanomally.plot()
plt.savefig('v2_anomalia.pdf')



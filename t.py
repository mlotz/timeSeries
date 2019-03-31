import numpy as np
import scipy
import matplotlib
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from datetime import datetime
from loss_mse import loss_mse_warmup
from custom_generator import batch_generator
#Keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau








#########Wczytanie danych############
dataset = pd.read_csv('dataset_PSE (1).csv', delimiter = ',')

#print(dataset)
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


#############Analiza sasiedztwa uszkodzonych danych##############

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


################Indeksowanie po datach###############
dataset.set_index('timeStamp', inplace=True)


############Reindeksowanie do pelnej dziedziny dat.#################
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
#plt.figure()
#anomally.plot()
#plt.show()
#plt.savefig('anomally_linear_ZAP.pdf')


#######Wizualizacja oetykietowanych probek########
#weekly_summary = pd.DataFrame()
#weekly_summary['ZAP'] = dataset['ZAP'].resample('W').mean()
#print(dataset)


#plt.figure()
#dataset.plot()
#plt.show()
#plt.savefig('filled_with_linear_interpolation_ZAP.pdf')


#
#print(dataset.iloc[:,3])

#print (dataset.iloc[:,1])
#dataset['Godzina'] = dataset['Godzina'].resample(freq='H', periods=24)
#dataset['DH'] = dataset['Data'].map(str)+ dataset['Godzina']
#print(dataset)


##########shiftowanie danych w przeszlosc#####
shift_days = 1
shift_steps = shift_days * 24  # Number of hours.

df_pred = dataset['ZAP'].shift(-shift_steps)

#print(df_pred)
#print(dataset)

######DO NumPy

x_data = dataset.values[0:-shift_steps] ##pierwotne dane

y_data = df_pred.values[:-shift_steps] ##wyjscia modelu 24 probki wstecz
y_data = y_data.reshape(-1, 1)
#print (y_data)
num_train = len(x_data)
#print(num_train)


##Podzial danych na zestawy uczace i testowe
train_split = 0.9

num_train = int(train_split * num_train)

x_train = x_data[0:num_train]
x_test = x_data[num_train:]
len(x_train) + len(x_test)


y_train = y_data[0:num_train]
y_test = y_data[num_train:]


print ('X: ucz:'+str(len(x_train))+' test:'+str(len(x_test))+' suma:'+str(len(x_train) + len(x_test)))
print ('Y: ucz:'+str(len(y_train))+' test:'+str(len(y_test))+' suma:'+str(len(y_train) + len(y_test)))


#Skalowanie wejsc do <0,1>
print("X::")
print("Min:", np.min(x_train))
print("Max:", np.max(x_train))

x_scaler = MinMaxScaler()
x_train_scaled = x_scaler.fit_transform(x_train)

print("Min:", np.min(x_train_scaled))
print("Max:", np.max(x_train_scaled))
x_test_scaled = x_scaler.transform(x_test)


y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

##datagen ?
print(x_train_scaled.shape)
print(y_train_scaled.shape)

batch_size = 256
sequence_length = 24 * 7 * 8
generator = batch_generator(batch_size=batch_size, sequence_length=sequence_length, num_train=num_train, x_train_scaled=x_train_scaled, y_train_scaled=y_train_scaled)

#x_batch, y_batch = next(generator)

#print(x_batch.shape)
#print(y_batch.shape)

##Val set

validation_data = (np.expand_dims(x_test_scaled, axis=0),
                   np.expand_dims(y_test_scaled, axis=0))

##model
model = Sequential()
model.add(GRU(units=32, return_sequences=True,input_shape=(None, 1,)))
model.add(Dense(1, activation='sigmoid'))

optimizer = RMSprop(lr=1e-3)
model.compile(loss=loss_mse_warmup, optimizer=optimizer)
model.summary()


##Checkpoints
path_checkpoint = '23_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)
##EarlyStop
callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=5, verbose=1)
##logi
callback_tensorboard = TensorBoard(log_dir='./23_logs/',
                                   histogram_freq=0,
                                   write_graph=False)
##learning rate
callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       min_lr=1e-4,
                                       patience=0,
                                       verbose=1)

#callback vector
callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_tensorboard,
             callback_reduce_lr]


##Train

#odkomentowac by uczyc

#model.fit_generator(generator=generator,epochs=20, steps_per_epoch=100, validation_data=validation_data, callbacks=callbacks)

##load checkpoint

#Odkomentowac by uzywac nauczonego modelu
try:
    model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)
##//Train

##test set
result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),y=np.expand_dims(y_test_scaled, axis=0))

print("loss (test-set):", result)



##rysownie
#print(len(x_train_scaled))
#plot_comparison(start_idx=0, length=17000, train=False)

x = np.expand_dims(x_train_scaled, axis=0)
y_pred = model.predict(x)
y_train_pred_rescaled = y_scaler.inverse_transform(y_pred[0])

x = np.expand_dims(x_test_scaled, axis=0)
y_pred = model.predict(x)
y_test_pred_rescaled = y_scaler.inverse_transform(y_pred[0])



df_y_train = pd.DataFrame(data=y_train_pred_rescaled[1:,0])
TIR = pd.date_range('2016-01-01 3:00', periods = len(df_y_train), freq='H')
df_y_train.index = TIR
df_y_train_true = pd.DataFrame(data=y_train[1:,0])
df_y_train_true.index = TIR
#print(df_y_train)


plt.figure()
plt.plot(df_y_train_true, label='true')
plt.plot(df_y_train, label='pred')
plt.ylabel('ZAP')
plt.legend()
plt.savefig('24h_pred_learn.pdf')


df_y_test = pd.DataFrame(data=y_test_pred_rescaled[1:,0])
TIR = pd.date_range('2018-01-07 23:00:00', periods = len(df_y_test), freq='H')
df_y_test.index = TIR
df_y_test_true = pd.DataFrame(data=y_test[1:,0])
df_y_test_true.index = TIR
#print(df_y_test)


plt.figure()
plt.plot(df_y_test_true, label='true')
plt.plot(df_y_test, label='pred')
plt.ylabel('ZAP')
plt.legend()
plt.savefig('24h_pred_test.pdf')




####Agregat###

##Suma na 24h oknie czasowym
df_agregat_train_true = df_y_train_true.rolling(min_periods=24, window=24).sum()
df_agregat_train = df_y_train.rolling(min_periods=24, window=24).sum()
df_agregat_test_true = df_y_test_true.rolling(min_periods=24, window=24).sum()
df_agregat_test = df_y_test.rolling(min_periods=24, window=24).sum()



#shift o 24 do tylu, bo rolling oblicza ostatnie 24, a ja chce nastepne 24.
df_agregat_train_true[0] = df_agregat_train_true[0].shift(-24)
df_agregat_train[0] = df_agregat_train[0].shift(-24)
df_agregat_test_true[0] = df_agregat_test_true[0].shift(-24)
df_agregat_test[0] = df_agregat_test[0].shift(-24)

#rysowanie

plt.figure()
plt.plot(df_agregat_train_true, label='true')
plt.plot(df_agregat_train, label='pred')
plt.ylabel('ZAP')
plt.legend()
plt.savefig('24h_agregat_learn.pdf')

plt.figure()
plt.plot(df_agregat_test_true, label='true')
plt.plot(df_agregat_test, label='pred')
plt.ylabel('ZAP')
plt.legend()
plt.savefig('24h_agregat_test.pdf')
##EOF##

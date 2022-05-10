import pandas as pd
import numpy as np
import openpyxl

# np.random.seed(202259)

data_train = pd.read_excel(f'02 Training_2021.xlsx', sheet_name="PCM_Norm", engine='openpyxl')
data_test = pd.read_excel(f'03 Testing_2022.xlsx', sheet_name="PCM_Norm", engine='openpyxl')
data_pred = pd.read_excel(f'06 Predict.xlsx', sheet_name="PCM_Norm", engine='openpyxl')

train = np.array(data_train).astype(float)
test = np.array(data_train).astype(float)
pred = np.array(data_pred).astype(float)

a = float(1)
alfa = 0.05

def sigmoid(a,x):
    return 1.0/(1 + np.exp(-a*x))


'''TRAINING THE MODEL'''
x_in = train[:,0:10]
y_out = train[:,10:11]

print()
print(f'Tamaño de datos training: {np.shape(x_in)}')
print(f'Tamaño de salidas training: {np.shape(y_out)}')
print()

N = np.shape(x_in)[1]
M = np.shape(y_out)[1]
L = 2*M + 2*N #neuronas
Q = np.shape(x_in)[0]

print('Condiciones de PMC')
print(f'Neuronas: {L}')
print(f'Aprendizaje: {alfa}')

w_h = np.random.random((L,N))
w_o = np.random.random((M,L))

E, i = 1, 0

while (E > 0.0001):
    for _ in range(Q):

        '''FORWARD'''
        net_h = np.reshape(w_h @ x_in[_].T,(L,1))
        y_h = sigmoid(1,net_h)
        net_o = np.reshape(w_o @ y_h,(M,1))
        y_o = sigmoid(1,net_o)

        '''BACKPROPAGATION'''
        delta_o = (np.reshape(y_out[_].T,(M,1)) - y_o) * y_o * (1 - y_o)
        delta_h = y_h * (1-y_h) * ((w_o.T) @ delta_o)
        w_o += alfa * delta_o @ np.reshape(y_h.T,(1,L))
        w_h += alfa * delta_h @ np.reshape(x_in[_],(1,N))

        '''ERROR'''
        E = abs((delta_o[0]**2)**0.5)

    i+=1

print()
print(f'iteraciones: {i} Error: {E}')
print()
print(f' pesos entrada {w_h}')
print()
print(f'pesos salida {w_o}')
print()


'''TESTING THE MODEL'''

x_in_t = test[:,0:10]
y_out_t = test[:,10:11]
print(f'Tamaño de datos testing: {np.shape(x_in_t)}')
print(f'Tamaño de salidas testing: {np.shape(y_out_t)}')
print()


P = np.shape(x_in_t)[0]
output = []
for _ in range(P):
    net_h_t = np.reshape(w_h @ x_in_t[_].T,(L,1))
    y_h_t = sigmoid(1,net_h_t)
    net_o_t = np.reshape(w_o @ y_h_t,(M,1))
    y_o_t = sigmoid(1,net_o_t)

    output.append(y_o_t)

output = np.asarray(output)
output = output.flatten()
output = np.reshape(output,(P,1))
print('Salida')
print(np.round(output,1))
print()

'''ACCURACY'''
dif = abs(y_out_t - np.round(output))
nnz_t = np.count_nonzero(dif)

acc = nnz_t / P

print(f'exactitud: {(1-np.round(acc,4))*100} %')
print()



'''USING THE MODEL - PREDICTION'''

x_in_p = pred[:,0:10]

print(f'Tamaño de datos predicting: {np.shape(x_in_p)}')
print()


PRDS = np.shape(x_in_p)[0]
predictions = []
for _ in range(PRDS):
    net_h_t = np.reshape(w_h @ x_in_p[_].T,(L,1))
    y_h_t = sigmoid(1,net_h_t)
    net_o_t = np.reshape(w_o @ y_h_t,(M,1))
    y_o_t = sigmoid(1,net_o_t)

    predictions.append(y_o_t)

predictions = np.asarray(predictions)
predictions = predictions.flatten()
predictions = np.reshape(predictions,(PRDS,1))
print('Salida')
print(np.round(predictions,1))
print()

Pred_output = np.round(output)
data_out = pd.concat([data_pred,pd.DataFrame(Pred_output)],axis = 1)
data_out.columns = ["Shift",
	                   "Group",
                       "Start_Temp [°C]",
                       "Drop_Temp [°C]",
                       "spec_Energy [Wh/kg]",
                       "Rolls [#]",
                       "Mass [kg]",
                       "Peak Power [kW]",
                       "Waiting [s]",
                       "Ram_down [s]",
                       "Pred."]

data_out.to_excel('PCM_0522.xlsx',index=False)

import numpy as np
import pandas as pd
import math
import copy

class MLP:
    def __init__(self, data, depth, nhidden, ntarget, nbatch, maxepoch, learning_rate):
        #Initial state:
        #Data sudah diatur sedemikian rupa sehingga feature target terdapat pada ntarget column paling kiri
        #Contoh: bila ntarget = 2 maka column terakhir dan satu column sebelum terakhir adalah column terget
        
        #Validasi
        if (ntarget > len(data.columns)):
            print("ntarget cannot be greater than data's feature")
            return None

        #Inisialisasi
        self.testdata = copy.deepcopy(data)
        self.data = copy.deepcopy(data) #dataset yang dipakai
        self.depth = depth #kedalaman MLP (banyaknya hidden layer)
        self.nhidden = nhidden #banyaknya perceptron dalam satu layer hidden
        self.ntarget = ntarget
        self.nY = ntarget #banyaknya node/feature target
        self.nX = len(self.data.columns) - self.nY #banyaknya node/feature input
        self.nbatch = nbatch #banyaknya instance dalam satu batch (kita bikin mini batch boys jangan lupa)
        self.maxepoch = maxepoch #jumlah epoch yang dilakukan (urang denger dari ditput harus banyak hmmmm)
        self.learning_rate = learning_rate #learning rate yang diinginkan

        #Mengubah column district mendjadi continue
        self.labeling()

        # Menginisialisasi matriks weights sebanyak depth + 1 (1 buat ke output)
        self.weights = []
        for i in range(self.depth + 1):
            if (i == 0): #dari input layer ke hidden layer pertama
                self.weights.append(np.random.rand(self.nX + 1, self.nhidden)*math.sqrt(2/(self.nX + self.nhidden)))
            elif (i == depth): #dari hidden layer terakhir ke output
                self.weights.append(np.random.rand(self.nhidden + 1, self.nY)*math.sqrt(2/(self.nhidden + self.nhidden)))
            else: #dari hidden layer ke hidden layer
                self.weights.append(np.random.rand(self.nhidden + 1, self.nhidden)*math.sqrt(2/(self.nhidden + self.nY)))                                    
        #Proses training
        self.train()

    def isContinue(self, series):
        if (isinstance(series.unique()[0], str)):
            return False
        else:
            if (len(series.unique()) <= 5):
                return False
            else:
                return True

    def label(self, columns, iterable):
        for column in columns:
            localiterable = dict()
            iterlabel = 0
            if (not self.isContinue(self.data[column])):
                columns_name = self.data[column].unique().tolist()
                self.discrete_column[column] = True                    
                new_data = []
                space = 1/(len(columns_name) - 1)
                
                i = 0
                local_class_array = []
                while i <= 1:
                    local_class_array.append(i)
                    i += space 
                local_class_array = np.array(local_class_array)
                self.class_array[column] = local_class_array

                for name in columns_name:
                    localiterable[iterlabel] = column + '::' + name
                    iterlabel += 1
                
                for value in self.data[column]:
                    for i in range(len(columns_name)):
                        if value == columns_name[i]:
                            new_data.append(i*space)
                
                self.data.drop(columns=column, inplace=True)
                NewData = pd.DataFrame(new_data, columns=[column], index=self.data.index)
                self.data = pd.concat([self.data, NewData], axis = 1)
            else:
                self.discrete_column[column] = False
            iterable[column] = localiterable

    def labeling(self):
        #Specify column mana yang jadi target, mana yang feature
        self.target = []
        self.feature = []
        for i in range(len(self.data.columns) - 1, len(self.data.columns) - 1 - self.nY, -1):
            self.target.append(self.data.columns[i])
        for i in range(len(self.data.columns) - self.nY):
            self.feature.append(self.data.columns[i])
        
        self.class_array = dict()
        self.discrete_column = dict() #column yang distinct (!continue) bakal punya value True di dict ini
        self.inmapping = dict()
        self.label(columns=self.feature, iterable=self.inmapping)

        #sama aja kaya yang diatas cuma buat Y        
        self.outmapping = dict()
        self.label(columns=self.target, iterable=self.outmapping)

    def sigmoid(self, net):
        #net adalah matriks of float
        #return matriks of float
        #formula : 1/(1+(e^-x))
        for i in range(len(net)):
            for j in range(len(net[i])):
                net[i][j] = 1/(1+(math.exp(-1*net[i][j])))
        
        return net

    def feedforward(self, X):
        self.hidden = []
        inp = X.to_numpy()
        for i in range(self.depth + 1):
            inp = self.sigmoid(np.dot(self.concat(inp), self.weights[i]))
            if (i < self.depth): 
                self.hidden.append(inp)
            else:
                out = inp
        return out

    def concat(self, matriks):
        temp = []
        for key, value in enumerate(matriks):
            temp.append(np.concatenate([[1], value]))
        return np.array(temp)

    def train(self):
        for e in range(self.maxepoch):
            for i in range(0, len(self.data), self.nbatch):
                if (i + self.nbatch - 1 > len(self.data)):
                    output = self.feedforward(self.data.iloc[i:len(self.data), :len(self.data.columns) - self.nY])
                else:
                    output = self.feedforward(self.data.iloc[i:i+self.nbatch, :len(self.data.columns) - self.nY])
                target = self.data.iloc[i:i+self.nbatch, len(self.data.columns) - self.nY:].to_numpy()
                global_output = output.copy()
                global_input = self.data.iloc[i:i+self.nbatch, :len(self.data.columns) - self.nY].copy().to_numpy()
                delta = np.multiply(np.multiply(output, 1 - output), target - output)
                deltaw = self.learning_rate*np.dot(self.concat(self.hidden[self.depth - 1]).transpose(), delta)
                self.weights[self.depth] = self.weights[self.depth] + deltaw
                for j in range(self.depth - 1, -1, -1):
                    output = self.hidden[j]
                    sigma = np.dot(delta, self.weights[j + 1][1:,:].transpose())
                    delta = np.multiply(np.multiply(output, 1 - output), sigma)
                    if (j == 0):
                        deltaw = self.learning_rate*np.dot(self.concat(global_input).transpose(), delta)
                    else:  
                        deltaw = self.learning_rate*np.dot(self.concat(self.hidden[j - 1]).transpose(), delta)
                    self.weights[j] = self.weights[j] + deltaw
            # print(e, 0.5*np.mean(np.square(target - output)))
            print(e, 0.5*np.mean(np.square(target - output)), self.test(self.testdata))

    def predict(self, X):
        for column in self.feature:
            if self.discrete_column[column]:
                new_data = []
                for value in X[column]:
                    idx = list(self.inmapping[column].keys())[list(self.inmapping[column].values()).index(column + "::" + value)]
                    new_data.append(self.class_array[column][idx])
                X.drop(columns=column, inplace=True)
                NewData = pd.DataFrame(new_data, columns=[column], index=X.index)
                X = pd.concat([X, NewData], axis = 1)

        output = self.feedforward(X)
        
        result = []
        for i in range(len(X)):
            row_result = []
            for key, column in enumerate(self.target):         
                idx = np.argmax(1 - np.absolute(self.class_array[column] - output[i, key]))
                row_result.append(self.outmapping[column][idx].split('::')[1])
            result.append(row_result)
        
        return result
        
    def test(self, test):
        count = 0
        result = self.predict(test.iloc[:, :len(test.columns) - self.ntarget])
        for i, key in zip(range(len(test)), test.index):
            correct = True
            for j in range(len(self.target)):
                if result[i][j] != test[self.target[j]][key]:                
                    correct = False
            if correct:
                count += 1
            
        return [count, (count/len(test))*100]

if __name__ == "__main__":
    iris = pd.read_csv('iris.csv')
    #Kalo mau nyoba gini ya gaes, column 'Id' di irisnya diilangin dulu biar ga jadi feature
    mask = np.random.rand(len(iris)) < 0.8
    train = iris[mask].iloc[:, 1:]
    test = iris[~mask].iloc[:, 1:]
    Model1 = MLP(data=train, depth=2, nhidden=3, ntarget=1, nbatch=3, maxepoch=3000, learning_rate=0.1, labeling_mode = "single")
    
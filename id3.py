import numpy as np
import pandas as pd
import math
import copy

class Tree:
    def __init__(self, attribute, values):
        self.attribute = attribute
        self.children = {}
        for value in values:
            self.children[value] = None
    
    def attachChild(self, value, tree):
        if value in self.children:
            self.children[value] = tree
    
    def __str__(self):
        return self.toStr(1)
        
    def toStr(self, indent):
        strret = ""
        for child in self.children:
            for i in range(indent):
                strret += "-"
            
            strret += ">"
            strret += self.attribute.__str__()
            strret += " = "
            strret += child.__str__()
            strret += '\n'
            
            strret += self.children[child].toStr(indent + 3)
        
        return strret
        
            
class Leaf:
    def __init__(self, value):
        self.value = value
        
    def toStr(self, indent):
        strret = ""
        for i in range(indent):
                strret += "-"
        strret += ">"
        strret += self.value.__str__()
        strret += '\n'
        
        return strret

class ID3Tree:
    def __init__(self, Data, Prune=False, useGainRatio=False):
        self.useGainRatio = useGainRatio
        #if prune true
        # then split into Data & testData randomly
        
        #Deteksi attr yang continu
        Attribute = Data.iloc[:, :Data.columns.size-1].columns.to_numpy()
        cont = {}
        for attr in Attribute:
            if self.isContinue(attr, Data):
                cont[attr] = True
            else:
                cont[attr] = False
        
        self.cont = cont
        
        newData = Data.copy()
        
        #Diskritisasi
        changes = {}
        for attr in Attribute:
            if cont[attr]:
                self.makeDiscrete(attr, newData, changes)
                        
        self.Tree = self.fit(newData, Attribute)
        
        
        #Ngubah informasi tree
        self.changeLabel(self.Tree, changes, cont)
        
    def changeLabel(self, T, changes, cont):
        if (isinstance(T, Tree)):
            if cont[T.attribute]:
                newChild = {}
                for child in T.children:
                    if child == 0.0:
                        newChild['<= ' + changes[T.attribute].__str__()] = T.children[child]
                    elif child == 1.0:
                        newChild['> ' + changes[T.attribute].__str__()] = T.children[child]
                
                T.children = newChild
                
            for child in T.children:
                self.changeLabel(T.children[child], changes, cont)

        
    def makeDiscrete(self, attr, Data, changes):
        newData = Data.sort_values(attr).copy()
        newData = newData.reset_index(drop=True)
        splitPoint = []
        last = newData[newData.columns[newData.columns.size - 1]][0]
        lastValue = newData[attr][0]
        for index, row in newData.iterrows():
            if row[newData.columns[newData.columns.size - 1]] != last:
                splitPoint.append((row[attr] + lastValue)/2)
            last = row[newData.columns[newData.columns.size - 1]]
            lastValue = row[attr]

        maxIG = 0
        maxSplit = 0
        for split in splitPoint:
            newData = Data.copy()
            for index, row in newData.iterrows():
                if row[attr] <= split:
                    newData.at[index, attr] = 0
                else:
                    newData.at[index, attr] = 1

            newData0 = newData[newData[attr].isin([0])]
            newData1 = newData[newData[attr].isin([1])]

            IG = self.calculateEntropy(Data.iloc[:, Data.columns.size - 1:]) - newData0.index.size * self.calculateEntropy(newData0.iloc[:, newData0.columns.size - 1:])/Data.index.size - newData1.index.size * self.calculateEntropy(newData1.iloc[:, newData1.columns.size - 1:])/Data.index.size  
            if (IG > maxIG):
                maxIG = IG
                maxSplit = split

        for index, row in Data.iterrows():
            if row[attr] <= maxSplit:
                Data.at[index, attr] = 0
            else:
                Data.at[index, attr] = 1

        changes[attr] = maxSplit
        return maxSplit

        
    def fit(self, Data, Attr):
        if Attr.size == 0: #if Attribute is empty (All Attribute has been chosen above this branch)
            return Leaf(self.mostCommon(Data)) #return leaf with label = most common label in Data
        else:
            Xentropy = self.calculateEntropy(Data.iloc[:, Data.columns.size-1:]) #Calculate the entropy of Data
            if (Xentropy == 0): #If data has been clasified
                return Leaf(self.mostCommon(Data)) #then return a leaf that clasifies Data
            else:
                if self.useGainRatio:
                    Choosen = self.chooseAttrRatio(Data, Attr) #choose attribute from Attribute that best classifies Data
                else:
                    Choosen = self.chooseAttr(Data, Attr) #choose attribute from Attribute that best classifies Data
                T = Tree(Choosen, Data.iloc[:, :Data.columns.size-1][Choosen].value_counts().keys().to_numpy()) #Create new subtree with Choosen as its label and its value as its branch 
                
                for child in T.children:
                    newData = Data[Data[Choosen].isin([child])] #Let newData be subset of Data that have value of child in the Choosen Attribute
                    if (newData.index.size == 0): #if newData is empty
                        return Leaf(self.mostCommon(Data)) #then add new leaf with label = most common label of Data bellow this branch 
                    else: #else bellow this branch add subtree:
                        Next = self.fit(newData, np.setdiff1d(Attr, [Choosen]))
                        T.attachChild(child, Next)
                
                return T

    def isContinue(self, attr, Data):
        if (isinstance(Data[attr][0], str)):
            return False
        else:
            if (Data[attr].value_counts().size <= 5):
                return False
            else:
                return True
            
    def predict(self, record):
        if isinstance(record, pd.core.series.Series):
            return self._predict(record, self.Tree)
        else:
            print("ID3Tree.predict only takes pandas.core.series.Series as argument")
            return None
        
    def _predict(self, record, currNode):
        if isinstance(currNode, Leaf):
            return currNode.value
        else:
            if self.cont[currNode.attribute]:
                for child in currNode.children.keys():
                    if child[0] == '<':
                        treshold = float(child[3:])
                        if record[currNode.attribute] <= treshold:
                            nextNode = currNode.children[child]
                            return self._predict(record, nextNode) 
                    elif child[0] == '>':
                        treshold = float(child[2:])
                        if record[currNode.attribute] > treshold:
                            nextNode = currNode.children[child]
                            return self._predict(record, nextNode)
            else:
                nextNode = currNode.children[record[currNode.attribute]]
                return self._predict(record, nextNode)
            
    def mostCommon(self, Data):
        Classes = Data.iloc[:, Data.columns.size-1].value_counts()
        max = 0
        Choo = ''
        for c in Classes.keys():
            if Classes[c] > max:
                max = Classes[c]
                Choo = c

        return Choo
            
    def chooseAttr(self, Data, Attr):
        Chosen = Attr[0]
        maxIG = 0
        for a in Attr:
            IG = self.calculateEntropy(Data.iloc[:, Data.columns.size-1:])
            for value in Data[a].value_counts().keys():
                vData = Data[Data[a].isin([value])] 
                IG -= (vData[a].size/Data[a].size)*self.calculateEntropy(vData.iloc[:, vData.columns.size-1:])
            
            if (IG > maxIG):
                maxIG = IG
                Chosen = a
            
        return Chosen
    
    def chooseAttrRatio(self, Data, Attr):
        Chosen = ''
        maxGR = 0
        for a in Attr:
            IG = self.calculateEntropy(Data.iloc[:, Data.columns.size-1:])
            for value in Data[a].value_counts().keys():
                vData = Data[Data[a].isin([value])] 
                IG -= (vData[a].size/Data[a].size)*self.calculateEntropy(vData.iloc[:, vData.columns.size-1:])
                
            GR = IG/self.calculateEntropy(vData.iloc[:, vData.columns.size-1:])
            
            if (GR > maxGR):
                maxGR = GR
                Chosen = a
            
        return Chosen
                     
    def calculateEntropy(self, Y):
        classes = Y.iloc[:,0].value_counts()
        
        Entropy = 0.0
        for c in classes:
            Entropy -= c/Y.iloc[:,0].size*(np.log(c/Y.iloc[:,0].size)/np.log(2))
        
        return Entropy
        
    def __str__(self):
        return self.Tree.__str__()

    
# if __name__ == "__main__":
#     import numpy as np
#     import pandas as pd

#     iris = pd.read_csv('iris.csv')
#     tennis = pd.read_csv('play-tennis.csv')

#     tennisTree = ID3Tree(tennis)

#     print(tennisTree)

#     print(tennisTree.predict(tennis.iloc[3, :4]))


#     irisTree = ID3Tree(iris.iloc[:, 1:])

#     print(irisTree)

#     print(irisTree.predict(iris.iloc[3, 1:5]))

    
if __name__ == "__main__":
    tennis = pd.read_csv('data/play-tennis.csv')
    Tree1 = ID3Tree(tennis)
    print(Tree1)

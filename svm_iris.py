#!/usr/bin/env python
# coding: utf-8

# In[63]:


import sys
from numpy import *
#from svm import *
from os import listdir
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i,-1] == 0:
            data[i,-1] = -1
    # print(data)
    return data[:,:2], data[:,-1]

def selectJrand(i,m):
    j = i
    while j == i:
        j = int(random.uniform(0,m))
    return j

def clipAlpha(a_j,H,L):
    if a_j > H:
        a_j = H
    if L > a_j:
        a_j = L
    return a_j
class PlattSMO:
    def __init__(self,dataMat,classlabels,C,toler,maxIter,**kernelargs):#构造SMO数据结构
        self.x = array(dataMat)
        self.label = array(classlabels).transpose()
        self.C = C
        self.toler = toler
        self.maxIter = maxIter
        self.m = shape(dataMat)[0]
        self.n = shape(dataMat)[1]
        self.alpha = array(zeros(self.m),dtype='float64')
        self.b = 0.0
        self.eCache = array(zeros((self.m,2)))
        self.K = zeros((self.m,self.m),dtype='float64')
        self.kwargs = kernelargs
        self.SV = ()
        self.SVIndex = None
        self.w = array(zeros(self.n),dtype='float64')
        
        for i in range(self.m):
            for j in range(self.m):
                self.K[i,j] = self.kernelTrans(self.x[i,:],self.x[j,:])
    def calcEK(self,k):#计算分类误差
        fxk = dot(self.alpha*self.label,self.K[:,k])+self.b
        Ek = fxk - float(self.label[k])
        return Ek
    def updateEK(self,k):#更新分类误差
        Ek = self.calcEK(k)

        self.eCache[k] = [1 ,Ek]
    def selectJ(self,i,Ei):#SMO转为二次规划问题时进行变量选择
        maxE = 0.0
        selectJ = 0
        Ej = 0.0
        validECacheList = nonzero(self.eCache[:,0])[0]
        if len(validECacheList) > 1:
            for k in validECacheList:
                if k == i:continue
                Ek = self.calcEK(k)
                deltaE = abs(Ei-Ek)
                if deltaE > maxE:
                    selectJ = k
                    maxE = deltaE
                    Ej = Ek
            return selectJ,Ej
        else:
            selectJ = selectJrand(i,self.m)
            Ej = self.calcEK(selectJ)
            return selectJ,Ej

    def innerL(self,i):#更新参数
        Ei = self.calcEK(i)
        if (self.label[i] * Ei < -self.toler and self.alpha[i] < self.C) or                 (self.label[i] * Ei > self.toler and self.alpha[i] > 0):
            self.updateEK(i)
            j,Ej = self.selectJ(i,Ei)
            alphaIOld = self.alpha[i].copy()
            alphaJOld = self.alpha[j].copy()
            if self.label[i] != self.label[j]:
                L = max(0,self.alpha[j]-self.alpha[i])
                H = min(self.C,self.C + self.alpha[j]-self.alpha[i])
            else:
                L = max(0,self.alpha[j]+self.alpha[i] - self.C)
                H = min(self.C,self.alpha[i]+self.alpha[j])
            if L == H:
                return 0
            eta = 2*self.K[i,j] - self.K[i,i] - self.K[j,j]
            if eta >= 0:
                return 0
            self.alpha[j] -= self.label[j]*(Ei-Ej)/eta
            self.alpha[j] = clipAlpha(self.alpha[j],H,L)
            self.updateEK(j)
            if abs(alphaJOld-self.alpha[j]) < 0.00001:
                return 0
            self.alpha[i] +=  self.label[i]*self.label[j]*(alphaJOld-self.alpha[j])
            self.updateEK(i)
            b1 = self.b - Ei - self.label[i] * self.K[i, i] * (self.alpha[i] - alphaIOld) -                  self.label[j] * self.K[i, j] * (self.alpha[j] - alphaJOld)
            b2 = self.b - Ej - self.label[i] * self.K[i, j] * (self.alpha[i] - alphaIOld) -                  self.label[j] * self.K[j, j] * (self.alpha[j] - alphaJOld)
            if 0<self.alpha[i] and self.alpha[i] < self.C:
                self.b = b1
            elif 0 < self.alpha[j] and self.alpha[j] < self.C:
                self.b = b2
            else:
                self.b = (b1 + b2) /2.0
            return 1
        else:
            return 0

    def smoP(self):#优化过程
        iter = 0
        entrySet = True
        alphaPairChanged = 0
        while iter < self.maxIter and ((alphaPairChanged > 0) or (entrySet)):
            alphaPairChanged = 0
            if entrySet:
                for i in range(self.m):
                    alphaPairChanged+=self.innerL(i)
                iter += 1
            else:
                nonBounds = nonzero((self.alpha > 0)*(self.alpha < self.C))[0]
                for i in nonBounds:
                    alphaPairChanged+=self.innerL(i)
                iter+=1
            if entrySet:
                entrySet = False
            elif alphaPairChanged == 0:
                entrySet = True
        self.SVIndex = nonzero(self.alpha)[0]
        self.SV = self.x[self.SVIndex]
        self.SVAlpha = self.alpha[self.SVIndex]
        self.SVLabel = self.label[self.SVIndex]
        #self.x = None
        self.K = None
        #self.label = None
        #self.alpha = None
        self.eCache = None
        #print(self.alpha,self.label)
     
        
#   def K(self,i,j):
#       return self.x[i,:]*self.x[j,:].T
    def kernelTrans(self,x,z):#输入x与训练样本输入的内积，核函数不同计算方式不同，本例用线性核函数
        if array(x).ndim != 1 or array(x).ndim != 1:
            raise Exception("input vector is not 1 dim")
        if self.kwargs['name'] == 'linear':
            return sum(x*z)
        elif self.kwargs['name'] == 'rbf':
            theta = self.kwargs['theta']
            return exp(sum((x-z)*(x-z))/(-1*theta**2))

    def calcw(self):#根据对偶问题的最优解计算w*
        for i in range(self.m):
            self.w += dot(self.alpha[i]*self.label[i],self.x[i,:])

        

    def predict(self,testData):#对测试集标签进行预测
        #for i in range(len(self.SVIndex)):
            #self.w1 += dot(self.SVAlpha[i]*self.SVLabel[i],self.SV[i,:])
        #print(self.w1)
        test = array(testData)
        #return (test * self.w + self.b).getA()
        result = []
        m = shape(test)[0]
        for i in range(m):
            tmp = self.b
            tmp1 = self.b
            for j in range(len(self.SVIndex)):
                tmp += self.SVAlpha[j] * self.SVLabel[j] * self.kernelTrans(self.SV[j],test[i,:])
            while tmp == 0:
                tmp = random.uniform(-1,1)
            if tmp > 0:
                tmp = 1
            else:
                tmp = -1
            
            result.append(tmp)
    
            
        return result
def plotBestfitTest(data,label,w,b):#绘制测试集散点图并画出分类超平面以观测分类效果
    import matplotlib.pyplot as plt
    n = shape(data)[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data[:,0],data[:,1],c=label)
    x = arange(3,8,0.1)
    y = ((-b-w[0]*x)/w[1])
    plt.plot(x,y)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('testDataset')
    plt.show()
    
def plotBestfitTrain(data,label,SV,SVLabel,w,b):#绘制训练集散点图，并画出分类超平面，支持向量增加红色边缘标注
    import matplotlib.pyplot as plt
    n = shape(data)[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data[:,0],data[:,1],c=label)
    x = arange(3,8,0.1)
    y = ((-b-w[0]*x)/w[1])
    plt.plot(x,y)
    plt.scatter(SV[:,0], SV[:,1],
               c=SVLabel, cmap=plt.cm.viridis, lw=1, edgecolors='r')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('trainDataset')
    plt.show()


def main():
 
    data,label = create_data()
    train, test, trainLabel, testLabel = train_test_split( data,label, test_size=0.2, random_state=42)
    #print(shape(train),shape(trainLabel),shape(test),shape(testLabel))
    smo = PlattSMO(train, trainLabel, 200, 0.0001, 10000, name='linear', theta=20)
    smo.smoP()
    print (len(smo.SVIndex))
    #test,testLabel = loadImage("digits/testDigits",maps)
    testResult = smo.predict(test)
    m = shape(test)[0]
    count  = 0.0
    for i in range(m):
        if testLabel[i] != testResult[i]:
            count += 1
    print ("classfied error rate is:",count / m)
    #print('testResult:',testResult,'testLabel',testLabel,'result1:',result1)
    #smo.kernelTrans(data,smo.SV[0])
    smo.calcw()
    print(smo.w,smo.b)
    plotBestfitTrain(train,trainLabel,smo.SV,smo.SVLabel,smo.w,smo.b)
    #plt.scatter(smo.SV[:,0], smo.SV[:,1],
            #   c=model.SVLabel, cmap=plt.cm.viridis, lw=1, edgecolors='k')
    plotBestfitTest(test,testLabel,smo.w,smo.b)
   
if __name__ == "__main__":
    main()
    #plotBestfit(data,label,smo.w,smo.b)
    #smo.calcw


# In[ ]:





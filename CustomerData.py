import numpy as np
import numpy 
import matplotlib.pyplot as plt
import sys
import pandas
import warnings

#suppress warnings
warnings.filterwarnings('ignore')



class LogisticRegression:
    
    def normalizeData(self,dataSet):
        for column in dataSet:
            if column != 'purchased':
                dataSet[column] = (dataSet[column] - dataSet[column].min()) / (
                    dataSet[column].max() - dataSet[column].min())
            
        return dataSet


    def divideData(self,dataSetNorm, featuresSel):
        dataDividion = (int((0.8) * (dataSetNorm.shape[0])))

        allInput = dataSetNorm.loc[:, dataSetNorm.columns.isin(featuresSel)]
        allOutput = dataSetNorm['purchased']

        inputTrain = allInput.iloc[0:dataDividion]
        inputTest = allInput.iloc[dataDividion:]
        outputTrain = allOutput.iloc[0:dataDividion]
        outputTest = allOutput.iloc[dataDividion:]

        return inputTrain, inputTest, outputTrain, outputTest


    def drawScatter(self,x,y):
        
        plt.scatter(x,y)

        plt.title('Tryiing Different Alpha ')

        plt.xlabel("Alpha")

        plt.ylabel("Accuracy")

        plt.show()
        
        
    def differentAlpha(self):
            
            accuracy1,theta1 =self.mainAlgorithm(0.003)
            accuracy2,theta2 =self.mainAlgorithm(0.004)
            accuracy3,theta3 =self.mainAlgorithm(0.0001)
            accuracy4,theta4 =self.mainAlgorithm(0.001)
        
            accuracyList=[accuracy1,accuracy2,accuracy3,accuracy4]
            alphaList=[0.003,0.004,0.0001,0.01]
            
            self.drawScatter(alphaList,accuracyList)
            

    def sigmouidFunc(self,y_pred):
        return 1/(1+np.e**(-1*y_pred))


    def addoneColumn (self,data):
            
            
            #data.shape[0]-->get the number of rows 
            #1-->column number
            #make array that have rows numbers and columns filling one (1).
            onesColumn = numpy.ones((data.shape[0], 1), dtype=data.dtype)
            
            #numpy.hstack() function is used to stack the sequence of input arrays horizontally (i.e. column wise) to make a single array.
            return numpy.hstack([onesColumn, data])
            

    def gradientDescent(self,xData,yData,theta,learningRate):
            
            iterations=100
            m=len(xData)
            for it in range(iterations):
                
                #convert matrix to array
                hypothesis=xData.dot(theta)
                sigmoudFun=self.sigmouidFunc(hypothesis)
                error=numpy.subtract(sigmoudFun,yData)
                derivationPart= (learningRate)*xData.transpose().dot(error)
                theta=theta-derivationPart
                
            return theta
        

    def logisticRegretion(self,inputTrain,outputTrain,alpha):
            #convert the data to array 
            inputArray=inputTrain.to_numpy()
            
            #add the column in front of each row =[1,1,1,..]
            xData=self.addoneColumn (inputArray)
            yData=outputTrain.to_numpy().reshape(-1,1)
            
            #define the theta with ones 
            theta=numpy.ones((xData.shape[1], 1))
        
            return self.gradientDescent(xData,yData,theta,alpha)
        
    
    def prediction(self,xData,theta):
        
        xData=xData.to_numpy()
            
        #add the column in front of each row =[1,1,1,..]
        xData=self.addoneColumn (xData)
        
        hypothesis=xData.dot(theta)
        sigmoudFun=self.sigmouidFunc(hypothesis)
        
        resultPred = []
        
        for itr in sigmoudFun:
            if itr>0.5:
                resultPred.append(1)
            else:
                resultPred.append(0)
       
        return resultPred


    def accuracy(self,yactualData,resultPred):
        
        dataLen=(len(yactualData))
        equalCount=np.sum(yactualData == resultPred)
        accuracy=(equalCount/dataLen)*100
        return accuracy


    def mainAlgorithm(self,alpha):
        
        # load data set
        orignialData = pandas.read_csv("customer_data.csv")
        featuresSel=['age','salary']
        # NORMALIZE THE DATA
        dataSetNorm = self.normalizeData(orignialData.copy())

        # Let us see how to shuffle the rows of a DataFrame
        dataSetNorm = dataSetNorm.sample(frac=1)
        
        inputTrain, inputTest, outputTrain, outputTest = self.divideData(dataSetNorm, featuresSel)
        
        # logistic Regertion
        
        theta=self.logisticRegretion(inputTrain,outputTrain,alpha)
        resultPred=self.prediction(inputTest,theta)
        accuracy=self.accuracy(outputTest,resultPred)
        
        return accuracy,theta
        
        

def main(args):
    LogisticModel = LogisticRegression()
    LogisticModel.differentAlpha()
    
    #The Best Alpha
    accuracy,theta=LogisticModel.mainAlgorithm(0.0001)
    print("The Accuracy is  : \n",accuracy)
    print("The Coefficient : \n",theta)
    
    
if __name__ == '__main__':
    main(sys.argv)
    
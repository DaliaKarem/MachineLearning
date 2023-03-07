
import pandas
import sns
import sys
import numpy 
import numpy as np
import matplotlib.pyplot as plt
import seaborn



class LinearRegretion:
    
    
    def differentAlpha(self,inputTrain,outputTrain):
        
            theta1, cost1 =self.linearRegretion(inputTrain,outputTrain,0.003)
            theta2, cost2 =self.linearRegretion(inputTrain,outputTrain,0.004)
            theta3, cost3 =self.linearRegretion(inputTrain,outputTrain,0.001)
            theta4, cost4 =self.linearRegretion(inputTrain,outputTrain,0.01)
        
            plt.plot(np.arange(len(cost1)),cost1, color ='black', label = 'alpha = 0.003')
            plt.plot(np.arange(len(cost2)),cost2, color ='red', label = 'alpha = 0.004')
            plt.plot(np.arange(len(cost3)),cost3, color ='blue', label = 'alpha = 0.001')
            plt.plot(np.arange(len(cost4)),cost4, color ='yellow', label = 'alpha = 0.01')
            

            plt.rcParams["figure.figsize"] = (10,6)
            plt.xlabel("Number of iterations")
            plt.ylabel("Cost")
            plt.title("Different Alpha")
            plt.legend()
            plt.show()
            
            
    def __init__(self,filePath):
        self.orignialData= pandas.read_csv(filePath)
        

    def drawScatter(self,x,y):
    
        dataSet=self.orignialData
        plt.scatter(dataSet[x],dataSet[y] )

        plt.title('Scatter Plot Of Selected Feature')

        plt.xlabel(x)

        plt.ylabel(y)

        plt.show()

    def selectFeatures(self):
    
        dataSet=self.orignialData
        #When make Correlation find the 
        #enginesize,curbweight,housepower,highwaympg is high correlative with price
        cor=dataSet.corr(numeric_only=True)
        plt.figure(figsize=(15,10))
        seaborn.heatmap(cor,annot=True,cmap=plt.cm.CMRmap_r)
        plt.show()

        featuersSel=['enginesize','highwaympg','curbweight','horsepower']
        
        
        for i in range(len(featuersSel)):
            self.drawScatter(featuersSel[i],'price')
        return featuersSel
            
            
    #Normaliztaion Function
    def normalizeData(self):
        
        dataSet=self.orignialData.copy()
        for column in dataSet:
            if column!='ID' and dataSet[column].dtypes !='object':
                dataSet[column] = (dataSet[column] - dataSet[column].min()) / (dataSet[column].max() - dataSet[column].min())  
            
        return dataSet


    def divideData(self,dataSetNorm,featuresSel):
    
        
        dataDividion=(int((0.8)*(dataSetNorm.shape[0])))
    
        allInput=dataSetNorm.loc[:, dataSetNorm.columns.isin(featuresSel)]
        allOutput=dataSetNorm['price']

        inputTrain = allInput.iloc[0:dataDividion]
        inputTest = allInput.iloc[dataDividion:]
        outputTrain = allOutput.iloc[0:dataDividion]
        outputTest= allOutput.iloc[dataDividion:]
        
        
        return inputTrain,inputTest,outputTrain,outputTest

            
    def addoneColumn (self,data):
    
        
        #data.shape[0]-->get the number of rows 
        #1-->column number
        #make array that have rows numbers and columns filling one (1).
        onesColumn = numpy.ones((data.shape[0], 1), dtype=data.dtype)
        
        #numpy.hstack() function is used to stack the sequence of input arrays horizontally (i.e. column wise) to make a single array.
        return numpy.hstack([onesColumn, data])
        

    def gradientDescent(self,xData,yData,theta,learningRate):
        
        iterations=200
        m=len(xData)
        costValues=[]
        
        for it in range(iterations):
            
            #convert matrix to array
            hypothesis=xData.dot(theta)
            
            error=numpy.subtract(hypothesis,yData)
            
        
            #get the mean square error
            jCost = (1 / 2 * m) * np.sum(error ** 2)
            costValues.append(jCost)
            print("In Iteration: ",it, "The cost is: ",jCost)
        
            #size of dervaition part lists is 5
            derivationPart= (learningRate)*(1/m)*xData.transpose().dot(error)
        
            theta=theta-derivationPart
            
        return theta,costValues
    
    
    
        def linearRegretion(inputTrain,outputTrain):
            
            #convert the data to array 
            inputArray=inputTrain.to_numpy()
            
            #add the column in front of each row =[1,1,1,..]
            xData=self.addoneColumn (inputArray)
            yData=outputTrain.to_numpy()
            
            #define the theta with ones 
            theta=numpy.ones((xData.shape[1], 1))
            
            return self.gradientDescent(xData,yData,theta,0.01)
        
        
        #linear Regertion
        theta,costValues=self.linearRegretion(inputTrain,outputTrain)
        
    def predctionCost(self,theta,xData):
        
        #convert the data to array 
        inputArray=xData.to_numpy()
        
        #add the column in front of each row =[1,1,1,..]
        xData=self.addoneColumn (inputArray)
        prediction=xData.dot(theta)
        return prediction
    
        
    def linearRegretion(self,inputTrain,outputTrain,alpha):
        #convert the data to array 
        inputArray=inputTrain.to_numpy()
        
        #add the column in front of each row =[1,1,1,..]
        xData=self.addoneColumn (inputArray)
        yData=outputTrain.to_numpy().reshape(-1,1)
        
        #define the theta with ones 
        theta=numpy.ones((xData.shape[1], 1))
    
        return self.gradientDescent(xData,yData,theta,alpha)
    
    
    def mainAlgorithm(self):
        #Feature Selection
        featuresSel=self.selectFeatures()
        
        #NORMALIZE THE DATA
        dataSetNorm=self.normalizeData()
        
        #Let us see how to shuffle the rows of a DataFrame
        dataSetNorm = dataSetNorm.sample(frac = 1)
        
        #Divide Data Trainng and Testing NumericalData (all Input,all output)//30% test 80%trainng-->205*0.8=164
        inputTrain,inputTest,outputTrain,outputTest=self.divideData(dataSetNorm,featuresSel)
        
        #Try Different Alpa
        self.differentAlpha(inputTrain,outputTrain)
        
        #Best Alpha-->linear Regertion
        theta,costValues=self.linearRegretion(inputTrain,outputTrain,0.01)
        
        print("\n\nThe Coeffient of the Linear Regregiion is : \n\n",theta)
        
        plt.plot(np.arange(len(costValues)), costValues)
        plt.xlabel("Number of iterations")
        plt.ylabel("Cost function")
        plt.title("Gradient Descent With Alpha 0.01")
        plt.show()
        
        print("\nThe Prediction Of Test Data\n")
        predictionPrice=self.predctionCost(theta,inputTest)
        print(predictionPrice)
        
        
        
def main(args):
    LinearModel = LinearRegretion('car_data.csv')
    LinearModel.mainAlgorithm()
    
    
if __name__ == '__main__':
    main(sys.argv)





import numpy as np
import pandas as pd
import pprint as pp

def loadDataFromCsv( fileName, fieldFilter = None ):

	return ( pd.read_csv( fileName ) if not fieldFilter else pd.read_csv( fileName ).loc[:,fieldFilter] )
	
def createVarMatrix( dataFrame, outputFields, inputFields ):

	return np.array( dataFrame.loc[:,outputFields] ), np.array( dataFrame.loc[:,inputFields] )
	
def calculateRegressionParams( x, y, l2Pentaly = 0.0 ):
	
	# Calculate the transpose seperately to avoid doing the calculation twice
	xt = x.transpose()
	return np.linalg.inv( xt.dot( x ) + l2Pentaly * np.eye( x.shape[-1] ) ).dot( xt.dot( y ) )

def computeFitStatistics( x, y, params ):
	
    statDict = dict()
    rssMat = y - x.dot( params )
    statDict['RSS'] = float( rssMat.transpose().dot( rssMat ) )
    statDict['RMSE'] = float( np.sqrt( statDict['RSS'] / y.shape[0] ) )
    return statDict

def calculateRegressionParamsGradientDescent( x, y, stepSize = 10 ** -12, convEpsilon = 10 ** -6, l2Penalty = 0.0, initialWeights = None ):

	weights = initialWeights if not initialWeights is None else np.zeros( (x.shape[-1], 1) )
	convFlag = False
	log = open( 'log.txt', 'w' )
	counter = 0
	
	def _computeGradient( weights ):
		return -2 * x.transpose().dot( y - x.dot( weights ) ) + 2 * l2Penalty * weights
	
	while not convFlag:
	
		log.write( 'Iteration %d weights: %s\n' %( counter, weights ) )
		log.write( 'Iteration %d gradient: %s\n' %( counter, _computeGradient( weights ) ) )
		log.write( 'Iteration %d gradient norm: %s\n' %( counter, np.linalg.norm( _computeGradient( weights ) ) ) )
		counter += 1
		
		weights = weights - stepSize * _computeGradient( weights )
		
		convFlag = ( np.linalg.norm( _computeGradient( weights ) )  < convEpsilon )
	
	log.close()
	return weights

def addIntercept( x ):

    return np.hstack( [ x, np.ones( ( x.shape[0], 1) ) ] )

def createPolynomialVariables( df, varList, degree ):

    for variable in varList:
        for n in range( degree +1 ):
            df[variable + '_' + str(n)] = df[variable] ** n

    return df

def fitRegressionModel( dataFrame, modelDict, intercept = True ):
    
    resultsDict = dict()

    # Create variable matricies 
    y, x = createVarMatrix( dataFrame, modelDict['OutputVariables'], modelDict['InputVariables'] )
    x = addIntercept(x) if intercept else x

    # Fit parameters
    params = calculateRegressionParams( x, y )
    paramNames = modelDict['InputVariables'] + ( ['Intercept'] if intercept else [] )
    resultsDict['Parameters'] = dict( (name, float( value ) ) for name, value in zip( paramNames, params ) )
    resultsDict['Parameters']['ParamArray'] = params

    # Compute fit statistics
    resultsDict['FitStatistics'] = computeFitStatistics( x, y, params )

    return resultsDict

def testRegression():

    # Learn model on the training data
    
    degree = 5
    outputVariables = ['price']
    inputVariables = [ 'sqft_living_' + str(n) for n in range(degree + 1) ]

    # Load training data
    df = loadDataFromCsv( 'kc_house_train_data.csv' )
    df = createPolynomialVariables( df, ['sqft_living'], degree )
    y, x = createVarMatrix( df, outputVariables, inputVariables )
    # x = np.hstack( [ x, np.ones( ( x.shape[0], 1) ) ] )

    # Fit model using closed form solution and gradient descent methods
    paramsExact = calculateRegressionParams( x, y )
    print 'Parameters from closed form solution: %s\n' %paramsExact

    #initialWeights = np.array([[192450.], [1.], [1.]])
    #paramsApprox = calculateRegressionParamsGradientDescent( x, y, convEpsilon = 2.5e7, stepSize = 7e-12, l2Penalty = 0.0, initialWeights = initialWeights )
    #print 'Parameters from gradient descent method: %s\n' %paramsApprox

    # Compute fit statistics
    fitStats = computeFitStatistics( x, y, paramsExact )
    print 'Fit statistics for training data with exact parameters: ' + ', '.join( [ '%s: %f' %(name, value) for name, value in fitStats.iteritems() ] ) + '\n'

    #fitStats = computeFitStatistics( x, y, paramsApprox )
    #print 'Fit statistics for training data with approximate parameters: ' + ', '.join( [ '%s: %f' %(name, value) for name, value in fitStats.iteritems() ] ) + '\n'

    # Compute fit on test data
    df = loadDataFromCsv( 'kc_house_test_data.csv' )
    df = createPolynomialVariables( df, ['sqft_living'], degree )
    y, x = createVarMatrix( df, outputVariables, inputVariables )

    fitStats = computeFitStatistics( x, y, paramsExact )
    print 'Fit statistics for test data with exact parameters: ' + ', '.join( [ '%s: %f' %(name, value) for name, value in fitStats.iteritems() ] ) + '\n'

    #fitStats = computeFitStatistics( x, y, paramsApprox )
    #print 'Fit statistics for test data with approximate parameters: ' + ', '.join( [ '%s: %f' %(name, value) for name, value in fitStats.iteritems() ] ) + '\n'

def writeObjectToFile( fileName , modelResultsDict ):

    fh = open( fileName, 'w' )
    pp.pprint( modelResultsDict, fh )
    fh.close()

if __name__ == '__main__':
    
    testRegression()

    ## Create training data variables
    #trainData = loadDataFromCsv( 'kc_house_train_Data.csv' )
    #trainData['bedrooms_squared'] = trainData['bedrooms'] * trainData['bedrooms']
    #trainData['bed_bath_rooms'] = trainData['bedrooms'] * trainData['bathrooms']
    #trainData['log_sqft_living'] = trainData['sqft_living'].apply( np.log )
    #trainData['lat_plus_long'] = trainData['lat'] + trainData['long']

    ## Fit models from the training data
    #model0 = {'InputVariables': [ 'sqft_living' ], 'OutputVariables': [ 'price' ] }
    #model0Results = fitRegressionModel( trainData, model0 ) 
    #writeObjectToFile( 'model0_train.txt', model0Results )

    #model1 = {'InputVariables': [ 'sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long' ], 'OutputVariables': [ 'price' ] }
    #model1Results = fitRegressionModel( trainData, model1 ) 
    #writeObjectToFile( 'model1_train.txt', model1Results )

    #model2 = {'InputVariables': [ 'sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long', 'bed_bath_rooms' ], 'OutputVariables': [ 'price' ] }
    #model2Results = fitRegressionModel( trainData, model2 ) 
    #writeObjectToFile( 'model2_train.txt', model2Results )

    #model3 = {'InputVariables': [ 'sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long', 'bed_bath_rooms', 'bedrooms_squared', 'log_sqft_living', 'lat_plus_long' ],
    #          'OutputVariables': [ 'price' ] }
    #model3Results = fitRegressionModel( trainData, model3 ) 
    #writeObjectToFile( 'model3_train.txt', model3Results )

    ## Create test data variables
    #testData = loadDataFromCsv( 'kc_house_test_Data.csv' )
    #testData['bedrooms_squared'] = testData['bedrooms'] * testData['bedrooms']
    #testData['bed_bath_rooms'] = testData['bedrooms'] * testData['bathrooms']
    #testData['log_sqft_living'] = testData['sqft_living'].apply( np.log )
    #testData['lat_plus_long'] = testData['lat'] + testData['long']

    ## Compute fit statistics on testing data
    #yTestModel0, xTestModel0 = createVarMatrix( testData, model0['OutputVariables'], model0['InputVariables'] )
    #writeObjectToFile( 'model0_test.txt', computeFitStatistics( addIntercept( xTestModel0 ), yTestModel0, model0Results['Parameters']['ParamArray'] ) )

    #yTestModel1, xTestModel1 = createVarMatrix( testData, model1['OutputVariables'], model1['InputVariables'] )
    #writeObjectToFile( 'model1_test.txt', computeFitStatistics( addIntercept( xTestModel1 ), yTestModel1, model1Results['Parameters']['ParamArray'] ) )

    #yTestModel2, xTestModel2 = createVarMatrix( testData, model2['OutputVariables'], model2['InputVariables'] )
    #writeObjectToFile( 'model2_test.txt', computeFitStatistics( addIntercept( xTestModel2 ), yTestModel2, model2Results['Parameters']['ParamArray'] ) )

    #yTestModel3, xTestModel3 = createVarMatrix( testData, model3['OutputVariables'], model3['InputVariables'] )
    #writeObjectToFile( 'model3_test.txt', computeFitStatistics( addIntercept( xTestModel3 ), yTestModel3, model3Results['Parameters']['ParamArray'] ) )

import numpy as np
import pandas as pd

def loadDataFromCsv( fileName, fieldFilter = None ):

	return ( pd.read_csv( fileName ) if not fieldFilter else pd.read_csv( fileName ).loc[:,fieldFilter] )
	
def createVarMatrix( dataFrame, depFields, indFields ):

	return np.array( dataFrame.loc[:,depFields] ), np.array( dataFrame.loc[:,indFields] )
	
def linearRegressionParameters( x, y, l2Pentaly = 0.0 ):
	
	# Calculate the transpose seperately to avoid doing it twice
	xt = x.transpose()
	return np.linalg.inv( xt.dot( x ) + l2Pentaly * np.eye( x.shape[-1] ) ).dot( xt.dot( y ) )

def computeRss( x, y, params ):
	
	rssMat = y - x.dot( params )
	return rssMat.transpose().dot( rssMat )

def linearRegressionGradientDescent( x, y, stepSize = 10 ** -12, convEpsilon = 10 ** -6, initialWeights = None ):

	weights = initialWeights if not initialWeights == None else np.zeros( (x.shape[-1], 1) )
	convFlag = False
	log = open( 'log.txt', 'w' )
	counter = 0
	
	def _computeGradient( weights ):
		return -2 * x.transpose().dot( y - x.dot( weights ) )
	
	while not convFlag:
	
		log.write( 'Iteration %d weights: %s\n' %( counter, weights ) )
		log.write( 'Iteration %d gradient: %s\n' %( counter, _computeGradient( weights ) ) )
		log.write( 'Iteration %d gradient norm: %s\n' %( counter, np.linalg.norm( _computeGradient( weights ) ) ) )
		counter += 1
		
		weights = weights - stepSize * _computeGradient( weights )
		
		convFlag = ( np.linalg.norm( _computeGradient( weights ) )  < convEpsilon )
	
	log.close()
	return weights
	
if __name__ == '__main__':
	
	y, x = createVarMatrix( loadDataFromCsv( 'kc_house_train_data.csv' ), ['price'], ['sqft_living'] )
	x = np.hstack( [ x, np.ones( ( x.shape[0], 1) ) ] )
	params = linearRegressionParameters( x, y )
	print params
	initialWeights = np.array([[1.],[-40000.]])
	params = linearRegressionGradientDescent( x, y, convEpsilon = 2.5e7, stepSize = 7e-12, initialWeights = initialWeights )
	print params
	# print computeRss( x, y, params )
	
	# yTest, xTest = createVarMatrix( loadDataFromCsv( 'kc_house_test_data.csv' ), ['price'], ['bedrooms'] )
	# xTest = np.hstack( [ xTest, np.ones( ( xTest.shape[0], 1) ) ] )
	# print computeRss( xTest, yTest, params )
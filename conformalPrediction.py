import math

############NonConformity measures############
'''Average Distance
Z is array of z(x,y): 'y' label and 'x' vector
noCons: the index to doesn't compute'''
def AvgDist(Z, noCons):
	tipo = Z[noCons][1]

	dimV = len(Z[0][0]) #dim of data vectors
	baryCenter = [] #baryCenter vector
	#for x,y,z,...
	for i in xrange(dimV):
		t = 0
		n = 0
		for y in xrange(len(Z)):
			if Z[y][1] != tipo:
				continue	
			t += Z[y][0][i]
			n += 1
		baryCenter.append( (float(t)/n) )

	return distanceFunction(baryCenter, Z[noCons][0])


'''Nearest Neighbors
Z is array of z(x,y): 'y' label and 'x' vector
noCons: the index to doesn't compute'''
def NN(Z, noCons):
	minEq = float('inf')
	minDis = float('inf')

	for i in xrange(len(Z)):
		if i == noCons:
			continue

		#same label
		if Z[i][1] == Z[noCons][1]:
			t = distanceFunction(Z[i][0], Z[noCons][0])
			if t < minEq:
				minEq = t
		else:
			t = distanceFunction(Z[i][0], Z[noCons][0])
			if t < minDis:
				minDis = t

	if minDis == 0:
		if minEq == 0:
			return 0
		else:
			return float('inf')

	return float(minEq)/minDis



############Distance functions############
def cosineSimilarity(vectorA, vectorB):
	
	lenV = __testLen(vectorA, vectorB)

	numerator = 0
	A2 = 0
	B2 = 0
	for i in xrange(lenV):
		numerator += vectorA[i] * vectorB[i]
		A2 += vectorA[i]*vectorA[i]
		B2 += vectorB[i]*vectorB[i]

	denominator = math.sqrt(A2) * math.sqrt(B2)

	return 1 - (float(numerator)/denominator)

def squaredDistance(vectorA, vectorB):
	
	lenV = __testLen(vectorA, vectorB)
	dist = 0
	for i in xrange(lenV):
		t = vectorA[i] - vectorB[i]
		dist += math.pow(t,2)

	return dist


def euclideanDistance(vectorA, vectorB):
	return math.sqrt(squaredDistance(vectorA, vectorB))

'''Check if vectors have the same dimension'''
def __testLen(vectorA, vectorB):
	lenA = len(vectorA)
	lenB = len(vectorB)

	if lenA != lenB:
		raise Exception("vectors dimension is different -> A:%d - B:%d"%(lenA, lenB))

	return lenA


############Algorithms############

'''Conformal Prediction
A: nonConformity function
B: training set
z: the test example
'''
def ConfPred(A, error, B, z):
	Aalpha = []
	B.append(z)
	n=len(B)
	for i in xrange(n):
		Aalpha.append(A(B, i))
	
	B.pop(n-1) #restore previus state of B

	c=0
	for i in xrange(n):
		if Aalpha[i] >= Aalpha[n-1]:
			c += 1

	pValue = float(c)/n
	if pValue > error:
		return True, pValue
	else:
		return False, pValue


'''Inductive Conformal Prediction
A: nonConformity function
B: training set
Aalpha: non-conformity scores
z: the test example'''
def IndConfPred(A, error, B, Aalpha, z):

	az = __nonconformityScore(A, B, z)

	n = len(Aalpha)
	c=0
	for i in xrange(n):
		if Aalpha[i] >= az: 
			c += 1

	pValue = float(c)/n
	if pValue > error:
		return True, pValue
	else:
		return False, pValue

def __nonconformityScore(A, B, z):
	B.append(z)
	n=len(B)
	az = A(B, n-1)
	
	B.pop(n-1) #restore previus state of B

	return az

############Main############


if __name__ == "__main__":

	sepalTraining = [] #training set
	sepalTest = [] #test set

	#read the file
	lineCounter = 1
	with open("testIris.csv", "r") as f:
		for line in f:
			t = (line.replace("\n", "").split(","))
			dataVector = []
			for i in range(4):
				dataVector.append(float(t[i]))

			if lineCounter > 60:
				sepalTest.append(dataVector)
			else:
				tipo = 0 #v=0, s=1
				if t[4] == "Iris-setosa":
					tipo = 1
				sepalTraining.append([dataVector, tipo])

			lineCounter += 1
	lineCounter -= 1 #fix the number of read lines

	#set the distance function
	distanceFunction =  cosineSimilarity #squaredDistance 
	NonConfFunction = NN #AvgDist

	#To test Conformal Prediction
	for i in xrange(20):
		v = sepalTest[i]
		canAddSetosa, pSetora = ConfPred(NonConfFunction, 0.05, sepalTraining, [v,1])
		canAddVers, pVersicolor =  ConfPred(NonConfFunction, 0.05, sepalTraining, [v,0])

		if pSetora > pVersicolor:
			print("%d -> Setosa"%(60+i+1))
			sepalTraining.append([v,1])
		else:
			print("%d -> Versicolor"%(60+i+1))
			sepalTraining.append([v,0])


	
	'''#To test the Inductive Conformal Prediction
	properSet = sepalTraining[:40]
	calibrationSet = sepalTraining[40:]

	Aalpha = []
	n = len(calibrationSet)
	for i in xrange(n):
		Aalpha.append(__nonconformityScore(NonConfFunction, properSet, calibrationSet[i]))


	for i in xrange(20):
		v = sepalTest[i]
		canAddSetosa, pSetora = IndConfPred(NonConfFunction, 0.05, properSet, Aalpha, [v,1])
		canAddVers, pVersicolor =  IndConfPred(NonConfFunction, 0.05, properSet, Aalpha, [v,0])

		if pSetora > pVersicolor:
			print("%d -> Setosa"%(60+i+1))
			Aalpha.append(__nonconformityScore(NonConfFunction, properSet, [v,1]))
		else:
			print("%d -> Versicolor"%(60+i+1))
			Aalpha.append(__nonconformityScore(NonConfFunction, properSet, [v,0]))'''


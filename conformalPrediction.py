#noCons the index to non compute
def AvgDist(B,noCons):
	z = B[noCons] #the z	
	n = len(B)-1 #B\zi
	bAvg = (sum(B)-z)/n
	return (float(n)/(n+1))*abs(bAvg-z)

#z(x,y) con y label
#noCons the index to non compute
def NN(Z, noCons):
	minEq = float('inf')
	minDis = float('inf')

	for i in xrange(len(Z)):
		if i == noCons:
			continue

		#same label
		if Z[i][1] == Z[noCons][1]:
			t = abs(Z[i][0]-Z[noCons][0])
			if t < minEq:
				minEq = t
		else:
			t = abs(Z[i][0]-Z[noCons][0])
			if t < minDis:
				minDis = t

	if minDis == 0:
		if minEq == 0:
			return 0
		else:
			return float('inf')

	return float(minEq)/minDis

#B training set
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


def __nonconformityScore(A, B, z):
	B.append(z)
	n=len(B)
	az = A(B, n-1)
	
	B.pop(n-1) #restore previus state of B

	return az

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


#v=0, s=1
sepal = [[5,1], [4.4, 1], [4.9,1], [4.4,1], [5.1,1], [5.9,0], [5, 1], [6.4, 0], [6.7,0], [6.2,0], [5.1, 1], [4.6, 1], [5,1], [5.4, 1], [5, 0], [6.7, 0], [5.8, 0], [5.5,1], [5.8,0], [5.4, 1], [5.1, 1], [5.7,0], [4.6,1], [4.6, 1]]

print ConfPred(NN, 0.05, sepal, [6.8, 0])

'''
#To test the inductive version
properSet = sepal[:15]
calibrationSet = sepal[15:]

Aalpha = []
n = len(calibrationSet)
for i in xrange(n):
	Aalpha.append(__nonconformityScore(NN, properSet, calibrationSet[i]))

print IndConfPred(NN, 0.05, properSet, Aalpha, [6.8, 0])'''

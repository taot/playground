import numpredict

data = numpredict.wineset1()

numpredict.euclidean(data[0]['input'], data[1]['input'])

def knn3(d, v): return numpredict.knnestimate(d, v, k = 3)

sdata = numpredict.rescale(data, [10, 10, 0, 0.5])

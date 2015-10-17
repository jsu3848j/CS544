from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np

square =  np.random.rand(1000,2)
alldata = ClassificationDataSet(2, 1, 2)
for coordinate in square:
    if (coordinate[0] - 0.5)**2 + (coordinate[1] - 0.5)**2 <= 0.4*0.4:
        klass = 0
    else:
        klass = 1
    alldata.addSample(coordinate, klass)

tstdata, trndata = alldata.splitWithProportion(.25)
tstdata._convertToOneOfMany()
trndata._convertToOneOfMany()

here, _ = where(trndata['class'] == 0)
there, _ = where(trndata['class'] == 1)
plt.plot(trndata['input'][here, 0], trndata['input'][here, 1], 'bo')
plt.plot(trndata['input'][there, 0], trndata['input'][there, 1], 'ro')
plt.savefig('Training.png')

fnn = buildNetwork(trndata.indim, 9, trndata.outdim, outclass=SoftmaxLayer, bias=True)
trainer = BackpropTrainer(fnn, dataset=trndata, momentum = 0.99, verbose=True, weightdecay=0.01)


trainer.trainUntilConvergence(maxEpochs=1000)
trnresult = percentError( trainer.testOnClassData(), 
                         trndata['class'] )
tstresult = percentError( trainer.testOnClassData( 
   dataset=tstdata ), tstdata['class'] )

print "epoch: %4d" % trainer.totalepochs, \
    "  train error: %5.2f%%" % trnresult, \
    "  test error: %5.2f%%" % tstresult


out = fnn.activateOnDataset(tstdata)
out = out.argmax(axis=1)

for point, test in enumerate(tstdata['class']):
    if test==out[point]:
        if test==0:
            plt.plot(tstdata['input'][point, 0], tstdata['input'][point, 1], 'bo')
        else:
            plt.plot(tstdata['input'][point, 0], tstdata['input'][point, 1], 'ro')
    else:
        plt.plot(tstdata['input'][point, 0], tstdata['input'][point, 1], 'ko')

plt.savefig('Test.png')

print "Precision:  ", precision_score(tstdata['class'], out)
print "Accuracy:  ", accuracy_score(tstdata['class'], out)
print "Recall:  ", recall_score(tstdata['class'], out)

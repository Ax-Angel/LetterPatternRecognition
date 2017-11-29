import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn import ensemble
import pylab as pl
import random

digits = load_digits()
#Define variables
n_samples = len(digits.images)
x = digits.images.reshape((n_samples, -1))
y = digits.target

#Create random indices
sample_index=random.sample(range(len(x)),len(x)/5) #20-80
valid_index=[i for i in range(len(x)) if i not in sample_index]

#Sample and validation images
sample_images=[x[i] for i in sample_index]
valid_images=[x[i] for i in valid_index]

#Sample and validation targets
sample_target=[y[i] for i in sample_index]
valid_target=[y[i] for i in valid_index]

#Using the Random Forest Classifier
classifier = ensemble.RandomForestClassifier()

#Fit model with sample data
classifier.fit(sample_images, sample_target)

#Attempt to predict validation data
score=classifier.score(valid_images, valid_target)
print 'Random Tree Classifier:\n'
print 'Score\t'+str(score)
print classifier.predict([x[i]])
i=800

pl.gray()
pl.matshow(digits.images[i])
pl.show()

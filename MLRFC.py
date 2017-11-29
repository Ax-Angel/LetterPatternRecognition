import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn import ensemble
import pylab as pl
import random

digits = load_digits()
# define variables
n_samples = len(digits.images)
x = digits.images.reshape((n_samples, -1))
y = digits.target

#crea indices aleatorios
sample_index=random.sample(range(len(x)),len(x)/5) #entre 20 y 80
valid_index=[i for i in range(len(x)) if i not in sample_index]

# imagenes muestra y validas
sample_images=[x[i] for i in sample_index]
valid_images=[x[i] for i in valid_index]

# etiqueta las muestras validas y de prueba
sample_target=[y[i] for i in sample_index]
valid_target=[y[i] for i in valid_index]

# usando random forest classifier
classifier = ensemble.RandomForestClassifier()

# ajusta el modelo a los datos muestra
classifier.fit(sample_images, sample_target)

# intenta predecir datos de validacion
score=classifier.score(valid_images, valid_target)
print 'Random Tree Classifier:\n'
print 'Puntaje\t'+str(score)
print classifier.predict([x[i]])
i=800

pl.gray()
pl.matshow(digits.images[i])
pl.show()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from sklearn.model_selection import train_test_split

class Ccomparator:
    def __init__ (self, name, ev):
        self.name = name
        self.ev = ev
    
    def __str__ (self):
        return "{} {}".format(self.name, str(self.ev["accuracy"]))
    
    def __eq__ (self, other):
        return self.ev["accuracy"] == other.ev["accuracy"]
    
    def __lt__ (self, other):
        return self.ev["accuracy"] < other.ev["accuracy"]
    

# leer archivo con los pacientes
pctes = pd.read_csv("datasets/exitosSinNulos.csv")
# lista de comparadores
comparators = []

# separo la info para hacer test y pruebas
X_train, X_test, y_train, y_test = train_test_split(pctes.iloc[::], pctes["class"], test_size=0.33, random_state=42)
columns = pctes.columns

# preparacion de datos para el modelo
feature_columns = [tf.contrib.layers.real_valued_column(k) for k in columns]
def input_fn(df,labels):
    feature_cols = {k:tf.constant(df[k].values,shape = [df[k].size,1]) for k in columns}
    label = tf.constant(labels.values, shape = [labels.size,1])
    return feature_cols,label

# itera las deep neural networks, recibe como input tuples
def test_dnn(inn, hidd, outt):
    classifier = None
    for i in range(inn[0], inn[1], inn[2]):
        for j in range(hidd[0], hidd[1], hidd[2]):
            for k in range(outt[0], outt[1], outt[2]):
                classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,hidden_units=[i,j,k],n_classes = 2)
                classifier.train(input_fn=lambda: input_fn(X_train,y_train),steps = 1000)
                ev = classifier.evaluate(input_fn=lambda: input_fn(X_test,y_test),steps=1)
                comparators.append(Ccomparator("DNN({},{},{})".format(i,j,k), ev))

# itera para los baseline classifiers
def test_baseline():
    classifier = tf.estimator.BaselineClassifier(n_classes=2)
    classifier.train(input_fn=lambda: input_fn(X_train,y_train),steps = 1000)
    ev = classifier.evaluate(input_fn=lambda: input_fn(X_test,y_test),steps=1)
    comparators.append(Ccomparator("Baseline classifier", ev))

# agrega el baseline regressor
def test_baselineRegressor():
    classifier = tf.estimator.BaselineRegressor() # no recibe nada
    classifier.train(input_fn=lambda: input_fn(X_train,y_train),steps = 1000)
    ev = classifier.evaluate(input_fn=lambda: input_fn(X_test,y_test),steps=1)
    print ev
    comparators.append(Ccomparator("Baseline regressor", ev))

# itera boostedTreesClassifier, hay que segmentar las columnas, sin terminar
def test_boostedTrees(n_trees, batches):
    classifier = tf.estimator.BoostedTreesClassifier(feature_columns=feature_columns, n_trees=n_trees, n_batches_per_layer= batches, n_classes=2)
    classifier.train(input_fn=lambda: input_fn(X_train,y_train),steps = 1000)
    ev = classifier.evaluate(input_fn=lambda: input_fn(X_test,y_test),steps=1)
    print ev
    comparators.append(Ccomparator("Boosted Tree (n_tree={}, learning_rate={})".format(n_trees, batches), ev))


# imprime la lista ordenada con los comparators
def print_comparators ():
    comparators.sort()
    for ccomp in comparators:
        print ccomp

# imprime la lista de comparadores a un archivo
def print_comparators_file (path):
    file = open(path, "w+")
    comparators.sort()
    for ccomp in comparators:
        file.write(str(ccomp) + "\n")
    file.close()

test_dnn((10,40,10), (10,50,10), (10,40,10))
test_baseline()
#test_baselineRegressor()
#test_boostedTrees(100, 4)
print_comparators_file("accuracy.txt")


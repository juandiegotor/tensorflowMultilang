import tensorflow as tf
import numpy as np
import pandas as pd
from pandas import get_dummies
from sklearn.model_selection import train_test_split
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from keras import backend as K

# mismo proceso que el aplicado en el selector
import os
import os.path as path

# Clase para exportar modelo a unity
def export_model(saver, input_node_names, output_node_name):
    # crea el folder out y escribe los archivos de salidda
    if not path.exists('out'):
        os.mkdir('out')

    # an arbitrary name for our graph
    GRAPH_NAME = 'grafopp'

    # guardar el grafo en formato pb
    tf.train.write_graph(K.get_session().graph_def, 'out', GRAPH_NAME + '_graph.pbtxt')

    # guarda el grafo en formato de checkpoint
    saver.save(K.get_session(), 'out/' + GRAPH_NAME + '.chkp')

    # congelat el grafo, toma el grafo escrito en formato pb y lo escribe en bytes, formato que lee unity
    freeze_graph.freeze_graph('out/' + GRAPH_NAME + '_graph.pbtxt', None, False,
                              'out/' + GRAPH_NAME + '.chkp', output_node_name,
                              "save/restore_all", "save/Const:0",
                              'out/frozen_' + GRAPH_NAME + '.bytes', True, "")

    # optimiza el grafo en .bytes
    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + GRAPH_NAME + '.bytes', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + GRAPH_NAME + '.bytes', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("grafo guardado")

# cargo el archivo
data = pd.read_csv('datasets/exitosSinNulos.csv')

# separo las clases de las propiedades
cols = data.columns
features = cols[0:22]
labels = cols[22]

data_norm = pd.DataFrame(data)

# randomizar los datos para entrenamiento y pruebas
indices = data_norm.index.tolist()
indices = np.array(indices)
np.random.shuffle(indices)
X = data_norm.reindex(indices)[features]
y = data_norm.reindex(indices)[labels]
y = get_dummies(y)

# generar los areglos con test y validacion
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.3)

# formar los datos
X_train = np.array(X_train).astype(np.float32)
X_test  = np.array(X_test).astype(np.float32)
y_train = np.array(y_train).astype(np.float32)
y_test  = np.array(y_test).astype(np.float32)

# tomar la forma de los tensores
training_size = X_train.shape[1]
test_size = X_test.shape[1]
num_features = 22
num_labels = 2
num_hidden = 100

# crear el modelo y entrenarolp
graph = tf.Graph()
with graph.as_default():
    tf_train_set    = tf.constant(X_train)
    tf_train_labels = tf.constant(y_train)
    tf_valid_set    = tf.constant(X_test)

    weights_1 = tf.Variable(tf.truncated_normal([num_features, num_hidden], name="input_node"))
    weights_2 = tf.Variable(tf.truncated_normal([num_hidden, num_labels]))

    bias_1 = tf.Variable(tf.zeros([num_hidden]))
    bias_2 = tf.Variable(tf.zeros([num_labels]))

    logits_1 = tf.matmul(tf_train_set , weights_1 ) + bias_1
    rel_1 = tf.nn.relu(logits_1) # nombre del nodo que leera unity
    logits_2 = tf.matmul(rel_1, weights_2) + bias_2

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=logits_2, name="output_node"))
    optimizer = tf.train.GradientDescentOptimizer(.005).minimize(loss)

    predict_train = tf.nn.softmax(logits_2)

    logits_1_val = tf.matmul(tf_valid_set, weights_1) + bias_1
    rel_1_val    = tf.nn.relu(logits_1_val)
    logits_2_val = tf.matmul(rel_1_val, weights_2) + bias_2
    predict_valid = tf.nn.softmax(logits_2_val)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


num_steps = 10000
with tf.Session(graph = graph) as session:
    tf.initialize_all_variables().run()
    print(loss.eval())
    for step in range(num_steps):
        _,l, predictions = session.run([optimizer, loss, predict_train])
        
        if (step % 2000 == 0):
              print('Loss at step %d: %f' % (step, l))
              print('Training accuracy: %.1f%%' % accuracy( predictions, y_train[:, :]))
              print('Validation accuracy: %.1f%%' % accuracy(predict_valid.eval(), y_test))
    

    export_model(tf.train.Saver(), ["input_node"], "output_node")
    # escribo para verlo en tensorboard
    writer = tf.summary.FileWriter(logdir="out/tb", graph=graph)
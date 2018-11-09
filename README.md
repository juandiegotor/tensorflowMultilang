# El problema
Se busca encontrar el mejor clasificador para encontrar la mejor forma de ventilacion para un paciente dado. Se cuneta con una base de datos que contiene informacion de pacientes que fueron conectados a un ventilador, entre ellos datos demograficos, datos de la mecanica venilatoria y modo de ventilacion

## Preprcesamiento
Se filtraron las columnas con nulos y se seleccionaron solo los casos exitosos para entrenar los clasificadores

# Instrucciones
## Python
### Dependencias
Lo primero es instalar las librerias requeridas, si se desea se puede instalr luego tensorflow GPU, ver [docs de tensorflow](https://www.tensorflow.org/install/).

```
pip install -r requirements.txt
```

### Codigo
**selector.py**
El codigo itera entre dnn, clasificadores baseline, regresores y arboles, para encontrar el mejor clasificador.

Hasta ahora el mejor resultado que se ha encontrado es una dnn[20, 20, 30].

Los log de salida se encuentran en la carpeta output

```
python python/selector.py
```

**freeze.py**
Congela un grafo para que se pueda ver en tensorboard y exportarlo a unity. El modelo a exportar se encuentra en la carpeta out y el modelo de tensorboard en out/tb

```
python python/freez.py
tensorboard --logdir=./out/tb
```

Aqui se puede observar la arquitectura de la red neuronal a implementar, el codigo entre mas se ejecute mejor resultados dara, ya que el retoma el grafo congelado para el modelo.

## Java
Actualmente java no soporta crear modelos de tensores, pero si puede
interacruar con modelos ya creados.

Para poder usar java se debe de instalar el jdk y maven
``` shell
# instalar java desde oracle
sudo add-apt-repository ppa:webupd8team/java
sudo apt update
sudo apt updatesudo apt-get install oracle-java8-installer

# instalar maven
sudo apt install maven
```
Es necesario un jdk sperior al 1.7.

Ahora para crear un proyecto con java y maven se debe de agregar la dependencia de tensorflow al pom.xml del proyecto

``` xml
<project>
  <modelVersion>4.0.0</modelVersion>
  <groupId>org.myorg</groupId>
  <artifactId>hellotensorflow</artifactId>
  <version>1.0-SNAPSHOT</version>
  <properties>
    <exec.mainClass>HelloTensorFlow</exec.mainClass>
    <!-- The sample code requires at least JDK 1.7. -->
    <!-- The maven compiler plugin defaults to a lower version -->
    <maven.compiler.source>1.7</maven.compiler.source>
    <maven.compiler.target>1.7</maven.compiler.target>
  </properties>
  <dependencies>
    <dependency>
      <groupId>org.tensorflow</groupId>
      <artifactId>tensorflow</artifactId>
      <version>1.12.0</version>
    </dependency>
  </dependencies>
</project>
```

luego de esto maven resulve la dependencia de manera automatica
``` java
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;

public class HelloTensorFlow {
  public static void main(String[] args) throws Exception {
    try (Graph g = new Graph()) {
      final String value = "Hello from " + TensorFlow.version();

      // Construct the computation graph with a single operation, a constant
      // named "MyConst" with a value "value".
      try (Tensor t = Tensor.create(value.getBytes("UTF-8"))) {
        // The Java API doesn't yet include convenience functions for adding operations.
        g.opBuilder("Const", "MyConst").setAttr("dtype", t.dataType()).setAttr("value", t).build();
      }

      // Execute the "MyConst" operation in a Session.
      try (Session s = new Session(g);
          // Generally, there may be multiple output tensors,
          // all of them must be closed to prevent resource leaks.
          Tensor output = s.runner().fetch("MyConst").run().get(0)) {
        System.out.println(new String(output.bytesValue(), "UTF-8"));
      }
    }
  }
}
```
## C#, .Net y Unity
### requerimientos
- [Unity](https://unity3d.com)
- [Anaconda3](https://www.anaconda.com/)
- [plugin de TFSharp de Unity](https://s3.amazonaws.com/unity-ml-agents/0.5/TFSharpPlugin.unitypackage)
- [Repositorio de ml-agents](https://github.com/Unity-Technologies/ml-agents)

### configuracion del entorno
Se debe de crear un entorno de python 3.6 en Anaconda
```
conda create -n ml-agents python=3.6
```
luego activarlo e instalar los requerimientos
```
conda create -n ml-agents python=3.6
pip install tensorflow==1.7.1
# ir al directorio de ml-agents
pip install .
```
se debe de mantener este entorno activo cuando se desee usar tensorflow

## proyectos en Unity3D
Se debe de crear un proyecto nuevo de unity, luego en build settengs (CRTL + SHIFT + B) -> player settings -> other settings

- poner scripting version en .net 4.x
- en flags adicionales agregar
```
ENABLE_TENSORFLOW
```
- luego buscar el paquete de TFSharp y agregarlo al proyecto'
- disfrutar!

ahora par poder crear scripts que usen tensorflow agregar
``` C#
using TensorFlow
```
## javascript

Para correr tensorflow se debe poner la siguiente linea de codigo en el head del html en el que se quiera usar:

<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@0.13.3/dist/tf.min.js"> </script>

Luego de importar la libreria anterior se crea un nuevo script donde se trabajara.

Lo primero que se debe hacer es crear el modelo

const model=tf.sequential();

Para clasificar en js se debe crear cada capa que se quiera usar y agregarlar al modelo, ademas la primera capa debe tener una variable llamada inputShape que son las entradas y la ultima capa debe tener un numero de neuronas igual al numero de salidas:

const layer=tf.layers.dense({
    units:nroNeuronas,
    inputShape: datosEntrada, //Solo si es primera capa
    activation:funcionActivacion
});

model.add(layer);

Tras lo anterior de debe compilar el modelo y luego entrenarlo, las funciones para hacer esto necesitan una configuracion, en el caso de compilar es crear un optimizer, un loss y metrics, para entrenar el modelo ademas se le debe proveer las x de entrenamiento y las y de entrenamiento::

const config={
    optimizer: tf.train.sgd(0.1), //learning rate
    loss: tf.losses.meanSquaredError, //error
    metrics: ['accuracy'],
}
model.compile(config);

Para entrenar la configuracion debe tener los epochs, es decir las veces que se pasaran los datos:

const configModel={
    epochs:100,
}

const h=await model.fit(xtrain,ytrain,configModel);

por ultimo se hace una evaluacion, con las x de prueba y las y de prueba:

const x=model.evaluate(xtest,ytest);

Como js es un lenguaje front end, el corre los modelos de forma asincrona y nos devuelve una promesa, por lo cual hay que buscar una manera de trabajar con las promises de js, mostraremos una forma de hacerlo pero no es la unica:

En nuestro caso creamos una funcion asincrona, en ella entrenamos y evaluamos el modelo, y luego la corremos aplicandole una funcion then():

async function train(){
        const h=await model.fit(xtrain,ytrain,configModel);
        const x=model.evaluate(xtrain,ytrain);
        console.log('Neuronas: '+neuronas+'\nFuncion de Activavion: '+activation+'\nError y precision:'+x.toString());
}
train().then(()=>console.log('Training complete'));

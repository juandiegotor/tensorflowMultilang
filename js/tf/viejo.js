var url = "exitosSinNulos.csv";

var request = new XMLHttpRequest();
request.open("GET", url, false);
request.send(null);

var datos = new Array();
var clases = new Array();
var jsonObject = request.responseText.split(/\r?\n|\r/);
var columnas;
var x;
for (var i = 0; i < jsonObject.length; i++) {
    if (i==0) {
        columnas=jsonObject[i].split(',');
    }else{
        x=jsonObject[i].split(',');
        datos.push(x);
        clases.push(x[x.length-1]);

        datos[datos.length-1].splice(datos[datos.length-1].length-1,1);
    }

}
// Retrived data from csv file content
datos.splice(datos.length-1,1);
clases.splice(clases.length-1,1);
const xtrain=tf.tensor(datos);
const ytrain=tf.tensor(clases);
// console.log(clases);


//activacion recomendadas: sigmoid, softmax

// //Creo el modelo para softmax
const model=tf.sequential();

// //Creo la capa oculta
const hidden1=tf.layers.dense({
    units:40,
    inputShape: [22],
    activation:'softmax'
});

model.add(hidden1);
//creo la capa de salida
const output=tf.layers.dense({
    units:1,
    activation:'softmax'
});
model.add(output);
//Compilo el modelo
const config={
    optimizer: tf.train.sgd(0.1), //learning rate
    loss: tf.losses.meanSquaredError, //error
    metrics: ['accuracy'],
}
model.compile(config);

const configModel={
    epochs:100,
}

async function train(){
    const h=await model.fit(xtrain,ytrain,configModel);
    const x=model.evaluate(xtrain,ytrain);
    console.log('Neuronas: '+20+'\nFuncion de Activavion: '+'softmax'+'\nError y precision:'+x.toString());
}
train().then(()=>console.log('Training complete'));



// //Creo el modelo para sigmoid
const model2=tf.sequential();

// //Creo la capa oculta
const hidden2=tf.layers.dense({
    units:40,
    inputShape: [22],
    activation:'sigmoid'
});

model2.add(hidden2);
//creo la capa de salida
const output2=tf.layers.dense({
    units:1,
    activation:'sigmoid'
});
model2.add(output2);
//Compilo el modelo
const config2={
    optimizer: tf.train.sgd(0.1), //learning rate
    loss: tf.losses.meanSquaredError, //error
    metrics: ['accuracy'],
}
model2.compile(config2);

const configModel2={
    epochs:100,
}

async function train2(){
    const h=await model2.fit(xtrain,ytrain,configModel2);
    const x=model2.evaluate(xtrain,ytrain);
    console.log('Neuronas: '+20+'\nFuncion de Activavion: '+'softmax'+'\nError y precision:'+x.toString());
}
train2().then(()=>console.log('Training complete'));

// const hidden2=tf.layers.dense({
//     units:20,
//     activation:'sigmoid'
// });

// model.add(hidden2);






// train().then(()=>{
//     console.log('Entenamiento terminado');
//     let ynew = model.predict(xtrain);
//     // ynew.print();
//     const x=model.evaluate(xtrain,ytrain);
//     console.log(x.toString());
//     // const out = tf.math.confusionMatrix(ytrain, ynew, 2);
//     // out.print();
// });

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
function modelo(activation,neuronas){
    const model=tf.sequential();

    // //Creo la capa oculta
    const hidden1=tf.layers.dense({
        units:neuronas,
        inputShape: [22],
        activation:activation
    });

    model.add(hidden1);
    //creo la capa de salida
    const output=tf.layers.dense({
        units:1,
        activation:activation
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
        console.log('Neuronas: '+neuronas+'\nFuncion de Activavion: '+activation+'\nError y precision:'+x.toString());
    }
    train().then(()=>console.log('Training complete'));
}
const acts=['softmax','sigmoid'];

for (var i = 1;i<= 8; i++) {
    neuronas=i*5;
    for (var j = 0; j <= 1; j++) {
        modelo(acts[j],neuronas)
    }
}

// modelo('sigmoid',40);
// modelo('softmax',30);




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

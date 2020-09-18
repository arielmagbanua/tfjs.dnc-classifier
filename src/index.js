import "./styles.css";
import * as tf from "@tensorflow/tfjs/";
import * as DogsNCats from "dogs-n-cats";

document.getElementById("app").innerHTML = `
<h1>It's Training Dogs n Cats</h1>
<button id='load'>Load Dogs n Cats</button>
<button id='create'>Create Model</button>
<button id='train'>Train Model</button>
<button id='test'>Test Model</button>
<button id='dispose'>Dispose Model</button>
`;

let dnc, model;

document.getElementById("load").onclick = async () => {
  console.log("Loading Dogs N Cats Data");
  dnc = await DogsNCats.load();
  console.log("Done Loading Dogs N Cats Data");
};

document.getElementById("create").onclick = async () => {
  console.log("Creating our Model");
  model = tf.sequential();

  // first batch
  model.add(
    tf.layers.conv2d({
      inputShape: [32, 32, 3],
      kernelSize: 3,
      padding: "same",
      filters: 32,
      strides: 1,
      activation: "relu",
      kernelInitializer: "heNormal"
    })
  );

  model.add(
    tf.layers.maxPooling2d({
      poolSize: [2, 2],
      strides: [2, 2]
    })
  );

  model.add(tf.layers.batchNormalization());

  model.add(
    tf.layers.dropout({
      rate: 0.25
    })
  );

  // second batch
  model.add(
    tf.layers.conv2d({
      kernelSize: 3,
      padding: "same",
      filters: 64,
      strides: 1,
      activation: "relu",
      kernelInitializer: "heNormal"
    })
  );

  model.add(
    tf.layers.maxPooling2d({
      poolSize: [2, 2],
      strides: [2, 2]
    })
  );

  model.add(tf.layers.batchNormalization());

  model.add(
    tf.layers.dropout({
      rate: 0.25
    })
  );

  model.add(tf.layers.flatten());

  model.add(
    tf.layers.dense({
      units: 64,
      activation: "relu",
      kernelInitializer: "heNormal"
    })
  );

  model.add(
    tf.layers.dense({
      units: 1,
      activation: "sigmoid"
    })
  );

  model.compile({
    optimizer: "adam", // or use tf.train.adam(); to adjust
    loss: "binaryCrossentropy",
    metrics: ["accuracy"]
  });

  console.log("Model created");
};

document.getElementById("train").onclick = async () => {
  console.log("Gathering training data");
  const [trainX, trainY] = dnc.training.get(1600);
  const [testX, testY] = dnc.test.get(400);

  // set up the callbacks
  const printCallback = {
    onEpochEnd: (epoch, log) => {
      console.log(epoch, log);
    }
  };

  console.log("Training has started!!!");
  const history = await model.fit(trainX, trainY, {
    batchSize: 128,
    validationData: [testX, testY],
    epochs: 20,
    shuffle: true,
    callbacks: printCallback
  });

  console.log("Training Complete");
  console.log("Training History", history);
  console.log("Cleaning up training/testing data");
  tf.dispose(trainX, trainY, testX, testY);
};

document.getElementById("test").onclick = async () => {
  console.log("Testing Model");
  tf.tidy(() => {
    const [someDogs] = dnc.dogs.get(15);
    const [someCats] = dnc.cats.get(15);

    const dogChecks = model.predict(someDogs);
    console.log("Dogs", dogChecks.dataSync());
    const catChecks = model.predict(someCats);
    console.log("Cats", catChecks.dataSync());
  });
};

document.getElementById("dispose").onclick = async () => {
  console.log("Cleaning up!");
  model.dispose();
};

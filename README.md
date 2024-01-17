**<h1>fksainetwork</h1>**

[<img alt="github" src="https://img.shields.io/badge/github-Felix1G/fksainetwork-8da0cb?style=for-the-badge&labelColor=555555&logo=github" height="20">](https://github.com/Felix1G/fksainetwork)
[<img alt="crates.io" src="https://img.shields.io/crates/v/fksainetwork.svg?style=for-the-badge&color=fc8d62&logo=rust" height="20">](https://crates.io/crates/fksainetwork)
[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-fksainetwork-66c2a5?style=for-the-badge&labelColor=555555&logo=docs.rs" height="20">](https://docs.rs/fksainetwork)


A **neural network** capable of **learning** and calculating output.<br/>
This project is just made for fun :)<br/>

---
Please go to [Inner Workings](https://github.com/Felix1G/fksainetwork/blob/main/INNERWORKINGS.md) to read more.

---
Usage
---
``````toml
[dependencies]
fksainetwork = "0.2.0"
``````

---
Example (Feed Forward)
---
``````rust
let mut network = Network::new(2, &[ //2 neuron inputs
            (10, Initialization::He, Activation::Sigmoid, false), //10 hidden neurons
            (2, Initialization::Xavier, Activation::LeakyReLU, true) //2 neuron outputs, true: batch normalisation
        ], Loss::BinaryCrossEntropy, true);
//or: let network = load_network("path/to/network-file");

//calculating
let input = vec![1.0, 1.0];
let output = network.calculate(&input); //calculate
println!("{:?}", output);

//learning
//batch size of 2
network.learn(0.01,
  &vec![vec![0.0, 1.0], vec![0.0, 3.0]], //input values, batch size of 2
  &vec![vec![1.0, 0.0], vec![0.0, 1.0]] //expected values
);

//NOTE: if you call 'learn', u do not need to call 'calculate' beforehand

//save
save_network("path/network-file", &network);
``````

---
Example (Convolutional)
---
``````rust
let network = ConvolutionalNetwork::new(
            //convolution layers
            &[
                (2, &[Initialization::Xavier;20], Activation::ReLU, 2, Pooling::Max), //20 channels, kernel 2x2, pooling max 2.0
                (3, &[Initialization::Xavier;40], Activation::ReLU, 2, Pooling::Max) //40 channels, kernel 3x3, pooling max 2.0
            ],
            13, 13, 1, //input size of w: 13, h: 13, channels: 1
            //input similar to the Feed Forward Network
            &[
                (3, Initialization::Xavier, Activation::LeakyReLU, false)
            ],
            Loss::BinaryCrossEntropy, true
        );
//or: let network = load_cnn_network("path/to/network-file");

//pretend these samples are actual images of something
let sample0 = Matrix::new(13, 13);
let sample1 = Matrix::new(13, 13);
let sample2 = Matrix::new(13, 13);

//calculate
let output = network.calculate(&vec![&sample0, &sample1, &sample2]);
println!("{:?}", output);

//learn
network.learn(0.04, &vec![&sample0, &sample1, &sample2], &vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);

//print outputs
cnn_network_bmp("path/to/directory", &network);

//save network
save_cnn_network("path/to/network-file", &network);
``````

---
Update Patches :)
---
0.1.3: Added Tanh Activation Function.<br/>
0.1.3: Improved Docs.<br/>
0.2.0: The save_network and load_network functions now use str instead of String for the path.<br/>
0.2.0: Improved learning algorithm.<br/>
0.2.0: Added the Convolutional Neural Network.<br/>
0.2.0: Added more Loss functions and Initialization functions.<br/>
0.2.0: Changed learn_bpg_mse to learn as the loss function is now a function parameter.<br/>
0.2.0: Changes to the feed forward network "new()" function parameter.<br/>

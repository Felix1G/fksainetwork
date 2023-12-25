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
let layers = [2, 2]; //2 input neurons, 2 output neurons
let initializations = [Initialization::He]; //He initialization
let activations = [Activation::Sigmoid]; //sigmoid for the 2nd layer as input layers don't need an activation

let mut network = Network::new(&layers, &initializations, &activations, Loss::BinaryCrossEntropy);
//or: let network = load_network("path/to/network-file");

//calculating
let input = [1.0, 1.0];
let output = network.calculate(&input);
println!("{:?}", output);

//learning
let expected = [0.0, 1.0];
network.learn(0.01, &expected);

//save
save_network(String::from("path/network-file"), &network);
``````

---
Example (Convolutional)
---
``````rust
//to be continued
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

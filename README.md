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
Example
---
``````rust
let layers = [2, 2]; //2 input neurons, 2 output neurons
let activations = [0, 1]; //linear, sigmoid

let mut network = Network::new(&layers, &activations);
//or: let network = load_network(String::from("path/network-file"));

//calculating
let input = [1.0, 1.0];
let output = network.calculate(&input);
println!("{:?}", output);

//learning
let expected = [0.0, 1.0];
network.learn_bpg_mse(0.01, &expected);

//save
save_network(String::from("path/network-file"), &network);
``````
---
Update Patches :)
---
0.1.3: Added Tanh Activation Function.<br/>
0.1.3: Improved Docs.<br/>
0.2.0: The save_network and load_network functions now use str instead of String for the path.
0.2.0: Improved learning algorithm.
0.2.0: Added the Convolutional Neural Network.

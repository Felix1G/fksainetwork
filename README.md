**<h1>fksainetwork</h1>**

[<img alt="github" src="https://img.shields.io/badge/github-Felix1G/fksainetwork-8da0cb?style=for-the-badge&labelColor=555555&logo=github" height="20">](https://github.com/Felix1G/fksainetwork)
[<img alt="crates.io" src="https://img.shields.io/crates/v/fksainetwork.svg?style=for-the-badge&color=fc8d62&logo=rust" height="20">](https://crates.io/crates/fksainetwork)
[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-fksainetwork-66c2a5?style=for-the-badge&labelColor=555555&logo=docs.rs" height="20">](https://docs.rs/fksainetwork)


A **neural network** capable of **learning** and calculating output.<br/>
This project is just made for fun :)<br/>
Read on to learn more about its inner workings.


**<h3>Inner Workings<h3/>**

The neural network is a network of artificial neurons where each neuron does some calculations and calculates
the output according to the input. For each neuron, the neuron of the previous layer's value is taken and
multiplied by the weight associated with it stored in this neuron. Then, a bias of this neuron is added and an activation function provided for each layer is run.

$`$ a_n = g_n(z_n) $`$
$`$ z_n = b_n + \sum_n w_na_{(n-1)i}$`$
$`a_n = `$ the neuron's result<br/>
$`g_n = `$ the neuron's activation function<br/>
$`z_n = `$ the neuron's value <br/>
$`b_n = `$ the neuron's bias <br/>
$`w_n = `$ the neuron's weight associated with $`a_{n-1}`$ <br/>
$`a_{(n-1)i} = `$ the previous neuron's value <br/>

---

The current activation functions [ $`g(v)`$ ] are as follows:
- Linear (index = 0)<br/>
	$`g(v) = v`$<br/>
	$`g'(v) = 1`$
- Sigmoid (index = 1)<br/>
	$`g(v) = \frac{v}{1 + e^{-v}}`$<br/>
	$`g'(v) = g(v) * (1 - g(v))`$
- ReLU (index = 2)<br/>
	$`g(v) = max(0, v)`$<br/>
	$`
	g'(v) =
	\begin{cases}
	0.0,  & \text{if } v \leq 0 \\
	1.0, & \text{if } v > 0
	\end{cases}
	`$

---

Learning involves changing the weights and biases of the neuron to an optimal value that can evaluate precise outputs.
The current methods of learning are as follows:
- Mean Squared Error (MSE) using Back Propagation (BPG)   ***learn_bpg_mse***<br/>
	Gradient descent is used to calculate the "right" values of the weights and biases.
	A gradient of 0 ($`m = 0`$) is what the gradients of the weights and biases should approach. <br/>
 	For each neuron, an error term will be calculated which will be used later to calculate the gradient.

	To calculate the bias:
	$`$ b_n = b_n - α \times ε_n$`$
	$`b_n =`$ the neuron's bias.<br/>
	$`α =`$ the learning rate.<br/>
	$`ε_n =`$ the neuron's error term, also the gradient of the bias.

	To calculate the weight:
	$`$ w_n = w_n - α \times ε_na_{(n-1)i}$`$
	$`w_n =`$ the neuron's weight associated with $`ε_na_{(n-1)i}`$.<br/>
	$`α =`$ the learning rate.<br/>
	$`ε_n =`$ the neuron's error term, also the gradient of the bias.<br/>
	$`a_{(n-1)i} = `$ the result of the neuron of the previous layer.

	Error terms are different for the output layer and the hidden layers.

	---
  	The Output Layer
  	---
  	The output neuron's error term is as follows.
	$`$ ε_k = \frac{∂E}{∂b} =  \sum_k^n (a_k - t_k)(g'_k(zk)) $`$<br/>
	$`t_k =`$ expected value of the output neuron.<br/>
	$`a_k =`$ the neuron's result.<br/>
	$`g_k'(v) = `$ activation derived function.<br/>
	$`z_k =`$ the neuron's value.<br/>
 
	<br/>
	
	The weight's gradient is just the error term multiplied by the result.
	$`$\frac{∂E}{∂w_j} = ε_k \times a_j$`$<br/>
	$`ε_k = `$ the error term of the output neuron.<br/>
	$`w_j = `$ the weight of which its gradient is to be calculated.<br/>
	$`a_j = `$ the result of the previous neuron associated with $`w_j`$.
	
	---
  	Hidden Layer
  	---
	Equation of the hidden layer bias is the utilization of a recursion where
	all neurons that are connected with this hidden neuron are taken into account
	during the calculation of its gradient.
	<br/>
	<br/>
	An error term of j is then created.
 	$`$ ε_j = \frac{∂E}{∂b_j} =  g'_j(zj) \times \sum_k^n ε_k w_{jk} $`$<br/>
	$`g'_j =`$ activation derived function.<br/>
	$`z_j =`$ this neuron's value.<br/>
	$`ε_k =`$ error term of the output neuron.<br/>
	$`w_{jk} =`$ the weight of the next neuron associated with this hidden neuron.
	
	Hence, the weight gradient can be calculated with the same way as above.
	$`$\frac{∂E}{∂w_{ij}} = ε_j * a_i$`$
	wij = the weight of which its gradient is to be calculated;
	ai = the result of the previous neuron;
	εj = the error term of this hidden layer;
	
	------------------
	
	For deeper neural networks, the error term is then plugged in by recursion.
	εj = [ gj'(zj) ][ Σ (εk * wjk) ];
	εi = [ gj'(zi) ][ Σ (εj * wij) ];
	εh = [ gj'(zh) ][ Σ (εi * whi) ];
	and so on.

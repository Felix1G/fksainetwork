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
Initialization Functions
---

Initialization functions rely on a standard deviation [ $`𝜎`$ ] to produce a normal distribution.
The current initialization functions are as follows:
- Xavier/Glorot Initialization<br/>
	$`𝜎 = \sqrt{\frac{2}{in}}`$<br/>
- He Initialization<br/>
	$`𝜎 = \sqrt{\frac{2}{in  +  out}}`$<br/>

---
Activation Functions
---

The current activation functions [ $`g(v)`$ ] are as follows:
- Linear<br/>
	$`g(v) = v`$<br/>
	$`g'(v) = 1`$
- Sigmoid<br/>
	$`g(v) = \frac{v}{1 + e^{-v}}`$<br/>
	$`g'(v) = g(v) * (1 - g(v))`$
- ReLU<br/>
	$`g(v) = max(0, v)`$<br/>
	$`
	g'(v) =
	\begin{cases}
	0.0,  & \text{if } v \leq 0 \\
	1.0, & \text{if } v > 0
	\end{cases}
	`$
- Tanh<br/>
	$`g(v) = \frac{e^v - e^{-v}}{e^v + e^{-v}}`$<br/>
	$`g'(v) = 1 - g(v)^2`$

---
Loss Functions
---

Note: The neural network only utilizes the derived loss function. However, the underived function [ $`L`$ ] is also shown.<br/>
The current loss functions [ $`L`$ ] are as follows:
- Mean Squared Error<br/>
	$`$L = \frac{1}{n}\sum_{i=1}^n{(\hat y_i - y_i)^2}$`$<br/>
	$`$L_i' = 2(\hat y_i - y_i)$`$<br/>
- Binary Cross-Entropy <br/>
	$`$L = -\frac{1}{n}\sum_{i=1}^n{y_i · ln(\hat y_i) + (1 - y_i) · ln(1 - \hat y_i)}$`$<br/>
	$`$L_i' = \frac{\hat y - y}{(1 - \hat y)\hat y} $`$<br/>

$` n = `$ amount of output values<br/>
$` \hat y = `$ predicted value<br/>
$` y = `$ target/expected/true value

Please note that a value (epsilon) of 10 billion clips the gradient of the loss function so that it does not approach infinity.
 
---
Learning
---

Learning involves changing the weights and biases of the neuron to an optimal value that can evaluate precise outputs.

Gradient descend is used to calculate the  "right" values of the weights and biases.
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

<h4>Methods of Learning</h4>

The current methods of learning are as follows:
- Mean Squared Error (MSE) using Back Propagation (BPG)   ***learn_bpg_mse***<br/>
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
  	The Hidden Layer
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
	$`wij = `$ the weight of which its gradient is to be calculated.<br/>
	$`ai = `$ the result of the previous neuron.<br/>
	$`εj = `$ the error term of this hidden layer.
	
	------------------
	
	For deeper neural networks, recursion is done.
	$`$ε_j = g'_j(z_j) \times \sum_k ε_kw_{jk}$`$
	$`$ε_i = g'_i(z_i) \times \sum_k ε_jw_{ij}$`$
	$`$ε_h = g'_h(z_h) \times \sum_k ε_iw_{hi}$`$
	and so on.

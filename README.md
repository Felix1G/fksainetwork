# fksainetwork

[<img alt="github" src="https://img.shields.io/badge/github-Felix1G/fksainetwork-8da0cb?style=for-the-badge&labelColor=555555&logo=github" height="20">](https://github.com/Felix1G/fksainetwork)
[<img alt="crates.io" src="https://img.shields.io/crates/v/fksainetwork.svg?style=for-the-badge&color=fc8d62&logo=rust" height="20">](https://crates.io/crates/fksainetwork)
[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-fksainetwork-66c2a5?style=for-the-badge&labelColor=555555&logo=docs.rs" height="20">](https://docs.rs/fksainetwork)

-- Documentation --

 This Neural Network is coded and created by Felix K.S
 (used to be in C++, ported to Rust)

 The current activation functions are as follows: (index) [function] [derivative function]
 - Linear (0) [value] [1.0]
 - Sigmoid (1) [value / (1 + e^(-value))] [f(value) * (1.0 - f(value))]
 - ReLU (2) [max(0, value)] [if value <= 0.0 ? 0.0 : 1.0]

 The current methods of learning are as follows:
 - Mean Squared Error (MSE) using Back Propogation (BPG)   ***learn_mse_bpg***

		    Equation of the output bias gradient (dE/db) is the error term of the output using the error function (E)

		    εk = Σ [ (ak - tk) * (gk'(zk)) ] where
		    tk = expected value of the output neuron;
		    ak = gk(zk) = Neuron::result;
		    gk = Activation::activate
		    gk` = Activation::activate_derivative;
		    zk = Neuron::value;


		    Equation of the output weight gradient is the error term multipled with
		    the previous neuron result associated with that weight (dE/dwj)
		    dE/dwj = εk * aj where
		    εk = the error term of the output neuron;
		    wj = the weight of which its gradient is to be calculated;
		    aj = the result of the previous neuron associated with wj;

		    ------------------

		    Equation of the hidden layer bias is the utilization of a recursion where
		    all neurons that are connected with this hidden neuron are taken onto account
		    during the calculation of its gradient. (dE/dbj)

		    An error term of j is then created.

		    εj = [ gj'(zj) ][ Σ (εk * wjk) ] where
		    gj' = Activation::activate_derivative;
		    zj = Neuron::value;
		    εk = error term of the output neuron;
		    wjk = the weight of the next neuron associated with this hidden neuron;

		    Hence, the weight gradient can be calculated using the same way as above. (dE/dwij)

		    dE/dwij = εj * ai where
		    wij = the weight of which its gradient is to be calculated;
		    ai = the result of the previous neuron;
		    εj = the error term of this hidden layer;

		    ------------------

		    For deeper neural networks, the error term is then plugged in by recursion.
		    εj = [ gj'(zj) ][ Σ (εk * wjk) ];
		    εi = [ gj'(zi) ][ Σ (εj * wij) ];
		    εh = [ gj'(zh) ][ Σ (εi * whi) ];
		    and so on.

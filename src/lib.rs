mod activation;
mod neuron;
mod tests;
mod util;
mod pooling;
mod loss;
mod initialization;

pub mod network {
    use std::fmt::Formatter;
    use std::fs::File;
    use std::io::Write;
    use bincode::config;
    use bincode_derive::{Decode, Encode};
    use crate::activation::Activation;
    use crate::initialization::Initialization;
    use crate::loss::{Loss};
    use crate::neuron::Neuron;

    /**
    A simple neural network.
    Inner workings are at GitHub: https://github.com/Felix1G/fksainetwork
     */
    #[derive(Encode, Decode, PartialEq, Debug)]
    pub struct Network {
        pub(crate) layers: Vec<Vec<Neuron>>,
        pub(crate) normalize: Vec<bool>,
        pub(crate) normalize_data: Vec<(f32, f32)>, //normalize, gamma, beta
        pub(crate) normalize_grad: Vec<(f32, f32)>,
        pub(crate) std_inv_length_inv: Vec<f32>, //inverse of standard deviation * inverse of layer length (used for batch norm learning)
        loss_func: Loss,
        has_hidden: bool,
        output: Vec<f32>,
        output_total_inv: f32,
        use_softmax: bool
    }

    impl Network {
        /**
        Create a new Network.

        Layers are provided in this format:

                [amount of neurons in layer, initialization, activation, batch normalization].

        Initializations are provided in this format:

                [layer 1 initialization, layer n initialization]

        Activations are provided in this format:

                [layer 1 activation, layer n activation]

        Indices are provided at the documentation of Network.

        IMPORTANT NOTE: layers and activations array size MUST be the same.
        */
        pub fn new(inputs: usize, layers: &[(usize, Initialization, Activation, bool)], loss_func: Loss, use_softmax: bool) -> Self {
            let mut neurons = Vec::<Vec<Neuron>>::new();
            let mut normalize = Vec::<bool>::new();
            let mut normalize_data = Vec::<(f32, f32)>::new();

            //add input
            {
                let mut input_layer = Vec::<Neuron>::new();
                for _ in 0..inputs {
                    input_layer.push(Neuron::new(0, &(inputs,
                                                      Initialization::Xavier, Activation::Linear, false)));
                }
                neurons.push(input_layer);
                normalize.push(false);
                normalize_data.push((1.0f32, 0.0f32));
            }

            let length = layers.len();
            for index in 0..length {
                let mut vec = Vec::<Neuron>::new();
                let prev_len = if index > 0 { layers[index - 1].0 } else { inputs };
                let layer = layers[index];

                for _ in 0..layer.0 {
                    vec.push(Neuron::new(prev_len, &layer))
                }

                neurons.push(vec);
                normalize.push(layer.3);
                normalize_data.push((1f32, 0f32));
            }

            let normalize_grad = vec![(0f32, 0f32); neurons.len()];

            Self {
                has_hidden: neurons.len() > 2,
                std_inv_length_inv: vec![0f32; neurons.len()],
                layers: neurons,
                normalize,
                normalize_data,
                normalize_grad,
                loss_func,
                output: vec![],
                output_total_inv: 0f32,
                use_softmax
            }
        }

        /**
        Calculates the output given the input.

        Returns: output.
        */
        pub fn calculate(&mut self, input: &[f32]) -> Vec<f32> {
            //set index layer to the inputs
            let input_layer = &mut (self.layers[0]);
            for idx in 0..input_layer.len() {
                let neuron = &mut input_layer[idx];
                neuron.raw = input[idx];
                neuron.value = input[idx];
                neuron.result = input[idx];
            }

            //begin calculation
            for index in 1..self.layers.len() {
                //retrieve the input values for the neurons
                let mut input = Vec::<f32>::new();
                {
                    for neuron in &self.layers[index - 1] {
                        input.push(neuron.result);
                    }
                }

                let normalize = self.normalize[index];
                let normalize_data = &self.normalize_data[index];

                //calculate
                {
                    let layer = &mut self.layers[index];
                    for neuron in layer {
                        let mut raw: f32 = 0.0;
                        for index in 0..neuron.weights.len() {
                            raw += neuron.weights[index] * input[index];
                        }

                        raw += neuron.bias;

                        neuron.raw = raw;
                        if !normalize {
                            neuron.value = raw;
                            neuron.result = neuron.activation.activate(raw);
                        }
                    }
                }

                if normalize {
                    let length_inv = 1f32 / self.layers[index].len() as f32;

                    //calculate average
                    let mut total = 0f32;
                    for neuron in &self.layers[index] {
                        total += neuron.raw;
                    }
                    let mean = total * length_inv;

                    //calculate variance
                    total = 0f32;
                    for neuron in &self.layers[index] {
                        let val = neuron.raw - mean;
                        total += val * val;
                    }
                    let variance = total * length_inv;
                    let std_inv = 1.0f32 / (variance + 0.00001f32).sqrt();

                    self.std_inv_length_inv[index] = std_inv * length_inv;

                    //normalize and shift
                    for neuron in &mut self.layers[index] {
                        let z_idx = (neuron.raw - mean) * std_inv;
                        neuron.raw = z_idx;
                        neuron.value = normalize_data.0 * z_idx + normalize_data.1;
                        neuron.result = neuron.activation.activate(neuron.value);
                    }
                }
            }

            let mut output = Vec::<f32>::new();
            for neuron in &self.layers[self.layers.len() - 1] {
                output.push(neuron.result);
            }

            self.output.clear();

            if self.use_softmax {
                //calculate total
                let mut total = 0.0f32;
                for val in &output {
                    total += val.exp();
                }
                let total_inv = 1.0 / total;
                self.output_total_inv = total_inv;

                //calculate softmax output
                let mut softmax_output = Vec::new();
                for index in 0..output.len() {
                    softmax_output.push(output[index].exp() * total_inv);
                }

                //append the results to be used for learning
                self.output = softmax_output.clone();

                softmax_output
            } else {
                self.output = output.clone();
                output
            }
        }

        fn input_layer_err_term_bpg_mse(&self) -> Vec<f32> {
            let mut err_terms = vec![];
            let next_layer = &self.layers[1];

            //loop through this layer
            for idx in 0..self.layers[0].len() {
                let mut error = 0.0f32;

                //add the errors related to this layer's neuron
                for neuron in next_layer {
                    error += neuron.weights[idx] * neuron.error_term;
                    //println!("{} {}", neuron.weights[idx], neuron.error_term)
                }

                //add the error
                err_terms.push(error);
            }

            return err_terms;
        }

        //after gradient calculation from learning, updates of weights and biases are done here
        pub(crate) fn update_weight_and_bias(&mut self, inputs_len: usize, learning_rate: f32) {
            //update weights and biases
            let inputs_len_inv = 1.0f32 / inputs_len as f32;
            for layer_idx in 1..self.layers.len() {
                let layer = &mut self.layers[layer_idx];
                let normalize = self.normalize[layer_idx];
                let normalize_data = &mut self.normalize_data[layer_idx];
                let normalize_grad = &mut self.normalize_grad[layer_idx];

                for neuron in layer {
                    for index in 0..neuron.weights.len() {
                        let gradient = neuron.weights_grad[index] * inputs_len_inv;
                        neuron.weights_grad[index] = 0f32;

                        let mut weight = neuron.weights[index] - learning_rate * gradient;

                        if weight.abs() > 1_000_000.0 { weight = weight.signum() * 0.001; }
                        neuron.weights[index] = weight;
                    }

                    neuron.bias -= learning_rate * neuron.bias_grad * inputs_len_inv;
                    neuron.bias_grad = 0f32;

                    if neuron.bias.abs() > 1_000_000.0 {
                        neuron.bias = neuron.bias.signum() * 0.001;
                    }
                }

                if normalize {
                    normalize_data.0 -= learning_rate * normalize_grad.0 * inputs_len_inv;
                    normalize_data.1 -= learning_rate * normalize_grad.1 * inputs_len_inv;
                    normalize_grad.0 = 0.0;
                    normalize_grad.1 = 0.0;
                }
            }
        }

        /**
        Backpropagation learning using gradient descent.

        Specify the input as an array of inputs. If the array of inputs has more than 1 array,
        then it will be considered a mini-batch.

        pass update_gradients as false if you only need to add the gradients.

        If the inputs size is 0, then you don't want to calculate the values of the network for learning.
        This is typically done if you have called the calculate function before learning.

        Provide the expected values that would be returned by the calculate function.
         */
        pub fn learn(&mut self, learning_rate: f32, inputs: &Vec<Vec<f32>>, expecteds: &Vec<Vec<f32>>,
                     update_gradients: bool) {
            let calculate = inputs.len() > 0;
            let inputs_len = expecteds.len();

            for input_idx in 0..inputs_len {
                let expected = &expecteds[input_idx];

                //calculate network
                if calculate {
                    let input = &inputs[input_idx];
                    self.calculate(&input);
                }

                //get previous layer values
                let mut out_prev_values = Vec::<f32>::new();
                {
                    for neuron in &self.layers[self.layers.len() - 2] {
                        out_prev_values.push(neuron.result);
                    }
                }

                //output layer gradient
                {
                    let mut delta_vec = vec![0f32; 0];
                    let mut total_delta = 0f32;
                    let mut total_delta_with_norm_val = 0f32;

                    let mut index = 0;
                    let length = self.layers.len();
                    let layer = &mut self.layers[length - 1];
                    let normalize = self.normalize[length - 1];
                    let normalize_data = &mut self.normalize_data[length - 1];
                    let normalize_grad = &mut self.normalize_grad[length - 1];
                    let std_inv_length_inv = self.std_inv_length_inv[length - 1];

                    let layer_len = layer.len();

                    for neuron_idx in 0..layer_len {
                        let neuron = &mut layer[neuron_idx];

                        //use the results of the output regardless of whether or not softmax is used
                        let result = self.output[index];
                        let error = self.loss_func.loss_derivative(result, index, expected);

                        let derivative_value = neuron.activation.derivative(neuron.value);
                        let mut delta = error * derivative_value;

                        //softmax is part of learning
                        if self.use_softmax {
                            let mut softmax_derivative = 0f32;
                            for output_idx in 0..self.output.len() {
                                if output_idx == neuron_idx {
                                    continue;
                                }

                                softmax_derivative += self.output[output_idx];
                            }
                            softmax_derivative *= result * self.output_total_inv;
                            delta *= softmax_derivative;
                        }

                        delta_vec.push(delta);
                        total_delta += delta;
                        total_delta_with_norm_val += delta * neuron.raw;
                    }

                    for neuron_idx in 0..layer_len {
                        let neuron = &mut layer[neuron_idx];
                        let mut error = delta_vec[neuron_idx];

                        if normalize {
                            normalize_grad.1 += error;
                            normalize_grad.0 += error * neuron.raw;

                            let normal_grad = error * normalize_data.0;
                            let total_delta_val = total_delta * normalize_data.0;

                            error =
                                (layer_len as f32 * normal_grad -
                                    total_delta_val -
                                    neuron.raw * total_delta_with_norm_val) *
                                    std_inv_length_inv;
                        }
                        neuron.bias_grad += error;

                        for index in 0..neuron.weights.len() {
                            let gradient = error * out_prev_values[index];

                            //add gradient to the total
                            neuron.weights_grad[index] += gradient;
                        }

                        //set the error term
                        neuron.error_term = error;

                        index += 1;
                    }

                    if normalize {
                        normalize_data.0 -= learning_rate * normalize_grad.0;
                        normalize_data.1 -= learning_rate * normalize_grad.1;
                    }
                }

                //hidden layer gradient
                if self.has_hidden {
                    let mut delta_vec = vec![0f32; 0];

                    //loop through layers from the end to the 2nd layer
                    for index in (1..self.layers.len() - 1).rev() {
                        //get previous neuron results for the weight gradients
                        let mut val_array = Vec::<f32>::new();
                        for neuron in &self.layers[index - 1] {
                            val_array.push(neuron.result);
                        }

                        let (prev_layers, next_layers) = self.layers.split_at_mut(index + 1);
                        let layer = &mut prev_layers[index];
                        let normalize = self.normalize[index];
                        let normalize_data = &mut self.normalize_data[index];
                        let normalize_grad = &mut self.normalize_grad[index];
                        let std_inv_length_inv = self.std_inv_length_inv[index];

                        let mut total_delta = 0f32;
                        let mut total_delta_with_norm_val = 0f32;

                        for neuron in &mut *layer {
                            let delta = neuron.activation.derivative(neuron.value);
                            delta_vec.push(delta);
                            total_delta += delta;
                            total_delta_with_norm_val += delta * neuron.raw;
                        }

                        //loop through the neurons of this layer
                        let layer_len = layer.len();
                        for idx in 0..layer_len {
                            let neuron = &mut layer[idx];
                            let delta = delta_vec[idx];
                            let mut error: f32 = 0.0;
                            {
                                let mut error_delta = 0f32;

                                if normalize {
                                    normalize_grad.1 += delta;
                                    normalize_grad.0 += delta * neuron.raw;

                                    let normal_grad = delta * normalize_data.0;
                                    let total_delta_val = total_delta * normalize_data.0;

                                    error_delta =
                                        (layer_len as f32 * normal_grad -
                                            total_delta_val -
                                            neuron.raw * total_delta_with_norm_val) *
                                            std_inv_length_inv;
                                }

                                //get error term
                                let mut error_term_total = 0.0;
                                let next_layer = &next_layers[0];
                                for next_neuron in next_layer {
                                    error_term_total += next_neuron.error_term * next_neuron.weights[idx];
                                }

                                error += delta * error_term_total;
                            }

                            //set the error term
                            neuron.error_term = error;

                            for w_index in 0..neuron.weights.len() {
                                neuron.weights_grad[w_index] += error * val_array[w_index];
                            }

                            neuron.bias_grad += error;
                        }


                        if normalize {
                            normalize_data.0 -= learning_rate * normalize_grad.0;
                            normalize_data.1 -= learning_rate * normalize_grad.1;
                        }

                        delta_vec.clear();
                    }
                }
            }

            if update_gradients {
                self.update_weight_and_bias(inputs_len, learning_rate);
            }
        }
    }

    impl std::fmt::Display for Network {
        fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
            Ok(
                for layer_idx in 0..self.layers.len() {
                    let layer = &self.layers[layer_idx];
                    let normalize_data = &self.normalize_data[layer_idx];
                    write!(formatter, "[layer {}, gamma = {}, beta = {}]\n",
                           layer_idx, normalize_data.0, normalize_data.1).unwrap();

                    for neuron in layer {
                        write!(formatter, "[weight = (").unwrap();

                        let length = neuron.weights.len();
                        for index in 0..length {
                            write!(formatter, "{}", neuron.weights[index]).unwrap();
                            if index < length - 1 {
                                write!(formatter, ", ").unwrap();
                            }
                        }

                        write!(formatter, "), bias = {}, raw = {}, value = {}, result = {}] \n",
                               neuron.bias, neuron.raw, neuron.value, neuron.result).unwrap();
                    }
                    write!(formatter, "\n").unwrap();
                }
            )
        }
    }

    /**
     * Loads the network from the given path.
     */
    pub fn load_network(path: &str) -> Network {
        let data = std::fs::read(path).expect("Unable to read file");
        let (network, _len): (Network, usize) =
            bincode::decode_from_slice(&data, config::standard()).unwrap();
        network
    }

    /**
    * Saves the network to the given path in binary format using bincode.
    */
    pub fn save_network(path: &str, network: &Network) {
        let data: Vec<u8> = bincode::encode_to_vec(network, config::standard()).unwrap();
        let mut file = File::create(path).unwrap();
        file.write_all(&data).unwrap();
    }

    pub mod cnn {
        use std::fs;
        use std::fs::File;
        use std::io::Write;
        use std::path::Path;
        use bincode::config;
        use bincode_derive::{Encode, Decode};
        use bmp::Pixel;
        use crate::activation::Activation;
        use crate::initialization::Initialization;
        use crate::loss::Loss;
        use crate::network::Network;
        use crate::neuron::ConvolutionalLayer;
        use crate::pooling::Pooling;
        use crate::util::{Matrix, max};

        #[derive(Encode, Decode, PartialEq, Debug)]
        pub struct ConvolutionalNetwork {
            pub network: Network,
            network_input_arr: Vec<f32>,
            layers: Vec<ConvolutionalLayer>,
            width: usize,
            height: usize,
            channels: usize
        }

        impl ConvolutionalNetwork {
            /**
            Creates a convolutional neural network built on top of the Feed Forward Network.

            All kernel size and pool size are as a square. Specify the square's side for the size.

            "convolution_layers" are layers of multiple kernels and pooling methods which are supplied in this way:

                    (size of the filter, filter initializations (num of initializations = num of filters),
                    kernel activation function, size of the pooling, the method of pooling,
                    use batch normalization)

            <br/>

            Do note in each layer of the kernels, the size of the kernel arrays must also be the same.

            <br/>

            IMPORTANT NOTE: inputs must follow the width, height, and channels specified.
            the feed forward network input layer is calculated automatically.
            Please only specify the hidden and output layers for the neuron layers.
            */
            pub fn new(convolution_layers: &[(usize, &[Initialization], Activation, usize, Pooling, bool)],
                       input_width: usize, input_height: usize, input_channels: usize,
                       neuron_layers: &[(usize, Initialization, Activation, bool)],
                       loss_func: Loss, use_softmax: bool) -> Self {
                let mut network_layers = Vec::<ConvolutionalLayer>::new();

                //add the first layer as the layer for the inputs
                network_layers.push(ConvolutionalLayer {
                    kernel_layers: vec![],
                    kernel_size: 0,
                    pooling_size: 0,
                    pooling_method: Pooling::Max,
                    bias: vec![],
                    values: vec![],
                    results: vec![],
                    learning_val: vec![],
                    kernel_grad: vec![],
                    bias_grad: vec![],
                    switch: vec![],
                    pooleds_raw: vec![],
                    pooleds: Vec::<Matrix>::from(
                        (0..input_channels).map(|_|
                            Matrix {
                                w: input_width,
                                h: input_height,
                                values: vec![0f32; input_width * input_height]
                            }
                        ).collect::<Vec<_>>()
                    ),
                    error_terms: vec![],
                    activation: Activation::Linear,
                    normalize: false,
                    normalize_data: vec![],
                    normalize_grad: vec![],
                    temp_matrix: Matrix::empty(),
                    temp_pooling_arr: vec![]
                });

                //calculate network input layer size
                let mut img_width = input_width;
                let mut img_height = input_height;
                let mut channels = input_channels;
                for idx in 0..convolution_layers.len() {
                    let conv_layer = &convolution_layers[idx];

                    //convolution
                    let conv_width = img_width - conv_layer.0 + 1;
                    let conv_height = img_height - conv_layer.0 + 1;

                    //add network layer
                    network_layers.push(ConvolutionalLayer::new(
                        channels, conv_width, conv_height,
                        conv_layer.0, &conv_layer.1,
                        conv_layer.2.clone(), conv_layer.3,
                        conv_layer.4.clone(), conv_layer.5
                    ));

                    //pooling
                    img_width = conv_width / conv_layer.3;
                    img_height = conv_height / conv_layer.3;

                    //multiply the channel
                    channels = conv_layer.1.len();
                }
                let input_layer_size = img_width * img_height * channels;

                //create fully connected network
                let mut layers = Vec::<(usize, Initialization, Activation, bool)>::new();
                for layer in neuron_layers {
                    layers.push(layer.clone());
                }
                let fully_connected_network = Network::new(
                    input_layer_size, &layers, loss_func, use_softmax);

                return ConvolutionalNetwork {
                    network: fully_connected_network,
                    network_input_arr: vec![0.0f32; input_layer_size],
                    layers: network_layers,
                    width: input_width,
                    height: input_height,
                    channels: input_channels
                };
            }

            /**
            Calculates the output given the input channels.

            The input array must be supplied like this: (x, y) coordinates

                    [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)]
            <br/>
            Returns: output.
             */
            pub fn calculate(&mut self, inputs: &[&Matrix]) -> Vec<f32> {
                if inputs.len() != self.channels {
                    panic!("Inputs channels ({}) and specified channels ({}) are different!",
                           inputs.len(), self.channels);
                }

                //copy inputs to input conv layer
                let input_layer = &mut self.layers[0];
                for index in 0..(inputs.len()) {
                    let input_from = &inputs[index];
                    let input_to = &mut input_layer.pooleds[index];
                    input_to.copy(input_from);
                }

                //calculate every layer
                let mut layer_index = 1;
                let layers_len = self.layers.len();
                while layer_index < layers_len {
                    let (left, right) =
                        &mut self.layers.split_at_mut(layer_index);
                    right[0].calculate(&left[layer_index - 1]);
                    layer_index += 1;
                }

                //feed the output conv layer to the input feed forward network
                let output_layer = &self.layers[layers_len - 1];
                let mut index = 0usize;
                for matrix in &output_layer.pooleds {
                    for value in &matrix.values {
                        self.network_input_arr[index] = *value;
                        index += 1;
                    }
                }

                //calculate network
                return self.network.calculate(&self.network_input_arr);
            }

            pub fn learn(&mut self, learning_rate: f32, inputs: &Vec<Vec<&Matrix>>, expecteds: &Vec<Vec<f32>>) {
                let inputs_len = inputs.len();

                for inputs_idx in 0..inputs_len {
                    let input = &inputs[inputs_idx];

                    //feed forward network learns
                    self.calculate(&input);
                    self.network.learn(learning_rate, &vec![], expecteds, false);
                    let input_layer_err_terms = self.network.input_layer_err_term_bpg_mse();

                    //calculate derivatives
                    for layer in &mut self.layers {
                        for value_idx in 0..layer.values.len() {
                            let value = &layer.values[value_idx];
                            let derivative = &mut layer.learning_val[value_idx];
                            let switch = &layer.switch[value_idx];
                            for val_idx in 0..value.values.len() {
                                derivative.values[val_idx] = switch.values[val_idx] *
                                    layer.activation.derivative(value.values[val_idx]);
                            }
                        }
                    }

                    //get the error terms of the output convolutional layer
                    {
                        let layer_length = self.layers.len();
                        let output_layer = &mut self.layers[layer_length - 1];

                        //calculate error terms
                        {
                            //learn the error terms first
                            for value_layer_idx in 0..output_layer.values.len() {
                                let pooled = &output_layer.pooleds[value_layer_idx];
                                let pooled_raw = &output_layer.pooleds_raw[value_layer_idx];
                                let derivative = &output_layer.learning_val[value_layer_idx];
                                let pooled_offset = value_layer_idx * pooled.values.len();

                                let mut x_loc = 0usize;
                                let mut y_loc = 0usize;
                                let mut pool_x = 0usize;
                                let mut pool_y = 0usize;

                                let mut error = 0.0f32;
                                let mut error_gamma = 0.0f32;
                                let mut error_beta = 0.0f32;

                                while x_loc + output_layer.pooling_size <= derivative.w {
                                    while y_loc + output_layer.pooling_size <= derivative.h {
                                        let mut total_derivative = 0.0;

                                        for x in x_loc..(x_loc + output_layer.pooling_size) {
                                            for y in y_loc..(y_loc + output_layer.pooling_size) {
                                                //add the error
                                                total_derivative += derivative.get(x, y);
                                            }
                                        }

                                        //get error term
                                        let err_index = pooled_offset +
                                            pooled.index_to_one_d(pool_x, pool_y);
                                        let err_term = input_layer_err_terms[err_index];
                                        error += err_term * total_derivative;

                                        if output_layer.normalize {
                                            error_gamma += err_term * pooled_raw.get(pool_x, pool_y);
                                            error_beta += err_term;
                                        }

                                        y_loc += output_layer.pooling_size;
                                        pool_y += 1;
                                    }
                                    x_loc += output_layer.pooling_size;
                                    pool_x += 1;

                                    y_loc = 0;
                                    pool_y = 0;
                                }

                                //set the error
                                if output_layer.normalize {
                                    let mut out = output_layer.normalize_grad[value_layer_idx];
                                    out.0 += error_gamma;
                                    out.1 += error_beta;

                                    error *= output_layer.normalize_data[value_layer_idx].0; //multiply with gamma
                                }

                                output_layer.error_terms[value_layer_idx] = error;
                                output_layer.bias_grad[value_layer_idx] += error;
                            }
                        }
                    }

                    //get the error term of the rest of the convolutional layers
                    if self.layers.len() > 1 {
                        //loop through all convolutional layers from the end before the output
                        for layer_index in (1..(self.layers.len() - 1)).rev() {
                            //get this and the next layer
                            let (left_layers, right_layers) =
                                self.layers.split_at_mut(layer_index + 1);
                            let next_layer = &mut right_layers[0];
                            let this_layer = &mut left_layers[left_layers.len() - 1];

                            //loop through the next layer outputs (to get the error term of the next layer)
                            for value_layer_idx in 0..this_layer.values.len() {
                                let derivative = &this_layer.learning_val[value_layer_idx];

                                let mut err_term = 0f32;
                                let mut err_gamma = 0.0f32;
                                let mut err_beta = 0.0f32;

                                //go through all kernels related to the pooled matrix of this layer
                                for kernel_layer_index in 0..next_layer.kernel_layers.len() {
                                    //get kernel layers
                                    let kernel_layer = &next_layer.kernel_layers[kernel_layer_index];
                                    //get kernel associated with this layer's value matrix
                                    let kernel = &kernel_layer[value_layer_idx];

                                    let mut error = 0f32;
                                    let mut total_weight = 0f32;

                                    //calculate the total derivative
                                    for kx in 0..kernel.w {
                                        for ky in 0..kernel.h {
                                            let mut total_derivative = 0.0f32;

                                            //beginning index a.k.a offset x/y
                                            let ox = kx * this_layer.pooling_size;
                                            let oy = ky * this_layer.pooling_size;

                                            //total value of the value matrix of this layer related
                                            //to this kernel's weight
                                            for x_loc in ox..(kernel.w * this_layer.pooling_size + ox) {
                                                for y_loc in oy..(kernel.h * this_layer.pooling_size + oy) {
                                                    total_derivative += derivative.get(x_loc, y_loc);
                                                }
                                            }

                                            //add the error
                                            error += total_derivative * kernel.get(kx, ky);
                                            total_weight += kernel.get(kx, ky);
                                        }
                                    }

                                    //multiply with the error term associated with this kernel layer
                                    let next_layer_err_term = next_layer.error_terms[kernel_layer_index];

                                    if this_layer.normalize {
                                        let mut total_raw = 0f32;
                                        this_layer.pooleds_raw[value_layer_idx].values
                                            .iter().for_each(|x| total_raw += x);

                                        let normalize_delta = total_weight * next_layer_err_term;
                                        err_gamma += normalize_delta * total_raw;
                                        err_beta += normalize_delta;

                                        error *= this_layer.normalize_data[value_layer_idx].0; //multiply with gamma
                                    }

                                    err_term += error * next_layer_err_term;
                                }

                                this_layer.error_terms[value_layer_idx] = err_term;
                                this_layer.bias_grad[value_layer_idx] += err_term;

                                if this_layer.normalize {
                                    let mut normalize_grad = &mut this_layer.normalize_grad[value_layer_idx];
                                    normalize_grad.0 += err_gamma;
                                    normalize_grad.1 += err_beta;
                                }
                            }
                        }
                    }

                    //calculate weight gradients
                    for layer_index in 1..self.layers.len() {
                        let (left_layers, right_layers) =
                            self.layers.split_at_mut(layer_index);
                        let this_layer = &mut right_layers[0];
                        let prev_layer = &mut left_layers[left_layers.len() - 1];

                        //loop through all the kernels
                        for kernel_layer_idx in 0..this_layer.kernel_layers.len() {
                            let kernels = &mut this_layer.kernel_layers[kernel_layer_idx];
                            let kernel_grads = &mut this_layer.kernel_grad[kernel_layer_idx];
                            let value = &mut this_layer.values[kernel_layer_idx];
                            let error_term = this_layer.error_terms[kernel_layer_idx];

                            //update kernel weights
                            for kernel_idx in 0..kernels.len() {
                                let input = &prev_layer.pooleds[kernel_idx];
                                let kernel = &mut kernels[kernel_idx];
                                let kernel_grad = &mut kernel_grads[kernel_idx];

                                //loop through each kernel weight
                                for ker_x in 0..kernel.w {
                                    for ker_y in 0..kernel.h {
                                        //get the total value connected with this weight
                                        let mut total_value = 0.0;
                                        for input_x in ker_x..(ker_x + value.w) {
                                            for input_y in ker_y..(ker_y + value.h) {
                                                total_value += input.get(input_x, input_y);
                                            }
                                        }

                                        let gradient = error_term * total_value;

                                        kernel_grad.set(ker_x, ker_y, kernel_grad
                                            .get(ker_x, ker_y) + gradient);
                                    }
                                }
                            }
                        }
                    }
                }

                //calculate change of the fully connected network
                self.network.update_weight_and_bias(inputs_len, learning_rate);

                //calculate the change in weights and biases of all the convolutional layers
                let inputs_len_inv = 1.0f32 / inputs_len as f32;
                for layer_index in 1..self.layers.len() {
                    let layer = &mut self.layers[layer_index];

                    //loop through all the kernels
                    for kernel_layer_idx in 0..layer.kernel_layers.len() {
                        let kernels = &mut layer.kernel_layers[kernel_layer_idx];
                        let kernel_grads = &mut layer.kernel_grad[kernel_layer_idx];

                        //update kernel weights
                        for kernel_idx in 0..kernels.len() {
                            let kernel = &mut kernels[kernel_idx];
                            let kernel_grad = &mut kernel_grads[kernel_idx];

                            //loop through each kernel weight
                            for ker_x in 0..kernel.w {
                                for ker_y in 0..kernel.h {
                                    kernel.set(ker_x, ker_y, kernel.get(ker_x, ker_y) -
                                        learning_rate * kernel_grad.get(ker_x, ker_y) * inputs_len_inv);
                                    kernel_grad.set(ker_x, ker_y, 0f32);
                                }
                            }
                        }

                        layer.bias[kernel_layer_idx] -= learning_rate *
                            layer.bias_grad[kernel_layer_idx] * inputs_len_inv;
                        layer.bias_grad[kernel_layer_idx] = 0f32;

                        if layer.normalize {
                            let normalize_data = &mut layer.normalize_data[kernel_layer_idx];
                            let normalize_grad = &mut layer.normalize_grad[kernel_layer_idx];
                            normalize_data.0 -= learning_rate * normalize_grad.0 * inputs_len_inv;
                            normalize_data.1 -= learning_rate * normalize_grad.1 * inputs_len_inv;
                            normalize_grad.0 = 0f32;
                            normalize_grad.1 = 0f32;
                        }
                    }
                }

                //copy the cache kernel layers to the actual layers
                /*for conv_layer in &mut self.layers {
                    for layer_idx in 0..conv_layer.kernel_layers_temp.len() {
                        let kernels = &mut conv_layer.kernel_layers[layer_idx];
                        let kernels_temp = &mut conv_layer.kernel_layers_temp[layer_idx];

                        for kernel_idx in 0..kernels_temp.len() {
                            let kernel = &mut kernels[kernel_idx];
                            let kernel_temp = &kernels_temp[kernel_idx];
                            for idx in 0..kernel.values.len() {
                                let mut value = kernel_temp.values[idx];
                                if value.abs() > 2f32 {
                                    value = value.signum() * 0.001f32;
                                }
                            }
                        }

                        let mut bias = conv_layer.bias[layer_idx];
                        if bias.abs() > 2f32 {
                            bias = bias.signum() * 0.001f32;
                        }
                        conv_layer.bias[layer_idx] = bias;
                    }
                }*/
            }
        } //impl Convolution Network

        /**
         * Loads the network from the given path.
         */
        pub fn load_cnn_network(path: &str) -> ConvolutionalNetwork {
            let data = fs::read(path).expect("Unable to read file");
            let (network, _len): (ConvolutionalNetwork, usize) =
                bincode::decode_from_slice(&data, config::standard()).unwrap();
            network
        }

        /**
         * Saves the network to the given path in binary format using bincode.
         */
        pub fn save_cnn_network(path: &str, network: &ConvolutionalNetwork) {
            let data: Vec<u8> = bincode::encode_to_vec(network, config::standard()).unwrap();
            let mut file = File::create(path).unwrap();
            file.write_all(&data).unwrap();
        }

        pub fn cnn_network_bmp(dir: &str, network: &ConvolutionalNetwork) {
            if !Path::new(dir).exists() {
                fs::create_dir(dir).unwrap();
            }

            let mut layer_idx = 0usize;
            for layer in &network.layers {
                let mut value_idx = 0usize;
                let mut pooled_idx = 0usize;

                for value in &layer.results {
                    let max_val = 1.0;/*value.values[value.values.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                        .map(|(index, _)| index)
                        .unwrap()];*/
                    let mut img = bmp::Image::new(value.w as u32, value.h as u32);
                    for x in 0..value.w {
                        for y in 0..value.h {
                            let l = max(0.0, value.get(x, y) / max_val * 255.0) as u8;
                            img.set_pixel(x as u32, y as u32, Pixel::new(l, l, l));
                        }
                    }
                    img.save(format!("{dir}\\{layer_idx}-{value_idx}val.bmp")).unwrap();
                    value_idx += 1;
                }

                for pooled in &layer.pooleds {
                    let max_val = pooled.values[pooled.values.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                        .map(|(index, _)| index)
                        .unwrap()];
                    let mut img = bmp::Image::new(pooled.w as u32, pooled.h as u32);
                    for x in 0..pooled.w {
                        for y in 0..pooled.h {
                            let l = (pooled.get(x, y) * 255.0 / max_val) as u8;
                            img.set_pixel(x as u32, y as u32, Pixel::new(l, l, l));
                        }
                    }
                    img.save(format!("{dir}\\{layer_idx}-{pooled_idx}pol.bmp")).unwrap();
                    pooled_idx += 1;
                }

                layer_idx += 1;
            }
        }
    }
}

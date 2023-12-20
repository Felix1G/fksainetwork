mod activation;
mod neuron;
mod tests;
mod util;
mod pooling;

pub mod network {
    use std::fmt::Formatter;
    use std::fs::File;
    use std::io::Write;
    use bincode::config;
    use bincode_derive::{Decode, Encode};
    use crate::activation::{derivative};
    use crate::neuron::Neuron;

    /**
    A simple neural network.
    Inner workings are at GitHub: https://github.com/Felix1G/fksainetwork
     */
    #[derive(Encode, Decode, PartialEq, Debug)]
    pub struct Network {
        pub(crate) layers: Vec::<Vec::<Neuron>>,
        has_hidden: bool,
    }

    impl Network {
        /**
        Create a new Network.

        Layers are provided in this format:

                [layer 0 size, layer 1 size, layer n size].

        Activations are provided in this format:

                [layer 0 activation (although not used), layer 1 activation, layer n activation]

        Indices are provided at the documentation of Network.

        IMPORTANT NOTE: layers and activations array size MUST be the same.
        */
        pub fn new(layers: &[usize], activations: &[usize]) -> Self {
            let mut neurons = Vec::<Vec::<Neuron>>::new();

            let length = layers.len();
            for index in 0..length {
                let mut vec = Vec::<Neuron>::new();
                let prev_len = if index > 0 { layers[index - 1] } else { 0 };
                let activation = activations[index];

                for _ in 0..layers[index] {
                    vec.push(Neuron::new(prev_len, activation))
                }

                neurons.push(vec);
            }

            Self {
                layers: neurons,
                has_hidden: layers.len() > 2
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

                //calculate
                let layer = &mut (&mut self.layers)[index];
                for neuron in layer {
                    neuron.calculate(&input);
                }
            }

            let mut output = Vec::<f32>::new();
            for neuron in &self.layers[self.layers.len() - 1] {
                output.push(neuron.result);
            }

            output
        }

        fn learn_bpg_mse_err_term(&mut self,
                                  layer_index: usize, neuron_index: usize,
                                  output_err_terms: &[f32]) -> f32 {
            return if layer_index == self.layers.len() - 1 {
                output_err_terms[neuron_index]
            } else {
                let neuron = &self.layers[layer_index][neuron_index];

                //derived activation
                let value = derivative(neuron.activation, neuron.value);

                let next_layer_index = layer_index + 1;

                //get error term
                let mut error = 0.0;
                for index in 0..self.layers[next_layer_index].len() {
                    let error_term = self.learn_bpg_mse_err_term(
                        next_layer_index,
                        index,
                        output_err_terms);
                    error += error_term * self.layers[layer_index + 1][index].weights[neuron_index];
                }

                value * error
            }
        }

        /**
        BPG learning using the MSE function.

        Provide the expected values that would be returned by the calculate function.
         */
        pub fn learn_bpg_mse(&mut self, learning_rate: f32, expected: &[f32]) {
            let mut output_err_terms = Vec::<f32>::new();

            //calculate previous layer values
            let mut out_prev_values = Vec::<f32>::new();
            {
                for neuron in &self.layers[self.layers.len() - 2] {
                    out_prev_values.push(neuron.result);
                }
            }

            //output layer gradient
            let mut index = 0;
            let length = self.layers.len();
            for neuron in &mut self.layers[length - 1] {
                let error = neuron.result - expected[index];

                let derivative_value = derivative(neuron.activation, neuron.value);
                let delta = error * derivative_value;

                //add the error term
                if self.has_hidden { output_err_terms.push(delta); }

                for index in 0..neuron.weights.len() {
                    let gradient = delta * out_prev_values[index];

                    //descend the gradient
                    neuron.weights_temp[index] = neuron.weights[index] - learning_rate * gradient;
                }

                neuron.bias_temp = neuron.bias - learning_rate * delta;

                index += 1;
            }

            //hidden layer gradient
            if self.has_hidden {
                for index in (1..self.layers.len() - 1).rev() {
                    let mut val_array = Vec::<f32>::new();
                    for neuron in &self.layers[index - 1] {
                        val_array.push(neuron.result);
                    }

                    for idx in 0..self.layers[index].len() {
                        let mut error: f32 = 0.0;
                        {
                            error += self.learn_bpg_mse_err_term(
                                index,
                                idx,
                                &output_err_terms);
                        }

                        let neuron = &mut self.layers[index][idx];

                        for w_index in 0..neuron.weights.len() {
                            neuron.weights_temp[w_index] = neuron.weights[w_index] -
                                learning_rate * error * val_array[w_index];
                        }
                    }
                }
            }

            //set the temporary values to the actual values
            for layer in &mut self.layers {
                for neuron in layer {
                    for index in 0..neuron.weights.len() {
                        let mut weight = neuron.weights_temp[index];
                        if weight > 1_000_000.0 { weight = 0.001; }
                        neuron.weights[index] = weight;
                    }
                    if neuron.bias_temp > 1_000_000.0 { neuron.bias_temp = 0.001; }
                    neuron.bias = neuron.bias_temp;
                }
            }
        }
    }

    impl std::fmt::Display for Network {
        fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
            Ok(
                for layer in &self.layers {
                    for neuron in layer {
                        write!(formatter, "[weight = (").unwrap();

                        let length = neuron.weights.len();
                        for index in 0..length {
                            write!(formatter, "{}", neuron.weights[index]).unwrap();
                            if index < length - 1 {
                                write!(formatter, ", ").unwrap();
                            }
                        }

                        write!(formatter, "), bias = {}, value = {}, result = {}] \n",
                               neuron.bias, neuron.value, neuron.result).unwrap();
                    }
                    write!(formatter, "\n").unwrap();
                }
            )
        }
    }

    /**
     * Loads the network from the given path.
     */
    pub fn load_network(path: String) -> Network {
        let data = std::fs::read(path).expect("Unable to read file");
        let (network, _len): (Network, usize) =
            bincode::decode_from_slice(&data, config::standard()).unwrap();
        network
    }

    /**
    * Saves the network to the given path in binary format using bincode.
    */
    pub fn save_network(path: String, network: &Network) {
        let data: Vec<u8> = bincode::encode_to_vec(network, config::standard()).unwrap();
        let mut file = File::create(path).unwrap();
        file.write_all(&data).unwrap();
    }

    pub mod cnn {
        use crate::network::Network;
        use crate::neuron::ConvolutionalLayer;
        use crate::util::Matrix;

        #[derive(Debug)]
        pub struct ConvolutionalNetwork {
            network: Network,
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

                    (amount of filters, size of the filter, activation function, size of the pooling, the method of pooling)

            <br/>

            Do note in each layer of the kernels, the size of the kernel arrays must also be the same.

            <br/>

            IMPORTANT NOTE: inputs must follow the width, height, and channels specified.
            the feed forward network input layer is calculated automatically.
            Please only specify the hidden and output layers for the neuron layers.
            */
            pub fn new(convolution_layers: &[(usize, usize, usize, usize, usize)],
                       input_width: usize, input_height: usize, input_channels: usize,
                       neuron_layers: &[usize], layer_activations: &[usize]) -> Self {
                let mut neurons = [0usize, neuron_layers.len() + 1];
                let mut activations = [0usize, layer_activations.len() + 1];

                //set neurons and activations
                for idx in 1..neurons.len() {
                    neurons[idx] = neuron_layers[idx - 1];
                    activations[idx] = layer_activations[idx - 1];
                }

                let mut network_layers = Vec::<ConvolutionalLayer>::new();

                //add the first layer as the layer for the inputs
                network_layers.push(ConvolutionalLayer {
                    kernels_temp: vec![],
                    kernels_layers: vec![],
                    kernels: 0,
                    kernel_size: 0,
                    pooling_size: 0,
                    pooling_method: 0,
                    bias_temp: 0.0,
                    bias: 0.0,
                    value: vec![],
                    result: vec![],
                    pooled: Vec::<Matrix>::from(
                        (0..input_channels).map(|_|
                            Matrix {
                                w: input_width,
                                h: input_height,
                                values: vec![0f32; input_width * input_height]
                            }
                        ).collect::<Vec<_>>()
                    ),
                    activation: 0,
                    temp_matrix: Matrix::empty(),
                    temp_pooling_arr: vec![]
                });

                //calculate network input layer size
                let mut img_width = input_width;
                let mut img_height = input_height;
                let mut channels = input_channels;
                for idx in 0..convolution_layers.len() {
                    let conv_layer = convolution_layers[idx];

                    //convolution
                    let conv_width = img_width - conv_layer.1 + 1;
                    let conv_height = img_height - conv_layer.1 + 1;

                    //add network layer
                    network_layers.push(ConvolutionalLayer::new(
                        channels, conv_width, conv_height,
                        conv_layer.0, conv_layer.1,
                        conv_layer.2, conv_layer.3, conv_layer.4
                    ));

                    //pooling
                    img_width = conv_width / conv_layer.3;
                    img_height = conv_height / conv_layer.3;

                    //multiply the channel
                    channels = conv_layer.0;
                }
                let input_layer_size = img_width * img_height * channels;
                neurons[0] = input_layer_size;

                //create feed forward network
                let ff_network = Network::new(&neurons, &activations);

                return ConvolutionalNetwork {
                    network: ff_network,
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
            pub fn calculate(&mut self, inputs: &[Matrix]) -> Vec<f32> {
                if inputs.len() != self.channels {
                    panic!("Inputs channels ({}) and specified channels ({}) are different!",
                           inputs.len(), self.channels);
                }

                let input_layer = &mut self.layers[0];
                for index in 0..(inputs.len()) {
                    let input_from = &inputs[index];
                    let input_to = &mut input_layer.pooled[index];
                    input_to.copy(input_from);
                }

                let mut layer_index = 1;
                let layers_len = self.layers.len();
                while layer_index < layers_len {
                    let (left, right) =
                        &mut self.layers.split_at_mut(layer_index);
                    right[0].calculate(&left[layer_index - 1]);
                    layer_index += 1;
                }

                let output_layer = &self.layers[layers_len - 1];
                let mut index = 0usize;
                for matrix in &output_layer.pooled {
                    for value in &matrix.values {
                        self.network_input_arr[index] = *value;
                        index += 1;
                    }
                }

                return self.network.calculate(&self.network_input_arr);
            }
        }
    }
}

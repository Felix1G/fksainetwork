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
        loss_func: Loss,
        has_hidden: bool
    }

    impl Network {
        /**
        Create a new Network.

        Layers are provided in this format:

                [layer 0 size, layer 1 size, layer n size].

        Initializations are provided in this format:

                [layer 1 initialization, layer n initialization]

        Activations are provided in this format:

                [layer 1 activation, layer n activation]

        Indices are provided at the documentation of Network.

        IMPORTANT NOTE: layers and activations array size MUST be the same.
        */
        pub fn new(layers: &[usize], initializations: &[Initialization],
                   activations: &[Activation], loss_func: Loss) -> Self {
            let mut neurons = Vec::<Vec<Neuron>>::new();

            let length = layers.len();
            for index in 0..length {
                let mut vec = Vec::<Neuron>::new();
                let prev_len = if index > 0 { layers[index - 1] } else { 0 };
                let init = if index == 0 { &Initialization::Xavier } else { &initializations[index - 1] };
                let activation = if index == 0 { Activation::Sigmoid } else { activations[index - 1].clone() };
                let number_of_neurons = layers[index];

                for _ in 0..number_of_neurons {
                    vec.push(Neuron::new(
                        prev_len, number_of_neurons, activation, &init))
                }

                neurons.push(vec);
            }

            Self {
                layers: neurons,
                loss_func,
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

        fn input_layer_err_term_bpg_mse(&self) -> Vec<f32> {
            let mut err_terms = vec![];
            let next_layer = &self.layers[1];

            //loop through this layer
            for idx in 0..self.layers[0].len() {
                let mut error = 0.0f32;

                //add the errors related to this layer's neuron
                for neuron in next_layer {
                    error += neuron.weights[idx] * neuron.error_term;
                }

                //add the error
                err_terms.push(error);
            }

            return err_terms;
        }

        /**
        BPG learning using the MSE function.

        Provide the expected values that would be returned by the calculate function.
         */
        pub fn learn(&mut self, learning_rate: f32, expected: &[f32]) {
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
                let error = self.loss_func.loss_derivative(
                    neuron.result, index, expected);

                let derivative_value = neuron.activation.derivative(neuron.value);
                let delta = error * derivative_value;

                //set the error term
                neuron.error_term = delta;

                for index in 0..neuron.weights.len() {
                    let gradient = delta * out_prev_values[index];

                    //descend the gradient
                    neuron.weights_temp[index] = neuron.weights[index] - learning_rate * gradient;
                }

                neuron.bias -= learning_rate * delta;

                index += 1;
            }

            //hidden layer gradient
            if self.has_hidden {
                //loop through layers from the end to the 2nd layer
                for index in (1..self.layers.len() - 1).rev() {
                    //get previous neuron results for the weight gradients
                    let mut val_array = Vec::<f32>::new();
                    for neuron in &self.layers[index - 1] {
                        val_array.push(neuron.result);
                    }

                    let (prev_layers, next_layers) = self.layers.split_at_mut(index + 1);
                    let layer = &mut prev_layers[index];

                    //loop through the neurons of this layer
                    for idx in 0..layer.len() {
                        let neuron = &mut layer[idx];
                        let mut error: f32 = 0.0;
                        {
                            let value = neuron.activation.derivative(neuron.value);

                            //get error term
                            let mut error_term_total = 0.0;
                            let next_layer = &next_layers[0];
                            for next_neuron in next_layer {
                                error_term_total += next_neuron.error_term * next_neuron.weights[idx];
                            }

                            error += value * error_term_total;
                        }

                        //set the error term
                        neuron.error_term = error;

                        for w_index in 0..neuron.weights.len() {
                            neuron.weights_temp[w_index] = neuron.weights[w_index] -
                                learning_rate * error * val_array[w_index];
                        }

                        neuron.bias -= learning_rate * error;
                    }
                }
            }

            //set the temporary values to the actual values
            for layer_idx in 1..self.layers.len() {
                let layer = &mut self.layers[layer_idx];
                for neuron in layer {
                    for index in 0..neuron.weights_temp.len() {
                        let mut weight = neuron.weights_temp[index];
                        if weight.abs() > 1_000_000.0 { weight = weight.signum() * 0.001; }
                        neuron.weights[index] = weight;
                    }
                    if neuron.bias.abs() > 1_000_000.0 { neuron.bias = neuron.bias.signum() * 0.001; }
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

                    (amount of filters, size of the filter,
                    initialization function, activation function,
                    size of the pooling, the method of pooling)

            <br/>

            Do note in each layer of the kernels, the size of the kernel arrays must also be the same.

            <br/>

            IMPORTANT NOTE: inputs must follow the width, height, and channels specified.
            the feed forward network input layer is calculated automatically.
            Please only specify the hidden and output layers for the neuron layers.
            */
            pub fn new(convolution_layers: &[(usize, usize, Initialization, Activation, usize, usize)],
                       input_width: usize, input_height: usize, input_channels: usize,
                       neuron_layers: &[usize], neuron_inits: &[Initialization],
                       neuron_activations: &[Activation], loss_func: Loss) -> Self {
                let mut neurons: Vec<usize> = (0..neuron_layers.len() + 1).map(|_| 0usize).collect();

                //set neurons
                for idx in 1..neurons.len() {
                    neurons[idx] = neuron_layers[idx - 1];
                }

                let mut network_layers = Vec::<ConvolutionalLayer>::new();

                //add the first layer as the layer for the inputs
                network_layers.push(ConvolutionalLayer {
                    kernel_layers_temp: vec![],
                    kernel_layers: vec![],
                    kernel_size: 0,
                    pooling_size: 0,
                    pooling_method: 0,
                    bias: vec![],
                    values: vec![],
                    results: vec![],
                    switch: vec![],
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
                    let conv_width = img_width - conv_layer.1 + 1;
                    let conv_height = img_height - conv_layer.1 + 1;

                    //add network layer
                    network_layers.push(ConvolutionalLayer::new(
                        channels, conv_width, conv_height,
                        conv_layer.0, conv_layer.1,
                        &conv_layer.2, conv_layer.3.clone(),
                        conv_layer.4, conv_layer.5
                    ));

                    //pooling
                    img_width = conv_width / conv_layer.4;
                    img_height = conv_height / conv_layer.4;

                    //multiply the channel
                    channels = conv_layer.0;
                }
                let input_layer_size = img_width * img_height * channels;
                neurons[0] = input_layer_size;

                //create fully connected network
                let fully_connected_network =
                    Network::new(&neurons, &neuron_inits,
                                 &neuron_activations, loss_func);

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
            pub fn calculate(&mut self, inputs: &[Matrix]) -> Vec<f32> {
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

            pub fn learn(&mut self, learning_rate: f32, expected: &[f32]) {
                //feed forward network learns
                self.network.learn(learning_rate, expected);
                let input_layer_err_terms = self.network.input_layer_err_term_bpg_mse();

                //get the error terms of the output convolutional layer
                {
                    let layer_length = self.layers.len();
                    let output_layer = &mut self.layers[layer_length - 1];

                    //calculate error terms
                    {
                        //learn the error terms first
                        for value_layer_idx in 0..output_layer.values.len() {
                            let pooled = &output_layer.pooleds[value_layer_idx];
                            let value = &output_layer.values[value_layer_idx];
                            let switch = &output_layer.switch[value_layer_idx];
                            let pooled_offset = value_layer_idx * pooled.values.len();

                            let mut x_loc = 0usize;
                            let mut y_loc = 0usize;
                            let mut pool_x = 0usize;
                            let mut pool_y = 0usize;

                            let mut error = 0.0f32;

                            while x_loc + output_layer.pooling_size <= value.w {
                                while y_loc + output_layer.pooling_size <= value.h {
                                    let mut error_in_pool = 0.0;

                                    for x in x_loc..(x_loc + output_layer.pooling_size) {
                                        'y_loop: for y in y_loc..(y_loc + output_layer.pooling_size) {
                                            //get switch
                                            let switch_val = switch.get(x, y);
                                            if switch_val == 0.0 {
                                                continue 'y_loop;
                                            }

                                            //calculate derivative
                                            let derivative = output_layer.activation
                                                .derivative(value.get(x, y));

                                            //add the error
                                            error_in_pool += switch_val * derivative;
                                        }
                                    }

                                    //get error term
                                    let err_index = pooled_offset +
                                        pooled.index_to_one_d(pool_x, pool_y);
                                    let err_term = input_layer_err_terms[err_index];
                                    error += err_term * error_in_pool;

                                    y_loc += output_layer.pooling_size;
                                    pool_y += 1;
                                }
                                x_loc += output_layer.pooling_size;
                                pool_x += 1;

                                y_loc = 0;
                                pool_y = 0;
                            }

                            //set the error
                            output_layer.error_terms[value_layer_idx] = error;

                            //update the bias
                            output_layer.bias[value_layer_idx] -= learning_rate * error;
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
                            let switches = &this_layer.switch[value_layer_idx];
                            let values = &this_layer.values[value_layer_idx];

                            let mut err_term = 0f32;

                            //go through all kernels related to the pooled matrix of this layer
                            for kernel_layer_index in 0..next_layer.kernel_layers.len() {
                                let kernel_layer = &next_layer.kernel_layers[kernel_layer_index];
                                let kernel = &kernel_layer[value_layer_idx];

                                let mut error = 0f32;

                                for kx in 0..kernel.w {
                                    for ky in 0..kernel.h {
                                        let mut total_value = 0.0f32;

                                        //total value of the value matrix of this layer related
                                        //to this kernel's weight
                                        for px in 0..this_layer.pooling_size {
                                            'yloop: for py in 0..this_layer.pooling_size {
                                                let x_loc = kx * this_layer.pooling_size + px;
                                                let y_loc = ky * this_layer.pooling_size + py;

                                                let switch = switches.get(x_loc, y_loc);
                                                if switch == 0.0 {
                                                    continue 'yloop;
                                                }

                                                total_value += switch * this_layer.activation
                                                    .derivative(values.get(x_loc, y_loc));
                                            }
                                        }

                                        //add the error
                                        error += total_value * kernel.get(kx, ky);
                                    }
                                }

                                //multiply with the error term
                                err_term += error * next_layer.error_terms[kernel_layer_index];
                            }

                            this_layer.error_terms[value_layer_idx] = err_term;

                            this_layer.bias[value_layer_idx] -= learning_rate * err_term;
                        }
                    }
                }

                //calculate the change in weight of all the convolutional layers
                for layer_index in 1..self.layers.len() {
                    let (left_layers, right_layers) =
                        self.layers.split_at_mut(layer_index);
                    let this_layer = &mut right_layers[0];
                    let prev_layer = &mut left_layers[left_layers.len() - 1];

                    //loop through all the kernels
                    for kernel_layer_idx in 0..this_layer.kernel_layers.len() {
                        let kernels = &mut this_layer.kernel_layers[kernel_layer_idx];
                        let kernels_temp = &mut this_layer.kernel_layers_temp[kernel_layer_idx];
                        let value = &mut this_layer.values[kernel_layer_idx];
                        let error_term = this_layer.error_terms[kernel_layer_idx];

                        //update kernel weights
                        for kernel_idx in 0..kernels.len() {
                            let input = &prev_layer.pooleds[kernel_idx];
                            let kernel = &mut kernels[kernel_idx];
                            let kernel_temp = &mut kernels_temp[kernel_idx];

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

                                    kernel_temp.set(ker_x, ker_y, kernel.get(ker_x, ker_y) - learning_rate * gradient);
                                }
                            }
                        }
                    }
                }

                //copy the cache kernel layers to the actual layers
                for conv_layer in &mut self.layers {
                    for layer_idx in 0..conv_layer.kernel_layers_temp.len() {
                        let kernels = &mut conv_layer.kernel_layers[layer_idx];
                        let kernels_temp = &mut conv_layer.kernel_layers_temp[layer_idx];

                        for kernel_idx in 0..kernels_temp.len() {
                            let kernel = &mut kernels[kernel_idx];
                            let kernel_temp = &kernels_temp[kernel_idx];
                            for idx in 0..kernel.values.len() {
                                let mut value = kernel_temp.values[idx];
                                if value.abs() > 1f32 {
                                    value = value.signum() * 0.001f32;
                                }
                            }
                        }

                        let mut bias = conv_layer.bias[layer_idx];
                        if bias.abs() > 1f32 {
                            bias = bias.signum() * 0.5f32;
                        }
                        conv_layer.bias[layer_idx] = bias;
                    }
                }
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
                    let mut img = bmp::Image::new(value.w as u32, value.h as u32);
                    for x in 0..value.w {
                        for y in 0..value.h {
                            let l = max(0.0, value.get(x, y) * 255.0) as u8;
                            img.set_pixel(x as u32, y as u32, Pixel::new(l, l, l));
                        }
                    }
                    img.save(format!("{dir}\\{layer_idx}-{value_idx}val.bmp")).unwrap();
                    value_idx += 1;
                }
                for pooled in &layer.pooleds {
                    let mut img = bmp::Image::new(pooled.w as u32, pooled.h as u32);
                    for x in 0..pooled.w {
                        for y in 0..pooled.h {
                            let l = (pooled.get(x, y) * 255.0) as u8;
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

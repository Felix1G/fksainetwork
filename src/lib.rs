mod activation;
mod neuron;
mod tests;
mod util;
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
}

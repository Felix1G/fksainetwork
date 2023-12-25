use bincode_derive::{Decode, Encode};
use rand::{self, distributions::{Distribution, Uniform}};
use crate::activation::{Activation};
use crate::initialization::Initialization;
use crate::pooling::{pooling, pooling_switch};
use crate::util::Matrix;

#[derive(Encode, Decode, PartialEq, Debug)]
pub struct Neuron {
    //weights
    pub(crate) weights_temp: Vec<f32>, //cache updated weights so that they won't affect backpropogation
    pub(crate) weights: Vec<f32>,

    //bias
    pub(crate) bias: f32,

    pub(crate) value: f32, //value of calculation
    pub(crate) result: f32, //value after activation

    pub(crate) activation: Activation,

    pub(crate) error_term: f32
}

impl Neuron {
    pub fn new(weights: usize, number_of_neurons: usize,
               activation: Activation, init: &Initialization) -> Neuron {
        let mut rng = rand::thread_rng();
        let range = Uniform::new_inclusive(-0.005f32, 0.005f32);

        let weights: Vec<f32> = init.init(&mut rng, weights, number_of_neurons, weights);

        Neuron {
            weights_temp: weights.clone(),
            weights,
            bias: range.sample(&mut rng),
            value: 0.0,
            result: 0.0,
            activation,
            error_term: 0.0
        }
    }

    pub fn calculate(&mut self, input: &Vec<f32>) {
        let mut value: f32 = 0.0;
        for index in 0..self.weights.len() {
            value += self.weights[index] * input[index];
        }

        value += self.bias;

        self.value = value;
        self.result = self.activation.activate(value);
    }
}

#[derive(Encode, Decode, PartialEq, Debug)]
pub struct ConvolutionalLayer {
    //kernels
    pub(crate) kernel_layers_temp: Vec<Vec<Matrix>>, //cache updated kernel layers so it doesn't affect backpropogation
    pub(crate) kernel_layers: Vec<Vec<Matrix>>, //layers of kernels
    pub(crate) kernel_size: usize,

    //pooling
    pub(crate) pooling_size: usize,
    pub(crate) pooling_method: usize,

    //bias
    pub(crate) bias: Vec<f32>,

    //calculated values storage
    pub(crate) values: Vec<Matrix>, //value of calculation (product + bias)
    pub(crate) results: Vec<Matrix>, //value of values after activation
    pub(crate) pooleds: Vec<Matrix>, //value after pooling
    pub(crate) switch: Vec<Matrix>, //used for the convolution derivation

    //error terms
    pub(crate) error_terms: Vec<f32>, //error term

    //activation
    pub(crate) activation: Activation,

    //cache variables
    pub(crate) temp_matrix: Matrix, //used for convolution calculations
    pub(crate) temp_pooling_arr: Vec<f32>
}

impl ConvolutionalLayer {
    //input width and height AFTER convolution
    pub fn new(prev_channels: usize, input_width: usize, input_height: usize,
               kernels: usize, kernel_size: usize, init: &Initialization, activation: Activation,
               pooling_size: usize, pooling_method: usize) -> Self {
        let mut rng = rand::thread_rng();
        let bias_range = Uniform::new_inclusive(-0.005f32, 0.005f32);

        let kernel_size_2 = kernel_size * kernel_size;
        let prev_input_size = input_width * input_height;

        let mut kernel_layers_temp: Vec<Vec<Matrix>> = Vec::<Vec<Matrix>>::with_capacity(kernels);

        let kernel_layers: Vec<Vec<Matrix>> = (0..kernels).map(|_| {
            //create the vector for the temp layers
            let mut vec_temp = Vec::<Matrix>::with_capacity(prev_channels);

            //create the actual vector
            let vec: Vec<Matrix> = (0..prev_channels).map(|_| {
                //the kernel matrix
                let matrix = Matrix {
                    w: kernel_size,
                    h: kernel_size,
                    values: init.init(&mut rng, kernels, kernel_size_2, kernel_size_2),
                };

                //add a clone to the cache vector
                vec_temp.push(matrix.clone());

                //return the matrix
                matrix
            }).collect();

            //add the cache vector
            kernel_layers_temp.push(vec_temp);

            //return the actual vector
            vec
        }).collect();

        return ConvolutionalLayer {
            kernel_layers_temp,
            kernel_layers,
            kernel_size,

            pooling_size,
            pooling_method,

            bias: (0..kernels).map(|_| bias_range.sample(&mut rng)).collect(),

            values: (0..kernels).map(|_|
                Matrix {
                    w: input_width,
                    h: input_height,
                    values: vec![0.0f32; prev_input_size],
                }
            ).collect(),
            results: (0..kernels).map(|_|
                Matrix {
                    w: input_width,
                    h: input_height,
                    values: vec![0.0f32; prev_input_size],
                }
            ).collect(),
            pooleds: (0..kernels).map(|_|
                Matrix {
                    w: input_width / pooling_size,
                    h: input_height / pooling_size,
                    values: vec![0.0f32; prev_input_size / pooling_size / pooling_size],
                }
            ).collect(),
            switch: (0..kernels).map(|_|
                Matrix {
                    w: input_width,
                    h: input_height,
                    values: vec![0.0f32; prev_input_size],
                }
            ).collect(),

            error_terms: vec![0f32; kernels],

            activation,

            temp_matrix: Matrix {
                w: kernel_size,
                h: kernel_size,
                values: (0..kernel_size_2).map(|_| 0.0).collect()
            },
            temp_pooling_arr: vec![0f32; pooling_size * pooling_size]
        }
    }

    pub(crate) fn calculate(&mut self, prev_layer: &ConvolutionalLayer) {
        let temp = &mut self.temp_matrix;

        //go to each kernel channels
        for kernel_layer_idx in 0..self.kernel_layers.len() {
            let kernel_layer = &self.kernel_layers[kernel_layer_idx];

            //loop through inputs
            for input_idx in 0..kernel_layer.len() {
                //get input matrix
                let input = &prev_layer.pooleds[input_idx];

                //loop through the slices
                for x in 0..(input.w - self.kernel_size) {
                    for y in 0..(input.h - self.kernel_size) {
                        //get kernel for the input layer
                        let kernel = &kernel_layer[input_idx];

                        //copy the slice data
                        for x_loc in 0..(self.kernel_size) {
                            for y_loc in 0..(self.kernel_size) {
                                temp.set(x_loc, y_loc, input.get(x_loc + x, y_loc + y));
                            }
                        }

                        //calculate
                        let value = temp.convolution(kernel);
                        let value_matrix = &mut self.values[kernel_layer_idx];
                        value_matrix.set(x, y, value + self.bias[kernel_layer_idx]);
                    }
                }
            }
        }

        //activating
        for idx in 0..self.results.len() {
            let value = &self.values[idx];
            let result = &mut self.results[idx];

            for index in 0..result.values.len() {
                result.values[index] = self.activation.activate(value.values[index]);
            }
        }

        //pooling
        for idx in 0..self.pooleds.len() {
            let result = &self.results[idx];
            let pooled = &mut self.pooleds[idx];
            let switch = &mut self.switch[idx];

            let mut x = 0usize;
            let mut y = 0usize;
            let mut pooled_x = 0usize;
            let mut pooled_y = 0usize;

            while y < result.h {
                while x < result.w {
                    //clear temporary array
                    self.temp_pooling_arr.clear();

                    //get the values to pool
                    for x_loc in x..(x + self.pooling_size) {
                        for y_loc in y..(y + self.pooling_size) {
                            let value = result.get(x_loc, y_loc);
                            self.temp_pooling_arr.push(value);
                        }
                    }

                    //pooling
                    let pooled_val = pooling(self.pooling_method, &self.temp_pooling_arr);
                    pooled.set(pooled_x, pooled_y, pooled_val);

                    //set the switch value
                    for x_loc in x..(x + self.pooling_size) {
                        for y_loc in y..(y + self.pooling_size) {
                            let value = result.get(x_loc, y_loc);
                            switch.set(x_loc, y_loc, pooling_switch(
                                self.pooling_method, &self.temp_pooling_arr, value));
                        }
                    }

                    //increment
                    x += self.pooling_size;
                    pooled_x += 1;
                }
                //increment
                y += self.pooling_size;
                pooled_y += 1;
                x = 0;
                pooled_x = 0;
            }
        }
    }
}
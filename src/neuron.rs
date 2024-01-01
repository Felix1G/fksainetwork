use bincode_derive::{Decode, Encode};
use rand::{self, Rng};
use crate::activation::{Activation};
use crate::initialization::Initialization;
use crate::pooling::{Pooling};
use crate::util::Matrix;

#[derive(Encode, Decode, PartialEq, Debug)]
pub struct Neuron {
    //weights
    pub(crate) weights: Vec<f32>,

    //bias
    pub(crate) bias: f32,

    //the total gradient of each weight and bias in a mini-batch
    pub(crate) weights_grad: Vec<f32>,
    pub(crate) bias_grad: f32,

    pub(crate) raw: f32, //value of calculation
    pub(crate) value: f32, //value after normalization (if not applicable, is equal to raw)
    pub(crate) result: f32, //value after activation

    pub(crate) activation: Activation,

    pub(crate) error_term: f32
}

impl Neuron {
    pub fn new(weights: usize, parameters: &(usize, Initialization, Activation, bool)) -> Neuron {
        let mut rng = rand::thread_rng();

        Neuron {
            weights: parameters.1.init(&mut rng, weights, parameters.0, weights),
            bias: rng.gen_range(-0.005f32..0.005f32),
            weights_grad: vec![0f32; weights],
            bias_grad: 0f32,
            raw: 0.0,
            value: 0.0,
            result: 0.0,
            activation: parameters.2,
            error_term: 0.0
        }
    }
}

#[derive(Encode, Decode, PartialEq, Debug)]
pub struct ConvolutionalLayer {
    //kernels
    pub(crate) kernel_layers: Vec<Vec<Matrix>>, //layers of kernels
    pub(crate) kernel_size: usize,

    //pooling
    pub(crate) pooling_size: usize,
    pub(crate) pooling_method: Pooling,

    //bias
    pub(crate) bias: Vec<f32>,

    //calculated values storage
    pub(crate) values: Vec<Matrix>, //value of calculation (product + bias)
    pub(crate) results: Vec<Matrix>, //value of values after activation
    pub(crate) pooleds_raw: Vec<Matrix>, //value after pooling
    pub(crate) pooleds: Vec<Matrix>, //pooling value after normalization if applicable
    pub(crate) switch: Vec<Matrix>, //used for the convolution derivation

    //value of values after the activation derivative multiplied by switch,
    //it is to prevent multiple calculations on the same value,
    //this is calculated during learning
    pub(crate) learning_val: Vec<Matrix>,
    pub(crate) kernel_grad: Vec<Vec<Matrix>>,
    pub(crate) bias_grad: Vec<f32>,

    //error terms
    pub(crate) error_terms: Vec<f32>, //error term

    //activation
    pub(crate) activation: Activation,

    //nomalize
    pub(crate) normalize: bool,
    pub(crate) normalize_data: Vec<(f32, f32)>, //gamma, beta
    pub(crate) normalize_grad: Vec<(f32, f32)>, //gamma, beta

    //cache variables
    pub(crate) temp_matrix: Matrix, //used for convolution calculations
    pub(crate) temp_pooling_arr: Vec<f32>
}

impl ConvolutionalLayer {
    //input width and height AFTER convolution
    pub fn new(prev_channels: usize, input_width: usize, input_height: usize,
               kernel_size: usize, kernel_inits: &[Initialization], activation: Activation,
               pooling_size: usize, pooling_method: Pooling, batch_normalization: bool) -> Self {
        let mut rng = rand::thread_rng();

        let filter_amount = kernel_inits.len();
        let kernel_size_2 = kernel_size * kernel_size;

        let mut kernel_layers_temp: Vec<Vec<Matrix>> = Vec::<Vec<Matrix>>::with_capacity(filter_amount);

        let kernel_layers: Vec<Vec<Matrix>> = (0..filter_amount).map(|_| {
            //create the vector for the temp layers
            let mut vec_temp = Vec::<Matrix>::with_capacity(prev_channels);

            //create the actual vector
            let vec: Vec<Matrix> = (0..prev_channels).map(| idx | {
                //the kernel matrix
                let matrix = Matrix {
                    w: kernel_size,
                    h: kernel_size,
                    values: kernel_inits[idx].init(&mut rng, filter_amount, kernel_size_2, kernel_size_2),
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
            kernel_layers,
            kernel_size,

            pooling_size,
            pooling_method,

            bias: (0..filter_amount).map(|_| rng.gen_range(-0.005f32..0.005f32)).collect(),

            values: (0..filter_amount).map(|_| Matrix::new(input_width, input_height)).collect(),
            results: (0..filter_amount).map(|_| Matrix::new(input_width, input_height)).collect(),
            learning_val: (0..filter_amount).map(|_| Matrix::new(input_width, input_height)).collect(),
            pooleds_raw: (0..filter_amount).map(|_|
                Matrix::new(
                    input_width / pooling_size,
                    input_height / pooling_size)
            ).collect(),
            pooleds: (0..filter_amount).map(|_|
                Matrix::new(
                    input_width / pooling_size,
                    input_height / pooling_size)
            ).collect(),
            switch: (0..filter_amount).map(|_| Matrix::new(input_width, input_height)).collect(),

            kernel_grad:  (0..filter_amount).map(|_| {
                (0..prev_channels).map(|_| {
                    Matrix::new(kernel_size, kernel_size)
                }).collect()
            }).collect(),
            bias_grad: (0..filter_amount).map(|_| 0.0f32).collect(),

            error_terms: vec![0f32; filter_amount],

            activation,

            normalize: batch_normalization,
            normalize_data: vec![(1f32, 0f32); filter_amount],
            normalize_grad: vec![(0f32, 0f32); filter_amount],

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
            let pooled_raw = &mut self.pooleds_raw[idx];
            let pooled = &mut self.pooleds[idx];
            let switch = &mut self.switch[idx];

            let normalize_data = &self.normalize_data[idx];

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
                    let pooled_val = self.pooling_method.pooling(&self.temp_pooling_arr);
                    pooled_raw.set(pooled_x, pooled_y, pooled_val);

                    if !self.normalize {
                        pooled.set(pooled_x, pooled_y, pooled_val);
                    }

                    //set the switch value
                    for x_loc in x..(x + self.pooling_size) {
                        for y_loc in y..(y + self.pooling_size) {
                            let value = result.get(x_loc, y_loc);
                            switch.set(x_loc, y_loc, self.pooling_method.pooling_switch(
                                &self.temp_pooling_arr, value));
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

                //normalize if applicable
                if self.normalize {
                    let len_inv = 1.0 / pooled.values.len() as f32;

                    //calculate mean
                    let mut total_value = 0f32;
                    pooled.values.iter().for_each(|x| total_value += x);
                    let mean = total_value * len_inv;

                    //calculate variance
                    let mut variance = 0f32;
                    pooled.values.iter().for_each(|x| {
                        let diff = x - mean;
                        variance += diff * diff;
                    });
                    variance *= len_inv;

                    //standard deviation
                    let std_inv = 1.0 / (variance + 0.00001f32).sqrt();

                    //set value
                    for index in 0..pooled.values.len() {
                        let raw = pooled_raw.values[index];
                        let value = (raw - mean) * std_inv;
                        pooled.values[index] = normalize_data.0 * value + normalize_data.1;
                    }
                }
            }
        }
    }
}
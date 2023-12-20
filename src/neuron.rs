use bincode_derive::{Decode, Encode};
use rand::{self, distributions::{Distribution, Uniform}};
use crate::activation::{activate};
use crate::pooling::pooling;
use crate::util::Matrix;

#[derive(Encode, Decode, PartialEq, Debug)]
pub struct Neuron {
    //temporary variables act as caches
    //weights
    pub(crate) weights_temp: Vec<f32>,
    pub(crate) weights: Vec<f32>,

    //bias
    pub(crate) bias_temp: f32,
    pub(crate) bias: f32,

    pub(crate) value: f32, //value of calculation
    pub(crate) result: f32, //value after activation

    pub(crate) activation: usize
}

impl Neuron {
    pub fn new(weights: usize, activation: usize) -> Neuron {
        let mut rng = rand::thread_rng();
        let range = Uniform::new_inclusive(-0.005f32, 0.005f32);

        Neuron {
            weights_temp: (0..weights).map(|_| 0.0).collect(),
            weights: (0..weights).map(|_| range.sample(&mut rng)).collect(),
            bias_temp: 0.0,
            bias: range.sample(&mut rng),
            value: 0.0,
            result: 0.0,
            activation
        }
    }

    pub fn calculate(&mut self, input: &Vec<f32>) {
        let mut value: f32 = 0.0;
        for index in 0..self.weights.len() {
            value += self.weights[index] * input[index];
        }

        value += self.bias;

        self.value = value;
        self.result = activate(self.activation, value);
    }
}

#[derive(Debug)]
pub struct ConvolutionalLayer {
    //temporary variables act as caches
    //weights
    pub(crate) kernels_temp: Vec<Matrix>,
    pub(crate) kernels_layers: Vec<Matrix>,
    pub(crate) kernels: usize,
    pub(crate) kernel_size: usize,

    pub(crate) pooling_size: usize,
    pub(crate) pooling_method: usize,

    //bias
    pub(crate) bias_temp: f32,
    pub(crate) bias: f32,

    pub(crate) value: Vec<Matrix>, //value of calculation
    pub(crate) result: Vec<Matrix>, //value after activation
    pub(crate) pooled: Vec<Matrix>, //value after pooling

    pub(crate) activation: usize,

    pub(crate) temp_matrix: Matrix, //used for convolution calculations
    pub(crate) temp_pooling_arr: Vec<f32>
}

impl ConvolutionalLayer {
    //input width and height AFTER convolution
    pub fn new(prev_channels: usize, input_width: usize, input_height: usize,
               kernels: usize, kernel_size: usize, activation: usize,
               pooling_size: usize, pooling_method: usize) -> Self {
        let mut rng = rand::thread_rng();
        let range = Uniform::new_inclusive(-0.005f32, 0.005f32);

        let kernel_size_2 = kernel_size * kernel_size;
        let prev_input_size = input_width * input_height;
        let final_channel_count = kernels * prev_channels;

        return ConvolutionalLayer {
            kernels_temp: (0..final_channel_count).map(|_|
                Matrix {
                    w: kernel_size,
                    h: kernel_size,
                    values: vec![0.0f32; kernel_size_2],
                }
            ).collect(),
            kernels_layers: (0..final_channel_count).map(|_|
                Matrix {
                    w: kernel_size,
                    h: kernel_size,
                    values: (0..kernel_size_2).map(|_| range.sample(&mut rng)).collect()
                }
            ).collect(),
            kernels,
            kernel_size,

            pooling_size,
            pooling_method,

            bias_temp: 0.0,
            bias: range.sample(&mut rng),

            value: (0..kernels).map(|_|
                Matrix {
                    w: input_width,
                    h: input_height,
                    values: vec![0.0f32; prev_input_size],
                }
            ).collect(),
            result: (0..kernels).map(|_|
                Matrix {
                    w: input_width,
                    h: input_height,
                    values: vec![0.0f32; prev_input_size],
                }
            ).collect(),
            pooled: (0..kernels).map(|_|
                Matrix {
                    w: input_width / pooling_size,
                    h: input_height / pooling_size,
                    values: vec![0.0f32; prev_input_size / pooling_size / pooling_size],
                }
            ).collect(),

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
        let input_size = prev_layer.pooled.len();

        //go to each kernel channels
        for kernel_index in 0..self.kernels {
            let kernel_index_addition = kernel_index * input_size;

            //loop through inputs
            for input_idx in 0..input_size {
                //get input matrix
                let input = &prev_layer.pooled[input_idx];

                //loop through the slices
                for x in 0..(input.w - self.kernel_size) {
                    for y in 0..(input.h - self.kernel_size) {
                        //get kernel for the input layer
                        let kernel = &self.kernels_layers[kernel_index_addition + input_idx];

                        //copy the slice data
                        for x_loc in 0..(self.kernel_size) {
                            for y_loc in 0..(self.kernel_size) {
                                temp.values[y_loc * temp.w + x_loc] =
                                    input.values[(y_loc + y) * input.w + (x_loc + x)];
                            }
                        }

                        //calculate
                        let value = temp.convolution(kernel);
                        let value_matrix = &mut self.value[kernel_index];
                        value_matrix.values[y * value_matrix.w + x] = value + self.bias;
                    }
                }
            }
        }

        //calculate activation
        for idx in 0..self.value.len() {
            let value_vec = &self.value[idx];
            let result_vec = &mut self.result[idx];
            for index in 0..result_vec.values.len() {
                result_vec.values[index] = activate(self.activation, value_vec.values[index]);
            }
        }

        //pooling
        for idx in 0..self.result.len() {
            let result = &self.result[idx];
            let pooled = &mut self.pooled[idx];

            let result_width = result.w;
            let result_height = result.h;

            let mut x = 0usize;
            let mut y = 0usize;
            let mut pooled_index = 0usize;


            while y < result_height {
                while x < result_width {
                    self.temp_pooling_arr.clear();

                    for x_loc in x..(x + self.pooling_size) {
                        for y_loc in y..(y + self.pooling_size) {
                            let value = result.values[y_loc * result.w + x_loc];
                            self.temp_pooling_arr.push(value);
                        }
                    }

                    pooled.values[pooled_index] = pooling(self.pooling_method, &self.temp_pooling_arr);

                    x += self.pooling_size;
                    pooled_index += 1;
                }
                y += self.pooling_size;
            }
        }
    }
}
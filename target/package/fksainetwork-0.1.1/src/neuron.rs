use bincode_derive::{Decode, Encode};
use rand::{self, distributions::{Distribution, Uniform}};
use crate::activation::activations;

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
        self.result = activations(self.activation).activate(value);
    }
}
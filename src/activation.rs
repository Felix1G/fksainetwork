use bincode_derive::{Decode, Encode};
use crate::util::max;

#[derive(Encode, Decode, PartialEq, Debug, Clone, Copy)]
pub enum Activation {
    Linear,
    Sigmoid,
    ReLU,
    Tanh,
    LeakyReLU
}

impl Activation {
    pub(crate) fn activate(&self, value: f32) -> f32 {
        return match self {
            Activation::Linear => value,
            Activation::Sigmoid => 1.0 / (1.0 + (-value).exp()),
            Activation::ReLU => max(0.0, value),
            Activation::Tanh => {
                let exp = value.exp();
                let expm = (-value).exp();
                (exp - expm) / (exp + expm)
            }
            Activation::LeakyReLU => if value > 0.0 { value } else { 0.1 * value },
        }
    }

    pub(crate) fn derivative(&self, value: f32) -> f32 {
        return match self {
            Activation::Linear => 1.0,
            Activation::Sigmoid => {
                let result = self.activate(value);
                result * (1.0 - result)
            },
            Activation::ReLU => {
                if value <= 0.0 {
                    0.0
                } else {
                    1.0
                }
            },
            Activation::Tanh => {
                let result = self.activate(value);
                1.0 - result * result
            },
            Activation::LeakyReLU => {
                if value <= 0.0 {
                    0.1
                } else {
                    1.0
                }
            },
        }
    }
}
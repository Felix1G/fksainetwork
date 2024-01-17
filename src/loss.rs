use std::cmp::min;
use bincode_derive::{Decode, Encode};
use crate::util::max;

const EPSILON: f32 = 1000f32;

#[derive(Encode, Decode, PartialEq, Debug)]
pub enum Loss {
    MeanSquaredError,
    BinaryCrossEntropy
}

impl Loss {
    pub(crate) fn loss(&self, result: &[f32], expected: &[f32]) -> f32 {
        let error = match self {
            Loss::MeanSquaredError => {
                let mut error = 0.0;

                for idx in 0..result.len() {
                    let r = result[idx];
                    let e = expected[idx];
                    let err = (r - e) * (r - e);
                    error += err;
                }

                error
            },
            Loss::BinaryCrossEntropy => {
                let mut error = 0.0;

                for idx in 0..result.len() {
                    let r = result[idx];
                    let e = expected[idx];

                    error += (1.0 - e) * (1.0 - r).ln() + e * r.ln();
                }

                -error
            }
        };

        return error / result.len() as f32;
    }

    pub(crate) fn loss_derivative(&self, result: f32, expected_index: usize, expected: &[f32]) -> f32 {
        let error = match self {
            Loss::MeanSquaredError => 2.0 * (result - expected[expected_index]),
            Loss::BinaryCrossEntropy => {
                let target = expected[expected_index];
                (result - target) / ((1.0 - result) * result)
            }
        };

        let err = if error < 0.0f32 {
            f32::max(-EPSILON, error)
        } else {
            f32::min(EPSILON, error)
        };

        err
    }
}
use bincode_derive::{Decode, Encode};

#[derive(Encode, Decode, PartialEq, Debug)]
pub enum Loss {
    MeanSquared,
    BinaryCrossEntropy
}

impl Loss {
    pub(crate) fn loss_derivative(&self, result: f32, expected_index: usize, expected: &[f32]) -> f32 {
        match self {
            Loss::MeanSquared =>  2.0 * (result - expected[expected_index]) / expected.len() as f32,
            Loss::BinaryCrossEntropy => {
                let target = expected[expected_index];
                -( ( target / result ) - ( (1.0 - target) / (1.0 - result) ) )
            }
        }
    }
}
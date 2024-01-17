use bincode_derive::{Decode, Encode};

#[derive(Encode, Decode, PartialEq, Debug, Clone)]
pub enum Pooling {
    Max,
    Average
}

impl Pooling {
    pub(crate) fn pooling(&self, values: &[f32]) -> f32 {
        return match self {
            Pooling::Max => {
                let mut val = 0f32;
                for value in values {
                    let v = *value;
                    if v > val {
                        val = v;
                    }
                }
                val
            }
            Pooling::Average => {
                let mut total = 0f32;
                for value in values {
                    total += value;
                }
                total / values.len() as f32
            }
        }
    }

    pub(crate) fn pooling_switch(&self, values: &[f32], expected: f32) -> f32 {
        return match self {
            Pooling::Max => {
                if self.pooling(values) == expected {
                    1.0
                } else {
                    0.0
                }
            }
            Pooling::Average => {
                1.0 / values.len() as f32
            }
        }
    }
}
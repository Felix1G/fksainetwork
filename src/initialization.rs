use bincode_derive::{Decode, Encode};
use rand::{Rng, thread_rng};
use rand::rngs::ThreadRng;

#[derive(Encode, Decode, PartialEq, Debug, Clone, Copy)]
pub enum Initialization {
    Xavier,
    He
}

impl Initialization {
    pub(crate) fn init(&self, rng: &mut ThreadRng, fan_in: usize, fan_out: usize, size: usize) -> Vec<f32> {
        match self {
            Initialization::Xavier => {
                let deviation = (2.0f32 / fan_in as f32).sqrt();
                let mut vec = vec![0.0f32; size];

                for idx in 0..vec.len() {
                    vec[idx] = rng.gen_range(-deviation..deviation);
                }

                vec
            },
            Initialization::He => {
                let deviation = (2.0 / (fan_in + fan_out) as f32).sqrt();
                let mut vec = vec![0.0f32; size];

                for idx in 0..vec.len() {
                    vec[idx] = rng.gen_range(-deviation..deviation);
                }

                vec
            }
        }
    }
}
use bincode_derive::{Decode, Encode};
use rand::Rng;
use rand::rngs::ThreadRng;

#[derive(Encode, Decode, PartialEq, Debug, Clone, Copy)]
pub enum Initialization {
    Xavier,
    He,
    LeftKernel,
    TopKernel,
    RightKernel,
    BottomKernel
}

impl Initialization {
    pub(crate) fn init(&self, rng: &mut ThreadRng, fan_in: usize, fan_out: usize, size: usize) -> Vec<f32> {
        match self {
            Initialization::Xavier => {
                let deviation = (6.0 / fan_in as f32).sqrt();
                let mut vec = vec![0.0f32; size];

                for idx in 0..vec.len() {
                    vec[idx] = rng.gen_range(-deviation..deviation);
                }

                vec
            },
            Initialization::He => {
                let deviation = (6.0 / (fan_in + fan_out) as f32).sqrt();
                let mut vec = vec![0.0f32; size];

                for idx in 0..vec.len() {
                    vec[idx] = rng.gen_range(-deviation..deviation);
                }

                vec
            },
            Initialization::LeftKernel | Initialization::RightKernel => {
                let delta: f32 = if Initialization::LeftKernel == *self { -1.0 } else { 1.0 };

                let kernel_size = (fan_out as f32).sqrt() as usize;
                let column_size = kernel_size / 3;

                let mut vec = vec![0.0f32; fan_out];

                for idx in 0..column_size {
                    for y in 0..kernel_size {
                        vec[y * kernel_size + idx] = delta;
                    }
                }

                for idx in (kernel_size - column_size)..kernel_size {
                    for y in 0..kernel_size {
                        vec[y * kernel_size + idx] = -delta;
                    }
                }

                vec
            }, Initialization::TopKernel | Initialization::BottomKernel => {
                let delta: f32 = if Initialization::TopKernel == *self { -1.0 } else { 1.0 };

                let kernel_size = (fan_out as f32).sqrt() as usize;
                let column_size = kernel_size / 3;

                let mut vec = vec![0.0f32; fan_out];

                for idx in 0..column_size {
                    for x in 0..kernel_size {
                        vec[idx * kernel_size + x] = delta;
                    }
                }

                for idx in (kernel_size - column_size)..kernel_size {
                    for x in 0..kernel_size {
                        vec[idx * kernel_size + x] = -delta;
                    }
                }

                vec
            }
        }
    }
}
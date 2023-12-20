use bincode_derive::{Encode, Decode};

#[derive(Encode, Decode, PartialEq, Debug)]
pub(crate) struct Matrix {
    pub(crate) w: usize,
    pub(crate) h: usize,
    pub(crate) values: Vec<f32>
}

impl Matrix {
    pub(crate) fn empty() -> Self {
        Matrix {
            w: 0,
            h: 0,
            values: Vec::new()
        }
    }

    pub(crate) fn convolution(&self, other: &Matrix) -> f32 {
        let mut value = 0.0f32;

        for x in 0..self.w {
            for y in 0..self.h {
                value += self.values[y * self.w + x] * other.values[y * self.w + x];
            }
        }

        return value;
    }

    pub(crate) fn copy(&mut self, other: &Matrix) {
        self.w = other.w;
        self.h = other.h;
        for i in 0..other.values.len() {
            self.values[i] = other.values[i];
        }
    }
}

pub fn max(val0: f32, val1: f32) -> f32 { if val0 > val1 { val0 } else { val1 } }
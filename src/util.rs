use bincode_derive::{Encode, Decode};

#[derive(Encode, Decode, PartialEq, Debug, Clone)]
pub(crate) struct Matrix {
    pub(crate) w: usize,
    pub(crate) h: usize,
    pub(crate) values: Vec<f32>
}

impl Matrix {
    pub fn new(w: usize, h: usize) -> Self {
        Matrix {
            w,
            h,
            values: vec![0f32; w * h]
        }
    }

    pub fn empty() -> Self {
        Matrix {
            w: 0,
            h: 0,
            values: Vec::new()
        }
    }

    pub fn convolution(&self, other: &Matrix) -> f32 {
        let mut value = 0.0f32;

        for x in 0..self.w {
            for y in 0..self.h {
                value += self.get(x, y) * other.get(x, y);
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

    pub(crate) fn get(&self, x: usize, y: usize) -> f32 {
        self.values[self.index_to_one_d(x, y)]
    }

    pub(crate) fn set(&mut self, x: usize, y: usize, v: f32) {
        let index = self.index_to_one_d(x, y);
        self.values[index] = v;
    }

    pub(crate) fn zero(&mut self) {
        self.values.iter_mut().for_each(|m| *m = 0.0);
    }

    pub(crate) fn index_to_one_d(&self, x: usize, y: usize) -> usize {
        y * self.w + x
    }
}

pub fn max(val0: f32, val1: f32) -> f32 { if val0 > val1 { val0 } else { val1 } }
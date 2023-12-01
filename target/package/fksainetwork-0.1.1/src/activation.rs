use crate::util;

pub trait Activation {
    fn activate(&self, _value: f32) -> f32 { 0.0 }
    fn derivative(&self, _value: f32) -> f32 { 0.0 }
}

struct LinearActivation;

struct SigmoidActivation;

struct ReLUActivation;

pub fn activations(index: usize) -> Box<dyn Activation> {
    return match index {
        0 => Box::new(LinearActivation),
        1 => Box::new(SigmoidActivation),
        2 => Box::new(ReLUActivation),
        _ => Box::new(LinearActivation)
    }
}

impl Activation for LinearActivation {
    fn activate(&self, value: f32) -> f32 { value }
    fn derivative(&self, _value: f32) -> f32 { 1.0 }
}

impl Activation for SigmoidActivation {
    fn activate(&self, value: f32) -> f32 { 1.0 / (1.0 + f32::exp(-value)) }
    fn derivative(&self, value: f32) -> f32 {
        let result = self.activate(value);
        result * (1.0 - result)
    }
}

impl Activation for ReLUActivation {
    fn activate(&self, value: f32) -> f32 { util::max(0.0, value) }
    fn derivative(&self, value: f32) -> f32 {
        if value <= 0.0 {
            0.0
        } else {
            1.0
        }
    }
}
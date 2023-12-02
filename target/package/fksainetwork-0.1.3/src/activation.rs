use crate::util;

trait Activation {
    fn activate(&self, _value: f32) -> f32 { 0.0 }
    fn derivative(&self, _value: f32) -> f32 { 0.0 }
}

struct LinearActivation;
struct SigmoidActivation;
struct ReLUActivation;
struct TanhActivation;

const LINEAR: LinearActivation = LinearActivation;
const SIGMOID: SigmoidActivation = SigmoidActivation;
const RELU: ReLUActivation = ReLUActivation;
const TANH: TanhActivation = TanhActivation;

pub fn activate(mode: usize, value: f32) -> f32 {
    return match mode {
        0 => LINEAR.activate(value),
        1 => SIGMOID.activate(value),
        2 => RELU.activate(value),
        3 => TANH.activate(value),
        _ => LINEAR.activate(value)
    }
}

pub fn derivative(mode: usize, value: f32) -> f32 {
    return match mode {
        0 => LINEAR.derivative(value),
        1 => SIGMOID.derivative(value),
        2 => RELU.derivative(value),
        3 => TANH.derivative(value),
        _ => LINEAR.derivative(value)
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

impl Activation for TanhActivation {
    fn activate(&self, value: f32) -> f32 {
        let exp = f32::exp(value);
        let expm = f32::exp(-value);
        (exp - expm) / (exp + expm)
    }
    fn derivative(&self, value: f32) -> f32 {
        let result = self.activate(value);
        1.0 - result * result
    }
}
use crate::util;

trait Activation {
    fn activate(_value: f32) -> f32 { 0.0 }
    fn derivative(_value: f32) -> f32 { 0.0 }
}

struct LinearActivation;
struct SigmoidActivation;
struct ReLUActivation;
struct TanhActivation;


pub fn activate(mode: usize, value: f32) -> f32 {
    return match mode {
        0 => LinearActivation::activate(value),
        1 => SigmoidActivation::activate(value),
        2 => ReLUActivation::activate(value),
        3 => TanhActivation::activate(value),
        _ => LinearActivation::activate(value)
    }
}

pub fn derivative(mode: usize, value: f32) -> f32 {
    return match mode {
        0 => LinearActivation::derivative(value),
        1 => SigmoidActivation::derivative(value),
        2 => ReLUActivation::derivative(value),
        3 => TanhActivation::derivative(value),
        _ => LinearActivation::derivative(value)
    }
}

impl Activation for LinearActivation {
    fn activate(value: f32) -> f32 { value }
    fn derivative(_value: f32) -> f32 { 1.0 }
}

impl Activation for SigmoidActivation {
    fn activate(value: f32) -> f32 { 1.0 / (1.0 + f32::exp(-value)) }
    fn derivative(value: f32) -> f32 {
        let result = SigmoidActivation::activate(value);
        result * (1.0 - result)
    }
}

impl Activation for ReLUActivation {
    fn activate(value: f32) -> f32 { util::max(0.0, value) }
    fn derivative(value: f32) -> f32 {
        if value <= 0.0 {
            0.0
        } else {
            1.0
        }
    }
}

impl Activation for TanhActivation {
    fn activate(value: f32) -> f32 {
        let exp = f32::exp(value);
        let expm = f32::exp(-value);
        (exp - expm) / (exp + expm)
    }
    fn derivative(value: f32) -> f32 {
        let result = TanhActivation::activate(value);
        1.0 - result * result
    }
}
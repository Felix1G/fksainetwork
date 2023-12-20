trait Pooling {
    fn pooling(values: &[f32]) -> f32 { values[0] }
}

struct MaxPooling;
struct AveragePooling;

pub fn pooling(mode: usize, values: &[f32]) -> f32 {
    return match mode {
        0 => MaxPooling::pooling(values),
        1 => AveragePooling::pooling(values),
        _ => MaxPooling::pooling(values)
    }
}

impl Pooling for MaxPooling {
    fn pooling(values: &[f32]) -> f32 {
        let mut val = 0f32;
        for value in values {
            let v = *value;
            if v > val {
                val = v;
            }
        }
        return val;
    }
}

impl Pooling for AveragePooling {
    fn pooling(values: &[f32]) -> f32 {
        let mut total = 0f32;
        for value in values {
            total += value;
        }
        return total / values.len() as f32;
    }
}
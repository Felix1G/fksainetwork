trait Pooling {
    fn pooling(values: &[f32]) -> f32 { values[0] }
    fn pooling_switch(values: &[f32], expected: f32) -> f32 { values[0] }
}

struct MaxPooling;
struct AveragePooling;

pub(crate) fn pooling(mode: usize, values: &[f32]) -> f32 {
    return match mode {
        0 => MaxPooling::pooling(values),
        1 => AveragePooling::pooling(values),
        _ => MaxPooling::pooling(values)
    }
}

pub(crate) fn pooling_switch(mode: usize, values: &[f32], expected: f32) -> f32 {
    return match mode {
        0 => MaxPooling::pooling_switch(values, expected),
        1 => AveragePooling::pooling_switch(values, expected),
        _ => MaxPooling::pooling_switch(values, expected)
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

    fn pooling_switch(values: &[f32], expected: f32) -> f32 {
        if MaxPooling::pooling(values) == expected {
            1.0
        } else {
            0.0
        }
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

    fn pooling_switch(values: &[f32], expected: f32) -> f32 {
        values.len() as f32
    }
}
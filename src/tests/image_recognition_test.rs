
#[cfg(test)]
mod image_recognition_test {
    use std::fs;
    use crate::network::cnn::{ConvolutionalNetwork, save_cnn_network};
    use crate::network::save_network;
    use crate::util::Matrix;

    #[test]
    fn main() {
        let mut network = ConvolutionalNetwork::new(
            &[
                (5, 5, 2, 2, 1),
                (10, 5, 2, 2, 1)
            ],
            32, 32, 1, &[10], &[1]
        );

        let mut samples = Vec::<([f32; 10], [f32; 32 * 32])>::new();
        let paths = fs::read_dir("C:/Users/ACER/RustroverProjects/fksainetwork-learns-numbers/samples").unwrap();
        for path in paths {
            let file = path.unwrap();
            let img = bmp::open(file.path()).unwrap();
            let num = file.file_name().as_encoded_bytes()[0] - 48;
            let mut pix_arr = [0f32; 1024];
            for x in 0u32..32u32 {
                for y in 0u32..32u32 {
                    pix_arr[(y * 32 + x) as usize] = img.get_pixel(x, y).r as f32 / 255f32;
                }
            }

            let mut ans_arr = [0f32; 10];
            ans_arr[num as usize] = 1.0;
            samples.push((ans_arr, pix_arr));
        }

        for sample in samples {
            println!("{:?}", network.calculate(&[Matrix {
                w: 32,
                h: 32,
                values: Vec::from(sample.1)
            }]));

            //println!("{:?}", network);
        }

        save_cnn_network("C:/Users/ACER/RustroverProjects/fksainetwork/networks/image_recognition.ai", &network);
    }
}

#[cfg(test)]
mod image_recognition_test {
    use crate::util::Matrix;
    use std::fs;
    use crate::network::cnn::{cnn_network_bmp, ConvolutionalNetwork, load_cnn_network, save_cnn_network};

    const PATH: &str = "C:\\Users\\ACER\\OneDrive\\Documents\\GitHub\\fksainetwork/networks/image_recognition.ai";

    #[test]
    fn main() {
        let mut network = ConvolutionalNetwork::new(
            &[
                (3, 3, 2, 6, 1),
                //(10, 5, 2, 2, 0)
            ],
            32, 32, 1, &[10], &[1]
        );
        //load_cnn_network(PATH);

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

        if true {
            for idx in 0..100000 {
                let index = ((idx % 10) * 50 + ((idx / 10) % 50)) % 500;
                let sample = samples[index];

                for _ in 0..20 {
                    let out = network.calculate(&[Matrix {
                        w: 32,
                        h: 32,
                        values: Vec::from(sample.1)
                    }]);

                    network.learn_bpg_mse(0.1, &sample.0);
                }

                if index == 499 {
                    save_cnn_network("C:\\Users\\ACER\\OneDrive\\Documents\\GitHub\\fksainetwork/networks/image_recognition.ai", &network);
                    cnn_network_bmp("C:/Users/ACER/OneDrive/Pictures/conv-ai-out", &network);
                    println!("Run: {idx}, {:?}", network.calculate(&[Matrix {
                        w: 32,
                        h: 32,
                        values: Vec::from(sample.1)
                    }]));
                }
            }
        }

        let sample_ = &samples[400..500];
        for sample in sample_ {
            let ans = network.calculate(&[Matrix {
                w: 32,
                h: 32,
                values: Vec::from(sample.1)
            }]);
            println!("{} {} {:?} {:?} \n",
                     sample.0
                         .iter()
                         .enumerate()
                         .max_by(|(_, a), (_, b)| a.total_cmp(b))
                         .map(|(index, _)| index)
                         .unwrap(),
                     ans
                         .iter()
                         .enumerate()
                         .max_by(|(_, a), (_, b)| a.total_cmp(b))
                         .map(|(index, _)| index)
                         .unwrap(), sample.0, ans);
        }

        //println!("{:?}", network);

        cnn_network_bmp("C:/Users/ACER/OneDrive/Pictures/conv-ai-out", &network);

        save_cnn_network(PATH, &network);
    }
}
#[cfg(test)]
mod image_recognition_test {
    use crate::util::Matrix;
    use std::fs;
    use crate::network::cnn::{cnn_network_bmp, ConvolutionalNetwork, load_cnn_network, save_cnn_network};

    const PATH: &str = "./networks/image_recognition.ai";

    fn most_predicted(arr: &[f32]) -> usize {
        arr.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index)
            .unwrap()
    }

    #[test]
    fn main() {
        let mut network = /*ConvolutionalNetwork::new(
            &[
                (6, 3, 0, 2, 2, 0),
                (15, 4, 0, 2, 2, 0)
            ],
            32, 32, 1, &[20, 10],
            &[0, 1], &[0, 1], 1
        );*/
        load_cnn_network(PATH);

        let mut samples = Vec::<([f32; 10], [f32; 32 * 32])>::new();
        let paths = fs::read_dir("./networks/num-dataset").unwrap();
        for path in paths {
            let file = path.unwrap();
            if file.file_type().unwrap().is_dir() {
                continue;
            }

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
            for idx in 0..10000000 {
                let index = ((idx % 10) * 50 + ((idx / 10) % 50)) % 500;
                let sample = samples[index];

                for i in 0..20 {
                    let out = network.calculate(&[Matrix {
                        w: 32,
                        h: 32,
                        values: Vec::from(sample.1)
                    }]);

                    network.learn(0.0001, &sample.0);

                    let prediction = most_predicted(&out);
                    if most_predicted(&sample.0) == prediction
                        && 1.0 - out[prediction] < 0.1 {
                        break;
                    }
                }

                if idx % 50 == 49 {
                    save_cnn_network(PATH, &network);
                    cnn_network_bmp("./networks/num-dataset/conv-ai-out", &network);

                    let ans = network.calculate(&[Matrix {
                        w: 32,
                        h: 32,
                        values: Vec::from(sample.1)
                    }]);

                    let sample_expect = most_predicted(&sample.0);
                    let prediction = most_predicted(&ans);

                    println!("Run: {idx} | {} {} {:?} {:?} \n", sample_expect, prediction, sample.0, ans);
                }
            }
        }

        let mut index = 0;
        let mut correct = 0;
        for sample in &samples {
            let ans = network.calculate(&[Matrix {
                w: 32,
                h: 32,
                values: Vec::from(sample.1)
            }]);

            let sample_expect = most_predicted(&sample.0);
            let prediction = most_predicted(&ans);

            println!("{index} {} {} {:?} {:?} \n", sample_expect, prediction, sample.0, ans);
            if sample_expect == prediction {
                correct += 1;
            }

            index += 1;
        }

        println!("{}", correct as f32 / samples.len() as f32);

        cnn_network_bmp("./networks/num-dataset/conv-ai-out", &network);

        save_cnn_network(PATH, &network);
    }
}
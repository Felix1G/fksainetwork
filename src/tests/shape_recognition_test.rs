#[cfg(test)]
mod shape_recognition_test {
    use std::cmp::{max, min};
    use std::fs;
    use crate::util::Matrix;
    use rand::distributions::Distribution;
    use bmp::{Image, Pixel};
    use rand::{random, Rng, thread_rng};
    use rand::distributions::Uniform;
    use rand::prelude::SliceRandom;
    use crate::activation::Activation;
    use crate::initialization::Initialization;
    use crate::initialization::Initialization::{BottomKernel, LeftKernel, RightKernel, TopKernel};
    use crate::loss::Loss;
    use crate::network::cnn::{cnn_network_bmp, ConvolutionalNetwork, load_cnn_network, save_cnn_network};
    use crate::pooling::Pooling;

    const PATH: &str = "./networks/shape_recognition.ai";

    /*
    0 -> Vertical
    1 -> Diagonal
    2 -> Horizontal
     */

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
                (2, &[Initialization::He, Initialization::He, Initialization::He, Initialization::He], Activation::Tanh, 2, Pooling::Max),
                //(3, &vec![Initialization::Xavier; 10], Activation::Tanh, 2, Pooling::Max),
            ],
            13, 13, 1, &[10, 10, 3],
            &[Initialization::He, Initialization::He, Initialization::He],
            &[Activation::Tanh, Activation::Tanh, Activation::Linear],
            Loss::BinaryCrossEntropy,
            true
        );*/
        load_cnn_network(PATH);

        let mut samples = Vec::<([f32; 3], Vec<f32>)>::new();
        let paths = fs::read_dir("./networks/shape-dataset").unwrap();
        for path in paths {
            let file = path.unwrap();
            if file.file_type().unwrap().is_dir() {
                continue;
            }

            let img = bmp::open(file.path()).unwrap();
            let num = file.file_name().as_encoded_bytes()[0] - 48;
            let mut pix_arr = Vec::<f32>::new();
            for y in 0u32..13u32 {
                for x in 0u32..13u32 {
                    pix_arr.push(img.get_pixel(x, y).r as f32 / 255f32);
                }
            }

            let mut ans_arr = [0f32; 3];
            ans_arr[num as usize] = 1.0;
            samples.push((ans_arr, pix_arr));
        }

        if !true {
            let mut rng = thread_rng();
            //samples.shuffle(&mut rng);
            let amount = 20;
            for run in 0..100000 {
                let mut error_total = 0.0f32;

                let mut idx_offset = rng.gen_range(0..40);

                for idx in 0..amount {
                    let sample = &samples[idx_offset + idx];

                    let out = network.calculate(&[Matrix {
                        w: 13,
                        h: 13,
                        values: sample.1.clone()
                    }]);

                    network.learn(0.001, &sample.0);

                    error_total += Loss::BinaryCrossEntropy.loss(&out, &sample.0);
                }

                if run % 3 == 1 {
                    let train_err = error_total / amount as f32;
                    save_cnn_network(PATH, &network);

                    if run % 100 == 99 {
                        cnn_network_bmp("./networks/shape-dataset/conv-ai-out", &network);
                        println!("GENERATED BITMAPS!");
                    }

                    println!("run: {run}, err: {train_err}");
                }
            }
        };

        let mut correct = 0;
        let mut index = 0;
        for sample in &samples {
            let ans = network.calculate(&[Matrix {
                w: 13,
                h: 13,
                values: sample.1.clone()
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

        cnn_network_bmp("./networks/shape-dataset/conv-ai-out", &network);
        save_cnn_network(PATH, &network);
    }
}
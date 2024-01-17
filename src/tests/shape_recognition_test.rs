#[cfg(test)]
mod shape_recognition_test {
    use std::cmp::{max, min};
    use std::fs;
    use std::process::exit;
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

    fn get_bmp(samples: &mut Vec<([f32; 3], Matrix)>, dir: &str) {
        let paths = fs::read_dir(dir).unwrap();
        for path in paths {
            let file = path.unwrap();
            if file.file_type().unwrap().is_dir() {
                continue;
            }

            let img = bmp::open(file.path()).unwrap();
            let num = file.file_name().as_encoded_bytes()[0] - 48;
            let mut arr = Vec::<f32>::new();
            for y in 0u32..13u32 {
                for x in 0u32..13u32 {
                    arr.push(img.get_pixel(x, y).r as f32 / 255f32);
                }
            }

            let mut ans_arr = [0f32; 3];
            ans_arr[num as usize] = 1.0;
            samples.push((ans_arr,
                          Matrix {
                              w: 13,
                              h: 13,
                              values: arr
                          }
            ));
        }
    }

    #[test]
    fn main() {
        let mut network = /*ConvolutionalNetwork::new(
            &[
                (2, &[Initialization::Xavier;20], Activation::ReLU, 2, Pooling::Max, false),
                (3, &[Initialization::Xavier;40], Activation::ReLU, 2, Pooling::Max, false)
            ],
            13, 13, 1,
            &[
                //(10, Initialization::Xavier, Activation::LeakyReLU, true),
                (3, Initialization::Xavier, Activation::LeakyReLU, false)],
            Loss::BinaryCrossEntropy, true
        );*/
        load_cnn_network(PATH);

        let mut samples = Vec::<([f32; 3], Matrix)>::new();
        let mut tests = Vec::<([f32; 3], Matrix)>::new();
        get_bmp(&mut samples, &"./networks/shape-dataset");
        get_bmp(&mut tests, &"./networks/shape-dataset/test");

        if !true {
            let mut rng = thread_rng();
            //samples.shuffle(&mut rng);
            let amount = 1;

            for run in 0..1000000 {
                samples.shuffle(&mut rng);
                let mut input_samples = Vec::<Vec<Vec<&Matrix>>>::new();
                let mut expecteds = Vec::<Vec<Vec<f32>>>::new();

                let mut idx = 0;
                while idx < samples.len() {
                    let mut vec = vec![];
                    let mut vec2 = vec![];

                    for i in idx..min(samples.len(), idx+amount) {
                        vec.push(vec![&samples[i].1]);
                        vec2.push(Vec::from(samples[i].0));
                    }

                    input_samples.push(vec);
                    expecteds.push(vec2);

                    idx += amount;
                }

                for idx in 0..input_samples.len() {
                    network.learn(0.04, &input_samples[idx], &expecteds[idx]);
                    //println!("{:?}, {:?}", sample.0, &out);
                }

                if (run % 5 < 4) {
                    continue;
                }

                save_cnn_network(PATH, &network);

                let mut train_err = 0f32;
                let mut out: Vec<f32> = vec![];

                for test in &tests {
                    out = network.calculate(&vec![&test.1]);
                    if out[0].is_nan() || out[1].is_nan() {
                        println!("ERROR: THE NEURAL NETWORK HAS ACHIEVED NAN STATE! EXPLODING GRADIENT?");
                        exit(0)
                    }
                    train_err += Loss::BinaryCrossEntropy.loss(&out, &test.0);
                }

                println!("\nEpoch: {run}, err: {}, latest out: {:?}", train_err / tests.len() as f32, out);
                cnn_network_bmp("./networks/shape-dataset/conv-ai-out", &network);
            }
        };

        let mut index = 0;
        let mut correct = 0;
        for sample in &tests {
            let ans = network.calculate(&[&sample.1]);

            let sample_expect = most_predicted(&sample.0);
            let prediction = most_predicted(&ans);

            println!("{index} {} {} {:?} {:?} \n", sample_expect, prediction, sample.0, ans);
            if sample_expect == prediction {
                correct += 1;
            }

            index += 1;
        }

        println!("{}", correct as f32 / tests.len() as f32);

        cnn_network_bmp("./networks/shape-dataset/conv-ai-out", &network);
        save_cnn_network(PATH, &network);
    }
}
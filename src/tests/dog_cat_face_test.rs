#[cfg(test)]
mod image_recognition_test {
    use std::cmp::min;
    use crate::util::{Matrix};
    use std::fs;
    use std::process::exit;
    use rand::{thread_rng};
    use rand::prelude::SliceRandom;
    use crate::activation::Activation;
    use crate::initialization::Initialization;
    use crate::loss::Loss;
    use crate::network::cnn::{cnn_network_bmp, ConvolutionalNetwork, load_cnn_network, save_cnn_network};
    use crate::pooling::Pooling;

    //0 -> dog
    //1 -> cat

    const PATH: &str = "./networks/dog-cat.ai";

    fn most_predicted(arr: &[f32]) -> usize {
        arr.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index)
            .unwrap()
    }

    fn get_bmp(samples: &mut Vec<([f32; 2], Matrix, Matrix, Matrix)>, dir: &str) {
        let paths = fs::read_dir(dir).unwrap();
        for path in paths {
            let file = path.unwrap();
            if file.file_type().unwrap().is_dir() {
                continue;
            }

            let img = bmp::open(file.path()).unwrap();
            let num = file.file_name().as_encoded_bytes()[0] - 48;
            let mut red = Vec::<f32>::new();
            let mut green = Vec::<f32>::new();
            let mut blue = Vec::<f32>::new();
            for y in 0u32..102u32 {
                for x in 0u32..102u32 {
                    red.push(img.get_pixel(x, y).r as f32 / 255f32);
                    green.push(img.get_pixel(x, y).g as f32 / 255f32);
                    blue.push(img.get_pixel(x, y).b as f32 / 255f32);
                }
            }

            let mut ans_arr = [0f32; 2];
            ans_arr[num as usize] = 1.0;
            samples.push((ans_arr,
                          Matrix {
                              w: 102,
                              h: 102,
                              values: red
                          },
                          Matrix {
                              w: 102,
                              h: 102,
                              values: green
                          },
                          Matrix {
                              w: 102,
                              h: 102,
                              values: blue
                          }
            ));
        }
    }

    #[test]
    fn main() {
        let mut network = ConvolutionalNetwork::new(
            &[
                (3, &[Initialization::He;10], Activation::ReLU, 2, Pooling::Max, false),
                (3, &[Initialization::He;20], Activation::ReLU, 2, Pooling::Max, true),
                (3, &[Initialization::He;40], Activation::ReLU, 2, Pooling::Max, true)
            ],
            102, 102, 3,
            &[
                (100, Initialization::Xavier, Activation::Tanh, false),
                (2, Initialization::Xavier, Activation::Linear, true)],
            Loss::BinaryCrossEntropy, true
        );
        //load_cnn_network(PATH);

        let mut samples = Vec::<([f32; 2], Matrix, Matrix, Matrix)>::new();
        let mut tests = Vec::<([f32; 2], Matrix, Matrix, Matrix)>::new();
        get_bmp(&mut samples, &"./networks/dog-cat-dataset");
        get_bmp(&mut tests, &"./networks/dog-cat-dataset/test");

        if true {
            let mut rng = thread_rng();

            let amount = 4;

            for run in 0..100000 {
                samples.shuffle(&mut rng);
                let mut input_samples = Vec::<Vec<Vec<&Matrix>>>::new();
                let mut expecteds = Vec::<Vec<Vec<f32>>>::new();

                let mut idx = 0;
                while idx < samples.len() {
                    let mut vec = vec![];
                    let mut vec2 = vec![];

                    for i in idx..min(samples.len(), idx+amount) {
                        vec.push(vec![&samples[i].1, &samples[i].2, &samples[i].3]);
                        vec2.push(Vec::from(samples[i].0));
                    }

                    input_samples.push(vec);
                    expecteds.push(vec2);

                    idx += amount;
                }

                for idx in 0..input_samples.len() {
                    network.learn(0.0001, &input_samples[idx], &expecteds[idx]);
                    //println!("{:?}, {:?}", sample.0, &out);
                }

                save_cnn_network(PATH, &network);

                let mut train_err = 0f32;
                let mut out: Vec<f32> = vec![];

                for test in &tests {
                    out = network.calculate(&vec![&test.1, &test.2, &test.3]);
                    if out[0].is_nan() || out[1].is_nan() {
                        println!("ERROR: THE NEURAL NETWORK HAS ACHIEVED NAN STATE! EXPLODING GRADIENT?");
                        exit(0)
                    }
                    train_err += Loss::BinaryCrossEntropy.loss(&out, &test.0);
                }

                println!("\nEpoch: {run}, err: {}, latest out: {:?}", train_err / tests.len() as f32, out);
                cnn_network_bmp("./networks/dog-cat-dataset/conv-ai-out", &network);
            }
        }

        let mut index = 0;
        let mut correct = 0;
        for sample in &samples {
            let ans = network.calculate(&[&sample.1, &sample.2, &sample.3]);

            let sample_expect = most_predicted(&sample.0);
            let prediction = most_predicted(&ans);

            println!("{index} {} {} {:?} {:?} \n", sample_expect, prediction, sample.0, ans);
            if sample_expect == prediction {
                correct += 1;
            }

            index += 1;
        }

        println!("{}", correct as f32 / samples.len() as f32);

        cnn_network_bmp("./networks/dog-cat-dataset/conv-ai-out", &network);

        save_cnn_network(PATH, &network);
    }
}
/*
Predicts whether x or y is higher.

2 3 3 2
inputs: x y
outputs:
x > y ? 1 : 0
x > y ? 0 : 1
*/
#[cfg(test)]
mod num_high_low_test {
    use rand::{Rng, thread_rng};
    use crate::activation::Activation;
    use crate::initialization::Initialization;
    use crate::loss::Loss;
    use crate::network::{load_network, Network, save_network};

    const PATH: &str = "C:/Users/ACER/OneDrive/Documents/GitHub/fksainetwork/num_high_low_test.ai";

    #[test]
    fn main() {
        //create or read the a neural network
        let mut network = Network::new(3, &[
            (10, Initialization::Xavier, Activation::LeakyReLU, true),
            (10, Initialization::Xavier, Activation::LeakyReLU, false),
            (3, Initialization::Xavier, Activation::LeakyReLU, true)
        ], Loss::BinaryCrossEntropy, true);
        //let mut network = load_network(PATH);

        let mut rng = thread_rng();

        //learning
        for i in 0..1000000 {
            let mut inputs = Vec::<Vec<f32>>::new();
            let mut expecteds = Vec::<Vec<f32>>::new();

            for _ in 0..3 {
                let mut input: [f32; 3] = [
                    rng.gen_range(-50..50) as f32,
                    rng.gen_range(-50..50) as f32,
                    rng.gen_range(-50..50) as f32
                ];

                let more0 = input[0] > input[1] && input[0] > input[2];
                let more1 = !more0 && input[1] > input[2];
                let more2 = !more0 && !more1;
                let expected: [f32; 3] = [if more0 { 1.0 } else { 0.0 },
                    if more1 { 1.0 } else { 0.0 }, if more2 { 1.0 } else { 0.0 }];

                inputs.push(Vec::from(input));
                expecteds.push(Vec::from(expected));
            }

            network.learn(0.01, &inputs, &expecteds, true);

            if i % 100000 == 0 {
                save_network(PATH, &network);
                let output = network.calculate(&inputs[0]);
                println!("Run {i} {:?} {}", output,
                         Loss::BinaryCrossEntropy.loss(&output, &expecteds[0]));
            }
        }

        //testing
        for _ in 0..1000 {
            let input: [f32; 3] = [
                rng.gen_range(-50..50) as f32,
                rng.gen_range(-50..50) as f32,
                rng.gen_range(-50..50) as f32
            ];
            let output = network.calculate(&input);
            println!("{:?} {:?}", input, output);
        }

        //print network
        println!("{network}");

        //save network
        save_network(PATH, &network);
    }
}
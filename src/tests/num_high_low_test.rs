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
    use crate::network::{load_network, Network, save_network};

    const PATH: &str = "C:/Users/ACER/OneDrive/Documents/GitHub/fksainetwork/num_high_low_test.ai";

    #[test]
    fn main() {
        //create or read the a neural network
        let layers = [2, 3, 3, 2];
        let activations: [usize; 4] = [0, 0, 1, 3];
        let mut network = Network::new(&layers, &activations);
        //let mut network = load_network(String::from(PATH));

        let mut rng = thread_rng();

        //learning
        for i in 0..10000 {
            let mut input: [f32; 2] = [
                rng.gen_range(-5..50) as f32, 0.0
            ];
            input[1] = input[1] + rng.gen_range(-3..3) as f32;

            let more = input[0] > input[1];
            let expected: [f32; 2] = [if more { 1.0 } else { -1.0 }, if more { -1.0 } else { 1.0 }];

            for _ in 0..100 {
                network.calculate(&input);
                network.learn_bpg_mse(0.01, &expected);
            }

            //println!("Run {i}");
        }

        //testing
        for _ in 0..1000 {
            let input = [rng.gen_range(-50..50) as f32, rng.gen_range(-50..50) as f32];
            let output = network.calculate(&input);
            println!("{} {} {} | {:?}", input[0], if output[0] > output[1] { ">" } else { "<" }, input[1], output);
        }

        //print network
        println!("{network}");

        //save network
        save_network(PATH, &network);
    }
}
/*
A display of 3x5 is as follows with the index shown:
012
345
678
9O1
234

The test passes if the neural network is able to guess the character
drawn on the 3x5 display. The display below should have the
3rd output neuron to be the highest value (which is character 2):
000
||0
000
0||
000

15 25 25 36
inputs: 3x5 pixels
outputs:
0 1 2 3 4 5 6 7 8 9
*/
#[cfg(test)]
mod alphanum_3x5_test {
    use rand::{Rng, thread_rng};
    use crate::network::{load_network, Network, save_network};
    use crate::neuron::Neuron;

    const PATH: &str = "C:/Users/ACER/RustroverProjects/fksainetwork/networks/alphanum_3x5_test.ai";

    const TEST_ARR: [(char, [f32; 15]); 25] = [
        ('0', [
            0.0, 1.0, 0.0,
            1.0, 0.0, 1.0,
            1.0, 0.0, 1.0,
            1.0, 0.0, 1.0,
            0.0, 1.0, 0.0
        ]),
        ('0', [
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            1.0, 0.0, 1.0,
            0.0, 1.0, 0.0
        ]),
        ('0', [
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            1.0, 0.0, 1.0,
            1.0, 0.0, 1.0,
            0.0, 1.0, 0.0
        ]),
        ('1', [
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0
        ]),
        ('1', [
            0.0, 0.0, 1.0,
            0.0, 1.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0
        ]),
        ('1', [
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0
        ]),
        ('1', [
            0.0, 1.0, 0.0,
            1.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0
        ]),
        ('1', [
            1.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            1.0, 0.0, 0.0
        ]),
        ('1', [
            0.0, 1.0, 0.0,
            1.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
            1.0, 1.0, 1.0
        ]),
        ('2', [
            1.0, 1.0, 1.0,
            0.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 0.0, 0.0,
            1.0, 1.0, 1.0
        ]),
        ('2', [
            1.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            0.0, 1.0, 1.0,
            1.0, 0.0, 0.0,
            1.0, 1.0, 1.0
        ]),
        ('3', [
            1.0, 1.0, 1.0,
            0.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
            0.0, 0.0, 1.0,
            1.0, 1.0, 1.0
        ]),
        ('3', [
            0.0, 1.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 1.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 1.0, 1.0
        ]),
        ('4', [
            1.0, 0.0, 1.0,
            1.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0
        ]),
        ('5', [
            1.0, 1.0, 1.0,
            1.0, 0.0, 0.0,
            1.0, 1.0, 1.0,
            0.0, 0.0, 1.0,
            1.0, 1.0, 1.0
        ]),
        ('5', [
            1.0, 1.0, 1.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            1.0, 1.0, 1.0
        ]),
        ('6', [
            1.0, 1.0, 1.0,
            1.0, 0.0, 0.0,
            1.0, 1.0, 1.0,
            1.0, 0.0, 1.0,
            1.0, 1.0, 1.0
        ]),
        ('6', [
            0.0, 1.0, 1.0,
            1.0, 0.0, 0.0,
            1.0, 1.0, 0.0,
            1.0, 0.0, 1.0,
            0.0, 1.0, 0.0
        ]),
        ('7', [
            1.0, 1.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0
        ]),
        ('7', [
            0.0, 1.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0
        ]),
        ('7', [
            1.0, 1.0, 1.0,
            1.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0
        ]),
        ('8', [
            1.0, 1.0, 1.0,
            1.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 0.0, 1.0,
            1.0, 1.0, 1.0
        ]),
        ('8', [
            0.0, 1.0, 0.0,
            1.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 0.0, 1.0,
            0.0, 1.0, 0.0
        ]),
        ('9', [
            1.0, 1.0, 1.0,
            1.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0
        ]),
        ('9', [
            0.0, 1.0, 1.0,
            1.0, 0.0, 1.0,
            0.0, 1.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0
        ])
    ];

    fn expected(c: char) -> Vec<f32> {
        let index = ((c as u8) - '0' as u8) as usize;
        let mut arr = vec![0.0f32; 10];
        arr[index] = 1.0;
        arr
    }

    fn from_calculation(arr: &Vec<f32>) -> char {
        let mut idx = 0;
        let mut value = 0.0;

        for index in 0..arr.len() {
            let val = arr[index];
            if val > value {
                value = val;
                idx = index;
            }
        }

        ('0' as u8 + idx as u8) as char
    }

    #[test]
    fn main() {
        //create or read the a neural network
        //let layers = [15, 20, 20, 10];
        //let activations: [usize; 4] = [0, 2, 2, 1];
        //let mut network = Network::new(&layers, &activations);
        let mut network = load_network(String::from(PATH));

        let mut rng = thread_rng();

        let len = TEST_ARR.len();
        //learning
        /*for run in 0..10000000 {
            let data = &TEST_ARR[rng.gen_range(0..len)];
            let array = &data.1;
            let expected = &expected(data.0);

            let output = network.calculate(array);
            network.learn_bpg_mse(0.001, &expected);

            //println!("Output for {}: {:?}", data.0, network.calculate(array));

            if run % 10000 == 0 {
                println!("Run: {run}");
                save_network(String::from(PATH), &network);
            }
        }*/


        //testing
        for data in TEST_ARR {
            let array = data.1;
            let output_arr = network.calculate(&array);
            println!("{} ? {:?}", data.0, from_calculation(&output_arr));
        }

        let output = &network.calculate(&[
            1.0, 1.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
            1.0, 0.0, 0.0
        ]);
        println!("{:?} {:?}", from_calculation(output), output);

        //save network
        save_network(String::from(PATH), &network);
    }
}
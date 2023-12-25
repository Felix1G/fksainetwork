#[cfg(test)]
mod shape_recognition_test {
    use std::cmp::{max, min};
    use std::fs;
    use crate::util::Matrix;
    use rand::distributions::Distribution;
    use bmp::{Image, Pixel};
    use rand::{random, Rng, thread_rng};
    use rand::distributions::Uniform;
    use crate::network::cnn::{cnn_network_bmp, ConvolutionalNetwork, load_cnn_network, save_cnn_network};

    const PATH: &str = "./networks/shape_recognition.ai";

    /*
    0 -> Square
    1 -> Triangle
    2 -> Elipse
    3 -> Line
     */

    fn most_predicted(arr: &[f32]) -> usize {
        arr.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index)
            .unwrap()
    }

    /*fn draw_line(img: &mut Image, x0: u32, y0: u32, x1: u32, y1: u32) {
        let mut dx = (x1 as i32 - x0 as i32);
        let mut dy = (y1 as i32 - y0 as i32);
        let mut px = x0 as i32;
        let mut py = y0 as i32;

        let mx = if dx >= 1 { 1 } else if dx <= -1 { -1i32 } else { 0 };
        let my = if dy >= 1 { 1 } else if dy <= -1 { -1i32 } else { 0 };

        while dx.abs() > 0 || dy.abs() > 0 {
            while (dy).abs() > (dx).abs() {
                if dy.abs() > 0 {
                    py += my;
                    dy -= my;
                }

                if px >= 0 && px < 12 && py >= 0 && py < 12 {
                    img.set_pixel(px as u32, py as u32, Pixel::new(255u8, 255u8, 255u8));
                }
            }

            if px >= 0 && px < 12 && py >= 0 && py < 12 {
                img.set_pixel(px as u32, py as u32, Pixel::new(255u8, 255u8, 255u8));
            }

            if dx.abs() > 0 {
                px += mx;
                dx -= mx;
            }
        }
    }

    fn gen_data_set() {
        let mut uniform_range = Uniform::new_inclusive(0u32, 11u32);
        let mut rng = thread_rng();

        for idx in 0..4000 {
            let mut img = Image::new(12, 12);
            let mut name: String;

            if idx < 1000 {
                let x0 = uniform_range.sample(&mut rng);
                let x1 = (uniform_range.sample(&mut rng));
                let y0 = uniform_range.sample(&mut rng);
                let y1 = (uniform_range.sample(&mut rng));
                for x in min(x0, x1)..(max(x0, x1) + 1) {
                    img.set_pixel(x, y0, Pixel::new(255u8, 255u8, 255u8));
                    img.set_pixel(x, y1, Pixel::new(255u8, 255u8, 255u8));
                }
                for y in min(y0, y1)..(max(y0, y1) + 1) {
                    img.set_pixel(x0, y, Pixel::new(255u8, 255u8, 255u8));
                    img.set_pixel(x1, y, Pixel::new(255u8, 255u8, 255u8));
                }

                name = format!("0{idx}");
            } else if idx < 2000 {
                name = format!("1{}", idx - 1000);

                let x = rng.gen_range(1..=10);
                let y = rng.gen_range(1..=10);
                let min_rad = min(12 - x, min(x,
                                              min(12 - y, y)));
                let rad = rng.gen_range(1..=min_rad);
                let rad2 = rad * rad;
                let from_rad2 = (rad - 1) * (rad - 1);

                //it is inefficient, but it gets the job done
                for x_loc in 0..12 {
                    for y_loc in 0..12 {
                        let x_diff = x as i32 - x_loc as i32;
                        let y_diff = y as i32 - y_loc as i32;
                        let diff = x_diff * x_diff + y_diff * y_diff;
                        if diff <= rad2 && diff > from_rad2 {
                            img.set_pixel(x_loc as u32, y_loc as u32, Pixel::new(255u8, 255u8, 255u8));
                        }
                    }
                }
            } else if idx < 3000 {
                name = format!("2{}", idx - 2000);

                let x0 = uniform_range.sample(&mut rng);
                let x1 = uniform_range.sample(&mut rng);
                let x2 = uniform_range.sample(&mut rng);
                let y0 = uniform_range.sample(&mut rng);
                let y1 = uniform_range.sample(&mut rng);
                let y2 = uniform_range.sample(&mut rng);

                draw_line(&mut img, x0, y0, x1, y1);
                draw_line(&mut img, x0, y0, x2, y2);
                draw_line(&mut img, x1, y1, x2, y2);
            } else {
                let x0 = uniform_range.sample(&mut rng);
                let x1 = uniform_range.sample(&mut rng);
                let y0 = uniform_range.sample(&mut rng);
                let y1 = uniform_range.sample(&mut rng);

                draw_line(&mut img, x0, y0, x1, y1);

                name = format!("3{}", idx - 3000);
            }

            img.save(format!("./networks/shape-dataset/{name}.bmp")).unwrap();
        }
    }*/

    #[test]
    fn main() {
        let mut network = /*ConvolutionalNetwork::new(
            &[
                (20, 3, 2, 2, 0),
                (50, 2, 1, 2, 0),
            ],
            12, 12, 1, &[4], &[1]
        );*/
        load_cnn_network(PATH);

        let mut samples = Vec::<([f32; 4], [f32; 144])>::new();
        let paths = fs::read_dir("./networks/shape-dataset").unwrap();
        for path in paths {
            let file = path.unwrap();
            if file.file_type().unwrap().is_dir() {
                continue;
            }

            let img = bmp::open(file.path()).unwrap();
            let num = file.file_name().as_encoded_bytes()[0] - 48;
            let mut pix_arr = [0f32; 144];
            for x in 0u32..12u32 {
                for y in 0u32..12u32 {
                    pix_arr[(y * 12 + x) as usize] = img.get_pixel(x, y).r as f32 / 255f32;
                }
            }

            let mut ans_arr = [0f32; 4];
            ans_arr[num as usize] = 1.0;
            samples.push((ans_arr, pix_arr));
        }

        if !true {
            for idx in 0..10000000 {
                let index = ((idx % 4) * 50 + ((idx / 4) % 50)) % 200;
                let sample = samples[index];

                for i in 0..20 {
                    let out = network.calculate(&[Matrix {
                        w: 12,
                        h: 12,
                        values: Vec::from(sample.1)
                    }]);

                    network.learn_bpg_mse(1.0, &sample.0);

                    let prediction = most_predicted(&out);
                    if i % 5 == 4 && most_predicted(&sample.0) == prediction &&
                        1.0 - out[prediction] < 0.1 {
                        break;
                    }
                }

                if index == 119 {
                    save_cnn_network(PATH, &network);
                    cnn_network_bmp("./networks/shape-dataset/conv-ai-out", &network);
                    println!("Run: {idx}, {:?}", network.calculate(&[Matrix {
                        w: 12,
                        h: 12,
                        values: Vec::from(sample.1)
                    }]));
                }
            }
        };

        let mut correct = 0;
        let mut index = 0;
        for sample in &samples {
            let ans = network.calculate(&[Matrix {
                w: 12,
                h: 12,
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

        cnn_network_bmp("./networks/shape-dataset/conv-ai-out", &network);
        save_cnn_network(PATH, &network);
    }
}
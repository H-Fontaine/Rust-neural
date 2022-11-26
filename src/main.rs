use matrix::Matrix;
use mnist;
use mnist::{labels_to_responses, load_images, load_labels};
use rand::thread_rng;
use rand_distr::StandardNormal;
use neural::Network;

fn main() {
    let images_train = load_images("dataset/train-images.idx3-ubyte");
    let images_test = load_images("dataset/t10k-images.idx3-ubyte");
    let label_train = load_labels("dataset/train-labels.idx1-ubyte");
    let label_test = load_labels("dataset/t10k-labels.idx1-ubyte");

    let mut rng = thread_rng();
    let network = Network::new(vec![784, 30, 10], 4.0f32, &mut rng, &StandardNormal);
    let responses = labels_to_responses(&label_train);

    println!("Efficiency of the network before training : {}%", network.test(images_test.clone(), label_test.clone()) * 100f32);
    network.training(images_train, responses, 700, 100);
    println!("Efficiency of the network after training : {}%", network.test(images_test, label_test) * 100f32);
}
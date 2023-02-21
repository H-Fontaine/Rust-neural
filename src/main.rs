use mnist::{labels_to_responses, load_images, load_labels};
use rand::thread_rng;
use rand_distr::StandardNormal;
use neural::threaded::ThreadedNetwork;

fn main() {
    let nb_of_batch = 1000;
    let nb_of_thread = 10;
    let batch_size = 5;
    let learning_rate = 3.5f64;


    let images_train = load_images("dataset/train-images.idx3-ubyte");
    let images_test = load_images("dataset/t10k-images.idx3-ubyte");
    let label_train = load_labels("dataset/train-labels.idx1-ubyte");
    let label_test = load_labels("dataset/t10k-labels.idx1-ubyte");
    let responses = labels_to_responses(&label_train);


    let mut rng = thread_rng();
    let network = ThreadedNetwork::new(vec![784, 50, 10], learning_rate, &mut rng, &StandardNormal, nb_of_thread);
    println!("Efficiency of the network before training : {}%", network.test(images_test.clone(), label_test.clone()) * 100f32);
    network.training(images_train, responses, nb_of_batch, batch_size);
    println!("Efficiency of the network after training : {}%", network.test(images_test, label_test) * 100f32);
}
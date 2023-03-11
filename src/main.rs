use mnist::{print, load_images, load_labels, select_specific_numbers, save_image_normalized, labels_to_responses};
use rand::thread_rng;
use rand_distr::StandardNormal;
use neural::adversarial::AdversarialNetworks;
use neural::threaded::ThreadedNetwork;

enum TestNeural {
    Network,
    NetworkThreaded,
    Adversarial,
}

fn main() {
    let networks_to_test = TestNeural::Adversarial;

    match networks_to_test {
        TestNeural::Network => {

        }




        TestNeural::NetworkThreaded => {
            let nb_of_batch = 1000;
            let nb_of_thread = 20;
            let batch_size = 3;
            let learning_rate = 3.5f64;

            let images_train = load_images("dataset/train-images.idx3-ubyte");
            let images_test = load_images("dataset/t10k-images.idx3-ubyte");
            let label_train = load_labels("dataset/train-labels.idx1-ubyte");
            let label_test = load_labels("dataset/t10k-labels.idx1-ubyte");
            let responses = labels_to_responses(&label_train);

            let mut rng = thread_rng();
            let network = ThreadedNetwork::new(vec![784, 50, 10], learning_rate, &mut rng, &StandardNormal, nb_of_thread);
            println!("Efficiency of the network before training : {}%", network.test(images_test.clone(), label_test.clone()) * 100f64);
            network.training(images_train, responses, nb_of_batch, batch_size);
            println!("Efficiency of the network after training : {}%", network.test(images_test, label_test) * 100f64);
        }




        TestNeural::Adversarial => {
            let nb_of_batch = 2000;
            let batch_size = 30;
            let learning_rate = 0.2f64;
            let number_to_generate = 10;

            let images_test = load_images("dataset/t10k-images.idx3-ubyte");
            let label_test = load_labels("dataset/t10k-labels.idx1-ubyte");

            let mut rng = thread_rng();
            let (images, _labels) = select_specific_numbers(&images_test, &label_test, vec![0]);
            let mut generative_network = AdversarialNetworks::new(vec![1, 20, 784], vec![784, 20, 2], learning_rate, &mut rng, &StandardNormal);
            generative_network.training(images, nb_of_batch, batch_size, &mut rng, &StandardNormal);
            let test = generative_network.generate(number_to_generate, &mut rng, &StandardNormal);

            let location = "C:/Users/Hugo/Downloads";
            print(&test, &vec![1], 0);
            for i in 0..number_to_generate {
                save_image_normalized(location, i.to_string(),&test, &vec![1], 0);
            }
        }
    }
}
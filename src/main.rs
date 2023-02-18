use std::sync::{Arc, RwLock};
use std::sync::mpsc::channel;
use mnist;
use mnist::{labels_to_responses, load_images, load_labels};
use rand::{Rng, thread_rng};
use rand_distr::StandardNormal;
use neural::Network;
use matrix::Matrix;
use rand::distributions::Uniform;
use thread_pool::ThreadPool;

fn main() {
    let nb_of_batch = 1000;
    let nb_of_thread = 20;
    let batch_size = 100;
    let learning_rate = 3.5f32;


    let images_train = load_images("dataset/train-images.idx3-ubyte");
    let images_test = load_images("dataset/t10k-images.idx3-ubyte");
    let label_train = load_labels("dataset/train-labels.idx1-ubyte");
    let label_test = load_labels("dataset/t10k-labels.idx1-ubyte");
    let responses = labels_to_responses(&label_train);


    let mut rng = thread_rng();
    let mut network = Arc::new(RwLock::new(Network::new(vec![784, 50, 10], learning_rate, &mut rng, &StandardNormal)));

    println!("Efficiency of the network before training : {}%", network.read().unwrap().test(images_test.clone(), label_test.clone()) * 100f32);

    let thread_pool = ThreadPool::new(nb_of_thread);
    let correction_coef =  learning_rate / ((batch_size * nb_of_thread) as f32);
    let range = Uniform::new(0, images_train.lines());

    for _ in 0..nb_of_batch {
        let (sender, receiver) = channel();

        for _ in 0..nb_of_thread {
            let choices : Vec<usize> = thread_rng().sample_iter(range).take(batch_size).collect();  //| Choosing randomly the images that will be use to train the Network for this batch
            let chosen_images = images_train.chose_lines(&choices);                           //|
            let chosen_expected_results = responses.chose_lines(&choices);       //|
            let cloned_sender = sender.clone();
            let cloned_arc_network = network.clone();
            let task = move || {cloned_arc_network.read().unwrap().gradient(correction_coef, chosen_expected_results, cloned_arc_network.read().unwrap().propagation(chosen_images))};
            thread_pool.add_task(Box::new(task), Some(cloned_sender));
        }

        drop(sender);
        let mut receiver_iter = receiver.into_iter();
        let (mut correction_weights_sum, mut correction_bias_sum) = receiver_iter.next().unwrap();
        for (correction_weights, correction_bias) in receiver_iter {
            for (((weights_sum, bias_sum), weights), bias) in (&mut correction_weights_sum).into_iter().zip(&mut correction_bias_sum).zip(correction_weights).zip(correction_bias) {
                *weights_sum += weights;
                *bias_sum += bias;
            }
        }
        network.write().unwrap().correction((correction_weights_sum, correction_bias_sum));
    }
    thread_pool.join();


    println!("Efficiency of the network after training : {}%", network.read().unwrap().test(images_test, label_test) * 100f32);
}
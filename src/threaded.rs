use std::ops::AddAssign;
use std::sync::{Arc, RwLock};
use std::sync::mpsc::channel;
use matrix::Matrix;
use num_traits::{Float, NumCast};
use rand::distributions::{Distribution, Uniform};
use rand::{Rng, thread_rng};
use thread_pool::ThreadPool;
use crate::Network;

pub struct ThreadedNetwork<T> where T : Float + Send + Sync + 'static {
    network : Arc<RwLock<Network<T>>>,
    nb_of_threads : usize,
}

//CONSTRUCTORS
impl<T : Float + Send + Sync + 'static> ThreadedNetwork<T> {
    pub fn new<R: ?Sized + Rng, D: Distribution<T>> (shape : Vec<usize>, learning_rate : T, rng: &mut R, distribution: &D, number_of_threads : usize) -> ThreadedNetwork<T> {
        ThreadedNetwork {
            network : Arc::new(RwLock::new(Network::new(shape, learning_rate, rng, distribution))),
            nb_of_threads : number_of_threads,
        }
    }
}

impl<T : Float + Send + Sync + 'static> ThreadedNetwork<T> where T : AddAssign<T> {
    pub fn training(&self, images : Matrix<T>, expected_results : Matrix<T>, nb_of_batch : usize, batch_size : usize) {
        let thread_pool = ThreadPool::new(self.nb_of_threads);
        let correction_coef =  self.network.read().unwrap().learning_rate / NumCast::from(batch_size * self.nb_of_threads).unwrap();
        let range = Uniform::new(0, images.lines());

        for _ in 0..nb_of_batch {
            let (sender, receiver) = channel();

            for _ in 0..self.nb_of_threads {
                let choices : Vec<usize> = thread_rng().sample_iter(range).take(batch_size).collect();  //| Choosing randomly the images that will be use to train the Network for this batch
                let chosen_images = images.chose_lines(&choices);                           //|
                let chosen_expected_results = expected_results.chose_lines(&choices);       //|
                let cloned_sender = sender.clone();
                let cloned_arc_network = self.network.clone();

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
            self.network.write().unwrap().correction((correction_weights_sum, correction_bias_sum));
        }
        thread_pool.join();
    }

    pub fn test(&self, images : Matrix<T>, labels : Vec<usize>) -> f32 {
        let mut res = 0f32;
        let number_of_test = labels.len() as f32;
        let thread_pool = ThreadPool::new(self.nb_of_threads);
        let split_images = images.split_lines(self.nb_of_threads);
        let mut labels_iter = labels.into_iter();
        let (sender, receiver) = channel();

        for images_batch in split_images {
            let labels_batch = labels_iter.by_ref().take(images_batch.lines()).collect();
            let sender_cloned = sender.clone();
            let network_ref = self.network.clone();
            let task = move || {(images_batch.lines() as f32) * network_ref.read().unwrap().test(images_batch, labels_batch)};
            thread_pool.add_task(Box::new(task), Some(sender_cloned));
        }
        drop(sender);
        for result in receiver {
            res += result;
        }
        thread_pool.join();
        res / number_of_test
    }
}
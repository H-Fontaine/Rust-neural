use std::ops::{Add, AddAssign, Mul};
use std::sync::{Arc, Mutex};
use std::sync::mpsc::channel;
use std::thread::available_parallelism;
use matrix::{Concatenate, Matrix};
use num_traits::{cast, Float, NumCast};
use rand::{distributions::Distribution, Rng, thread_rng};
use rand::distributions::Uniform;
use thread_pool::ThreadPool;
use utils::math::sigmoid;

pub struct Network<T> {
    output_size : usize,
    nb_layers: usize,
    learning_rate : T,
    weights : Vec<Matrix<T>>,
    bias : Vec<Matrix<T>>,
}

//CONSTRUCTORS
impl<T : Float + Send> Network<T> {
    pub fn new<R: ?Sized + Rng, D: Distribution<T>> (shape : Vec<usize>, learning_rate : T, rng: &mut R, distribution: &D) -> Network<T> {
        let nb_layers = shape.len() - 1;
        let mut weights = Vec::<Matrix<T>>::with_capacity(nb_layers);
        let mut bias = Vec::<Matrix<T>>::with_capacity(nb_layers);

        for i in 0..nb_layers {
            weights.push(Matrix::new_rand(shape[i], shape[i + 1], rng, distribution) * num::cast(1f32 / (shape[i + 1] as f32).sqrt()).unwrap());
            bias.push(Matrix::new_rand(1, shape[i + 1], rng, distribution));
        }

        Network {
            output_size : *shape.last().unwrap(),
            nb_layers,
            learning_rate,
            weights,
            bias,
        }
    }
}

//USAGE
impl<T : Float> Network<T> {
    /*
    Propagate the input images through the network
     - images : Matrix<T>           The images that will go through the network
    */
    pub fn down(&self, images : Matrix<T>) -> Matrix<T> where T :  Mul<Output = T> + AddAssign + Copy {
        let mut weights = (&self.weights).into_iter();
        let mut bias = (&self.bias).into_iter();
        let mut res = (images * weights.next().unwrap() + bias.next().unwrap()).map(|a| sigmoid(a));
        for _ in 1..self.nb_layers {
            res = (res * weights.next().unwrap() + bias.next().unwrap()).map(|a| sigmoid(a));
        }
        res
    }
}

//TRAINING OF THE NETWORK
impl<T : Float> Network<T> where T : AddAssign<T> {
    /*
    Realise the training of the Network with the given images
     - images : Matrix<T>                   Images on which the network is trained
     - expected_results : Matrix<T>         The results expected for every images to calculate the cost function
     - nb_of_batch : usize                  The number of batch that will be processed
     - batch_size : usize                   The number of image that will be use per batch
    */
    pub fn training(&mut self, images : Matrix<T>, expected_results : Matrix<T>, nb_of_batch : usize, batch_size : usize) {
        let range = Uniform::new(0, images.lines());
        for _ in 0..nb_of_batch {
            let choices : Vec<usize> = thread_rng().sample_iter(range).take(batch_size).collect();  //| Choosing randomly the images that will be use to train the Network for this batch
            let chosen_images = images.chose_lines(&choices);                           //|
            let chosen_expected_results = expected_results.chose_lines(&choices);       //|

            //Realising the backpropagation with the result of the parallelization of propagation
            self.backpropagation(batch_size, chosen_expected_results, self.propagation(chosen_images));
        }
    }

    /*
    Realise the propagation of the given images through the network and returning all the necessary data to realise the backpropagation
    - images : Matrix<T>                            The images on which the propagation is made
    */
    fn propagation(&self, images : Matrix<T>) -> (Vec<Matrix<T>>, Vec<Matrix<T>>) {
        let mut weights_iter = (&self.weights).into_iter();
        let mut bias_iter = (&self.bias).into_iter();
        let mut lasts_res = Vec::<Matrix<T>>::with_capacity(self.nb_layers + 1);
        let mut lasts_cl = Vec::<Matrix<T>>::with_capacity(self.nb_layers);
        lasts_res.push(images);

        for _ in 0..self.nb_layers {
            lasts_cl.push(
                match lasts_res.last() {
                    Some(matrix) => matrix * weights_iter.next().unwrap() + bias_iter.next().unwrap(),
                    None => panic!("Error during propagation"),
                }
            );

            lasts_res.push(
                match lasts_cl.last() {
                    Some(matrix) => matrix.clone().map(|a| sigmoid(a)),
                    None => panic!("Error during propagation"),
                }
            );
        }
        (lasts_res, lasts_cl)
    }


    /*
    Realise the gradient descent according to the tuple of (lasts_res, lasts_cl) coming from propagation
     - batch_size : usize                                                   The size of the batch, it as to be known to correct the learning rate depending on the number image used to propagate
     - responses : Matrix<T>                                                The correction needed is based on the distance between responses and the actual result from the network coming in the last case of lasts_res
     - (mut lasts_res, mut lasts_cl) : (Vec<Matrix<T>>, Vec<Matrix<T>>)     Those are the computed results after each layer during the propagation of images and are needed to calculate how to correct the weights and bias of the network
    */
    fn backpropagation(&mut self, batch_size : usize, responses : Matrix<T>, (mut lasts_res, mut lasts_cl) : (Vec<Matrix<T>>, Vec<Matrix<T>>)) {
        let correction_coef : T = self.learning_rate / NumCast::from(batch_size).unwrap();

        let cost = responses + -lasts_res.pop().expect("Error during backpropagation");

        let mut layer_error = cost & lasts_cl.pop().expect("Error during backpropagation").map(|a| sigmoid(a) * (-sigmoid(a) + NumCast::from(1).unwrap()));

        self.weights[self.nb_layers - 1] += (lasts_res.pop().expect("Error during backpropagation").t() * layer_error.clone()) * correction_coef;
        self.bias[self.nb_layers - 1] += layer_error.clone().sum_line() * correction_coef;

        for i in 1..self.nb_layers {
            layer_error = (layer_error * self.weights[self.nb_layers - i].t()) & (lasts_cl.pop().expect("Error during backpropagation").map(|a| sigmoid(a) * (-sigmoid(a) + NumCast::from(1).unwrap())));
            self.weights[self.nb_layers - (i + 1)] += (lasts_res.pop().expect("Error during backpropagation").t() * layer_error.clone()) * correction_coef;
            self.bias[self.nb_layers - (i + 1)] += layer_error.clone().sum_line() * correction_coef;
        }
    }
}

impl<T : Float> Network<T> where T : Clone + Copy + Add<T, Output = T> + Mul<T, Output = T> + AddAssign {
    pub fn test(&self, images : Matrix<T>, labels : Vec<usize>) -> f32 {
        let nb_images = images.lines();
        let results = self.down(images);
        let mut res = 0f32;
        for i in 0..nb_images {
            let mut index = 0;
            let mut max = results[i][0];
            for j in 1..self.output_size {
                if results[i][j] >= max {
                    index = j;
                    max = results[i][j];
                }
            }
            if labels[i] == index {
                res += 1f32;
            }
        }
        res / (nb_images as f32)
    }
}













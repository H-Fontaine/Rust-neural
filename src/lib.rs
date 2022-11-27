use std::ops::{Add, AddAssign, Mul};
use std::sync::{Arc, Mutex};
use std::sync::mpsc::channel;
use std::thread::available_parallelism;
use matrix::{Concatenate, Matrix};
use num_traits::{Float, NumCast};
use rand::{distributions::Distribution, Rng, thread_rng};
use rand::distributions::Uniform;
use thread_pool::ThreadPool;
use utils::math::sigmoid;

pub struct Network<T> {
    output_size : usize,
    nb_layers: usize,
    learning_rate : T,
    weights : Arc<Mutex<Vec<Matrix<T>>>>,
    bias : Arc<Mutex<Vec<Matrix<T>>>>,
}

//CONSTRUCTORS
impl<T : Float + Send> Network<T> {
    pub fn new<R: ?Sized + Rng, D: Distribution<T>>(shape : Vec<usize>, learning_rate : T, rng: &mut R, distribution: &D) -> Network<T> {
        let nb_layers = shape.len() - 1;
        let mut weights = Vec::<Matrix<T>>::with_capacity(nb_layers);
        let mut bias = Vec::<Matrix<T>>::with_capacity(nb_layers);

        for i in 0..nb_layers {
            weights.push(Matrix::new_rand(shape[i], shape[i + 1], rng, distribution));
            bias.push(Matrix::new_rand(1, shape[i + 1], rng, distribution));
        }

        Network {
            output_size : *shape.last().unwrap(),
            nb_layers,
            learning_rate,
            weights : Arc::new(Mutex::new(weights)),
            bias : Arc::new(Mutex::new(bias)),
        }
    }
}

//USAGE
impl<T : Float + Send> Network<T> {
    /*
    Propagate the input images through the network
     - images : Matrix<T>           The images that will go through the network
    */
    pub fn down(&self, images : Matrix<T>) -> Matrix<T> where T : Clone + Copy + Add<T, Output = T> + Mul<T, Output = T> + AddAssign {
        let weights;
        let bias;
        {
            weights = self.weights.lock().unwrap().clone(); // Getting the lock and cloning data
        }
        {
            bias = self.bias.lock().unwrap().clone(); // Getting the lock and cloning data
        }
        let mut weights = weights.into_iter();  //| Turning the data into iterators
        let mut bias = bias.into_iter();                        //|
        let mut res = (images * weights.next().unwrap() + bias.next().unwrap()).map(|a| sigmoid(a));
        for _ in 1..self.nb_layers {
            res = (res * weights.next().unwrap() + bias.next().unwrap()).map(|a| sigmoid(a));
        }
        res
    }
}

//TRAINING OF THE NETWORK
impl<T : Float + Send + 'static> Network<T> where T : AddAssign<T> {
    /*
    Realise the training of the Network with the given images
     - images : Matrix<T>                   Images on which the network is trained
     - expected_results : Matrix<T>         The results expected for every images to calculate the cost function
     - nb_of_batch : usize                  The number of batch that will be processed
     - batch_size : usize                   The number of image that will be use per batch
    */
    pub fn training(&self, images : Matrix<T>, expected_results : Matrix<T>, nb_of_batch : usize, batch_size : usize) {
        let range = Uniform::new(0, images.lines());

        let number_of_threads : usize = From::from(available_parallelism().unwrap());
        let thread_pool= ThreadPool::new(number_of_threads);

        let learning_rate = self.learning_rate;
        let nb_layers = self.nb_layers;

        for _ in 0..nb_of_batch {
            let choices : Vec<usize> = thread_rng().sample_iter(range).take(batch_size).collect();  //| Choosing randomly the images that will be use to train the Network for this batch
            let chosen_images = images.chose_lines(&choices);                           //|
            let chosen_expected_results = expected_results.chose_lines(&choices);       //|

            let split_chosen_images = chosen_images.split_lines(number_of_threads); //Splitting the chosen images to parallelize the propagation on multiple thread using the thread pool


            let (sender, receiver) = channel();
            let mut id = 0;
            for images in split_chosen_images {                                                      //| Sending all the images to the threads to realise the propagation
                let weights_ref = self.weights.clone();                                     //|
                let bias_ref = self.bias.clone();                                           //|
                let sender_clone = sender.clone();                                                   //|
                let runnable = move || {                                                              //|
                    (Network::propagation(nb_layers, images, weights_ref, bias_ref), id)           //| Sending an id to know witch propagation is associated to with expected_results
                };                                                                                             //|
                thread_pool.add_task(Box::new(runnable), Some(sender_clone));         //|
                id +=1;                                                                                        //|
            }
            drop(sender);


            //| Sorting things back together according to the ids
            let mut lasts_res : Vec<Vec<Matrix<T>>> = vec![vec![Matrix::new(); number_of_threads]; nb_layers + 1];
            let mut lasts_cl: Vec<Vec<Matrix<T>>> = vec![vec![Matrix::new(); number_of_threads]; nb_layers];
            for ((lasts_res_to_concatenate, lasts_cl_to_concatenate), id) in receiver {
                let mut lasts_res_iter = lasts_res_to_concatenate.into_iter();
                let mut lasts_cl_iter = lasts_cl_to_concatenate.into_iter();
                for i in 0..(nb_layers + 1) {
                    lasts_res[i][id] = lasts_res_iter.next().unwrap();
                }
                for i in 0..nb_layers {
                    lasts_cl[i][id] = lasts_cl_iter.next().unwrap();
                }
            }

            //Concatenating things back together
            let lasts_res = {
                let mut res = Vec::with_capacity(nb_layers + 1);
                for matrixs in lasts_res {
                    res.push(matrixs.concatenate_lines());
                }
                res
            };
            let lasts_cl = {
                let mut res = Vec::with_capacity(nb_layers);
                for matrixs in lasts_cl {
                    res.push(matrixs.concatenate_lines());
                }
                res
            };

            //Realising the backpropagation with the result of the parallelization of propagation
            Network::backpropagation(batch_size, nb_layers, learning_rate, chosen_expected_results, self.weights.clone(), self.bias.clone(), (lasts_res, lasts_cl));

        }
        thread_pool.join();
    }

    /*
    Realise the propagation of the given images through the network and returning all the necessary data to realise the backpropagation
    - nb_layers : usize                             The number of layer in the Network
    - images : Matrix<T>                            The images on which the propagation is made
    - weights : Arc<Mutex<Vec<Matrix<T>>>>          The reference to the weights that must be cloned to realise the propagation
    - bias : Arc<Mutex<Vec<Matrix<T>>>>             The reference to the bias that must be cloned to realise the propagation
    */
    fn propagation(nb_layers : usize, images : Matrix<T>, weights : Arc<Mutex<Vec<Matrix<T>>>>, bias : Arc<Mutex<Vec<Matrix<T>>>>) -> (Vec<Matrix<T>>, Vec<Matrix<T>>) {
        let mut weights_iter;
        let mut bias_iter;
        {
            weights_iter = weights.lock().unwrap().clone().into_iter();
            bias_iter = bias.lock().unwrap().clone().into_iter();
        }
        let mut lasts_res = Vec::<Matrix<T>>::with_capacity(nb_layers + 1);
        let mut lasts_cl = Vec::<Matrix<T>>::with_capacity(nb_layers);
        lasts_res.push(images);

        for _ in 0..nb_layers {
            lasts_cl.push(
                match lasts_res.last() {
                    Some(matrix) => matrix.clone() * weights_iter.next().unwrap() + bias_iter.next().unwrap(),
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
     - nb_layers : usize                                                    The number of layers in the Network
     - learning_rate : T                                                    The learning rate is a coefficient that define the strength of the correction applied to the weights and bias
     - responses : Matrix<T>                                                The correction needed is based on the distance between responses and the actual result from the network coming in the last case of lasts_res
     - weights : Arc<Mutex<Vec<Matrix<T>>>>                                 Reference to the weights that are going to be corrected according to the data sent by the propagation
     - bias : Arc<Mutex<Vec<Matrix<T>>>>                                    Reference to the bias that are going to be corrected according to the data sent by the propagation
     - (mut lasts_res, mut lasts_cl) : (Vec<Matrix<T>>, Vec<Matrix<T>>)     Those are the computed results after each layer during the propagation of images and are needed to calculate how to correct the weights and bias of the network
    */
    fn backpropagation(batch_size : usize, nb_layers : usize, learning_rate : T, responses : Matrix<T>, weights : Arc<Mutex<Vec<Matrix<T>>>>, bias : Arc<Mutex<Vec<Matrix<T>>>>, (mut lasts_res, mut lasts_cl) : (Vec<Matrix<T>>, Vec<Matrix<T>>)) {
        let mut weights = weights.lock().unwrap();
        let mut bias = bias.lock().unwrap();

        let correction_coef : T = learning_rate / NumCast::from(batch_size).unwrap();

        let cost = responses + -lasts_res.pop().expect("Error during backpropagation");

        let mut layer_error = cost & lasts_cl.pop().expect("Error during backpropagation").map(|a| sigmoid(a) * (-sigmoid(a) + NumCast::from(1).unwrap()));

        weights[nb_layers - 1] += (lasts_res.pop().expect("Error during backpropagation").t() * layer_error.clone()) * correction_coef;
        bias[nb_layers - 1] += layer_error.clone().sum_line() * correction_coef;

        for i in 1..nb_layers {
            layer_error = (layer_error * weights[nb_layers - i].t()) & (lasts_cl.pop().expect("Error during backpropagation").map(|a| sigmoid(a) * (-sigmoid(a) + NumCast::from(1).unwrap())));
            weights[nb_layers - (i + 1)] += (lasts_res.pop().expect("Error during backpropagation").t() * layer_error.clone()) * correction_coef;
            bias[nb_layers - (i + 1)] += layer_error.clone().sum_line() * correction_coef;
        }
    }
}

impl<T : Float + Send> Network<T> where T : Clone + Copy + Add<T, Output = T> + Mul<T, Output = T> + AddAssign {
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













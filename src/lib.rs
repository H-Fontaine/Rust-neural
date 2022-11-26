use std::ops::{Add, AddAssign, Mul};
use std::sync::{Arc, Mutex};
use std::thread::available_parallelism;
use matrix::Matrix;
use num_traits::{Float, NumCast};
use rand::{distributions::Distribution, Rng, thread_rng};
use rand::distributions::Uniform;
use thread_pool::ThreadPool;
use utils::math::sigmoid;

pub struct Network<T> {
    entry_size : usize,
    output_size : usize,
    nb_layers: usize,
    learning_rate : T,
    shape : Vec<usize>,
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
            entry_size : *shape.first().unwrap(),
            output_size : *shape.last().unwrap(),
            nb_layers,
            learning_rate,
            shape,
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
        let number_of_threads : usize = From::from(available_parallelism().unwrap());
        let thread_pool: ThreadPool<()> = ThreadPool::new(1);

        let range = Uniform::new(0, images.lines());
        for _ in 0..nb_of_batch {
            let choices : Vec<usize> = thread_rng().sample_iter(range).take(batch_size).collect();
            let chosen_images = images.chose_lines(&choices);
            let chosen_expected_results = expected_results.chose_lines(&choices);
            let weights = self.weights.clone();
            let bias = self.bias.clone();
            let nb_layers = self.nb_layers.clone();
            let learning_rate = self.learning_rate.clone();
            let runnable = move || {
                let weights_cloned;
                let bias_cloned;
                {
                    weights_cloned = weights.lock().unwrap().clone();
                    bias_cloned = bias.lock().unwrap().clone();
                }
                let mut lasts_res = Vec::<Matrix<T>>::with_capacity(nb_layers);
                let mut lasts_cl = Vec::<Matrix<T>>::with_capacity(nb_layers);
                lasts_res.push(chosen_images);

                for i in 0..nb_layers {
                    lasts_cl.push(
                        match lasts_res.last() {
                            Some(matrix) => matrix.clone() * weights_cloned[i].clone() + bias_cloned[i].clone(),
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

                let mut weights = weights.lock().unwrap();
                let mut bias = bias.lock().unwrap();

                let correction_coef : T = learning_rate / NumCast::from(batch_size).unwrap();

                let cost = chosen_expected_results + -lasts_res.pop().expect("Error during backpropagation");

                let mut layer_error = cost & lasts_cl.pop().expect("Error during backpropagation").map(|a| sigmoid(a) * (-sigmoid(a) + NumCast::from(1).unwrap()));

                weights[nb_layers - 1] += (lasts_res.pop().expect("Error during backpropagation").t() * layer_error.clone()) * correction_coef;
                bias[nb_layers - 1] += layer_error.clone().sum_line() * correction_coef;

                for i in 1..nb_layers {
                    layer_error = (layer_error * weights[nb_layers - i].t()) & (lasts_cl.pop().expect("Error during backpropagation").map(|a| sigmoid(a) * (-sigmoid(a) + NumCast::from(1).unwrap())));
                    weights[nb_layers - (i + 1)] += (lasts_res.pop().expect("Error during backpropagation").t() * layer_error.clone()) * correction_coef;
                    bias[nb_layers - (i + 1)] += layer_error.clone().sum_line() * correction_coef;
                }
            };

            thread_pool.add_task(Box::new(runnable), None);
        }
        thread_pool.join();
    }


    /*
    Realise the propagation of the given images through the network and returning all the necessary data to realise the backpropagation
    - images : Matrix<T>            The images on which the propagation is made
    */
    fn propagation(nb_layers : usize, images : Matrix<T>, weights : Arc<Mutex<Vec<Matrix<T>>>>, bias : Arc<Mutex<Vec<Matrix<T>>>>) -> (Vec<Matrix<T>>, Vec<Matrix<T>>) {
        let weights_cloned;
        let bias_cloned;
        {
            weights_cloned = weights.lock().unwrap().clone();
            bias_cloned = bias.lock().unwrap().clone();
        }
        let mut lasts_res = Vec::<Matrix<T>>::with_capacity(nb_layers);
        let mut lasts_cl = Vec::<Matrix<T>>::with_capacity(nb_layers);
        lasts_res.push(images);

        for i in 0..nb_layers {
            lasts_cl.push(
                match lasts_res.last() {
                    Some(matrix) => matrix.clone() * weights_cloned[i].clone() + bias_cloned[i].clone(),
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
     - images : Matrix<T>                                                   The images from which the correction is made
     - responses : Matrix<T>                                                The correction needed is based on the distance between responses and the actual result from the network coming in the last case of lasts_res
     - (mut lasts_res, mut lasts_cl) : (Vec<Matrix<T>>, Vec<Matrix<T>>)     Those are the computed results after each layer during the propagation of images and are needed to calculate how to correct the weights and bias of the network
    */
    fn backpropagation(batch_size : usize, nb_layers : usize, learning_rate : T, responses : Matrix<T>, weights : Arc<Mutex<Vec<Matrix<T>>>>, bias : Arc<Mutex<Vec<Matrix<T>>>>, (mut lasts_res, mut lasts_cl) : (Vec<Matrix<T>>, Vec<Matrix<T>>)) {
        let mut weights = weights.lock().unwrap();
        let mut bias = bias.lock().unwrap();

        let correction_coef : T = learning_rate / NumCast::from(batch_size).unwrap();

        let cout = responses + -lasts_res.pop().expect("Error during backpropagation");

        let mut layer_error = cout & lasts_cl.pop().expect("Error during backpropagation").map(|a| sigmoid(a) * (-sigmoid(a) + NumCast::from(1).unwrap()));

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













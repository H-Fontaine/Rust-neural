use std::ffi::c_void;
use std::ops::{Add, AddAssign, Div, Mul, Neg};
use std::process::Output;
use std::sync::Mutex;
use std::thread::available_parallelism;
use matrix::Matrix;
use num_traits::{NumCast, Zero};
use rand::{distributions::Distribution, Rng, thread_rng};
use rand::distributions::Uniform;
use thread_pool::ThreadPool;

pub struct Network<T, F> where F : Fn(T) -> T {
    entry_size : usize,
    output_size : usize,
    nb_layers: usize,
    learning_rate : T,
    shape : Vec<usize>,
    bias : Mutex<Vec<Matrix<T>>>,
    weights : Mutex<Vec<Matrix<T>>>,
    activation_fn : F,
    derivative_fn : F,
}

impl<T, F : Fn(T) -> T> Network<T, F> where T : Copy {
    pub fn new<R: ?Sized + Rng, D: Distribution<T>>(shape : Vec<usize>, learning_rate : T, activation_fn : F, derivative_fn : F, rng: &mut R, distribution: &D) -> Network<T, F> {
        let nb_layers = shape.len() - 1;
        let mut weights = Vec::<Matrix<T>>::with_capacity(nb_layers);
        let mut bias = Vec::<Matrix<T>>::with_capacity(nb_layers);

        for i in 0..nb_layers {
            weights.push(Matrix::new_rand(shape[i], shape[i + 1], rng, distribution));
            bias.push(Matrix::new_rand(1, shape[i + 1], rng, distribution));
        }

        Network {
            entry_size : *shape.first().unwrap(),
            output_size : *shape.first().unwrap(),
            nb_layers,
            learning_rate,
            shape,
            bias : Mutex::new(bias),
            weights : Mutex::new(weights),
            activation_fn,
            derivative_fn,
        }
    }
}

//CONSTRUCTORS
impl<T, F : Fn(T) -> T> Network<T, F> {
    /*
    Propagate the input images through the network
     - images : Matrix<T>           The images that will go through the network
    */
    pub fn down(&self, images : Matrix<T>) -> Matrix<T> where T : Clone + Copy + Add<T, Output = T> + Mul<T, Output = T> + AddAssign {
        let mut weights = self.weights.lock().unwrap(); //| Getting the lock
        let mut bias = self.bias.lock().unwrap();       //|
        let mut res = (images * weights[0].clone() + bias[0].clone()).map(|a| (self.activation_fn)(a));
        for i in 1..self.nb_layers {
            res = (res * weights[i].clone() + bias[i].clone()).map(|a| (self.activation_fn)(a));
        }
        res
    }
}

//TRAINING OF THE NETWORK
impl<T, F : Fn(T) -> T + Send + Sync> Network<T, F> where T : Mul<T, Output = T> + Div<T, Output = T> + Add<T, Output = T> + AddAssign<T> + Neg<Output = T> + NumCast + Copy + Zero + Clone + Send + Sync {
    /*
    Realise the training of the Network with the given images
     - images : Matrix<T>                   Images on which the network is trained
     - expected_results : Matrix<T>         The results expected for every images to calculate the cost function
     - nb_of_batch : usize                  The number of batch that will be processed
     - batch_size : usize                   The number of image that will be use per batch
    */
    pub fn training(&self, images : &Matrix<T>, expected_results : &Matrix<T>, nb_of_batch : usize, batch_size : usize) {
        let number_of_threads = From::from(available_parallelism().unwrap());
        let mut thread_pool: ThreadPool<()> = ThreadPool::new(number_of_threads);

        let range = Uniform::new(0, images.lines());
        for _i in 0..nb_of_batch {
            let choices : Vec<usize> = thread_rng().sample_iter(range).take(batch_size).collect();
            let chosen_images = images.chose_lines(&choices);
            let chosen_expected_results = expected_results.chose_lines(&choices);
            thread_pool.add_task(Box::new(move || {
                    let cloned_chosen_images = chosen_images.clone();
                    self.backpropagation(chosen_images.clone(), chosen_expected_results.clone(), self.propagation(cloned_chosen_images));
                }
            ), None);
        }
    }


    /*
    Realise the propagation of the given images through the network and returning all the necessary data to realise the backpropagation
    - images : Matrix<T>            The images on which the propagation is made
    */
    fn propagation(&self, images : Matrix<T>) -> (Vec<Matrix<T>>, Vec<Matrix<T>>) {
        let weights;
        let bias;
        {
            weights = self.weights.lock().unwrap().clone(); // Getting the lock and cloning data
        }
        {
            bias = self.bias.lock().unwrap().clone(); // Getting the lock and cloning data
        }
        let mut weights = weights.into_iter();  //| Turning the data into iterators
        let mut bias = bias.into_iter();        //|

        let mut lasts_res = Vec::<Matrix<T>>::with_capacity(self.nb_layers);
        let mut lasts_cl = Vec::<Matrix<T>>::with_capacity(self.nb_layers);
        lasts_res.push(images);

        for _ in 0..self.nb_layers {
            lasts_cl.push(
                match lasts_res.last() {
                    Some(matrix) => matrix.clone() * weights.next().unwrap() + bias.next().unwrap(), //Realising the linear product of the layer
                    None => panic!("Error during propagation"),
                }
            );
            lasts_res.push(
                match lasts_cl.last() {
                    Some(matrix) => matrix.clone().map(|a| (self.activation_fn)(a)), //Applying the activation function to the linear product of the layer
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
    fn backpropagation(&self, images : Matrix<T>, responses : Matrix<T>, (mut lasts_res, mut lasts_cl) : (Vec<Matrix<T>>, Vec<Matrix<T>>)) {
        let weights = self.weights.lock().unwrap(); //| Getting the locks
        let bias = self.bias.lock().unwrap();       //|

        let nb_images : T = NumCast::from(images.lines()).unwrap();
        let correction_coef : T = self.learning_rate / nb_images;

        self.propagation(images);
        let cost = responses + -lasts_res.pop().expect("Error during backpropagation");

        let mut layer_error = cost & lasts_cl.pop().expect("Error during backpropagation").map(|a| (self.derivative_fn)(a));

        weights[self.nb_layers - 1] += (lasts_res.pop().expect("Error during backpropagation").t() * layer_error.clone()) * correction_coef;
        bias[self.nb_layers - 1] += layer_error.clone().sum_line() * correction_coef;

        for i in 1..self.nb_layers {
            layer_error = (layer_error * weights[self.nb_layers - i].t()) & (lasts_cl.pop().expect("Error during backpropagation").map(|a| (self.derivative_fn)(a)));
            weights[self.nb_layers - (i + 1)] += (lasts_res.pop().expect("Error during backpropagation").t() * layer_error.clone()) * correction_coef;
            bias[self.nb_layers - (i + 1)] += layer_error.clone().sum_line() * correction_coef;
        }
    }
}













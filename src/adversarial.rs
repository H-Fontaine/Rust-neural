use std::ops::AddAssign;
use matrix::Matrix;
use num_traits::Float;
use rand::distributions::{Distribution, Standard, Uniform};
use rand::{Rng, thread_rng};
use crate::Network;

pub struct AdversarialNetworks<T> where T : Float {
    generative_network : Network<T>,
    discriminator_network : Network<T>,
}

//CONSTRUCTORS
impl<T : Float> AdversarialNetworks<T> {
    pub fn new<R: ?Sized + Rng, D: Distribution<T>> (generative_shape : Vec<usize>, discriminator_shape : Vec<usize> ,learning_rate : T, rng: &mut R, distribution: &D) -> AdversarialNetworks<T> {
        assert_eq!(generative_shape.last().unwrap(), discriminator_shape.first().unwrap(), "The output size of the generative network must be equal to the input of the discriminator network");
        assert_eq!(*discriminator_shape.last().unwrap(), 2, "The output of the discriminator network must be 2");

        AdversarialNetworks {
            generative_network : Network::new(generative_shape, learning_rate, rng, distribution),
            discriminator_network : Network::new(discriminator_shape, learning_rate, rng, distribution),
        }
    }
}

//USAGE
impl<T : Float> AdversarialNetworks<T> {
    pub fn generate<R: ?Sized + Rng, D: Distribution<T>>(&self, nb_to_generate : usize, rng : &mut R, distribution : D) -> Matrix<T> where T : AddAssign {
        self.generative_network.down(Matrix::new_rand(nb_to_generate, self.get_input_size(), rng, distribution))
    }

    pub fn training<R: ?Sized + Rng, D: Distribution<T>>(&mut self, to_mimic : Matrix<T>, nb_of_batch : usize, half_batch_size : usize, rng : &mut R, distribution : D) where T : AddAssign {
        for _ in 0..nb_of_batch {
            let fakes = self.generate(half_batch_size, rng, &distribution);
            let real = to_mimic.chose_rnd_lines(half_batch_size);
            self.train_discriminator(fakes, real);

            let input = Matrix::new_rand(half_batch_size, self.get_input_size(), rng, &distribution);
            self.train_generative(input);
        }
    }
}

//INTERNAL FUNCTIONALITIES
impl<T : Float> AdversarialNetworks<T> {
    fn train_discriminator(&mut self, fakes : Matrix<T>, real : Matrix<T>) where T : AddAssign {
        let mut responses = Matrix::zeros(fakes.lines() + real.lines(), 2);
        for i in 0..fakes.lines() {
            responses[i][1] = T::one(); //If it is a fake we want index 1 to trigger
        }
        for i in fakes.lines()..responses.lines() {
            responses[i][0] = T::one(); //If it is a real one we want index 0 to trigger
        }
        let input = fakes.concatenate_lines(real);
        self.discriminator_network.simple_train(input, responses);
    }
}


//GETTERS
impl<T : Float> AdversarialNetworks<T> {
    pub fn get_input_size(&self) -> usize {
        self.generative_network.bias.first().unwrap().lines()
    }

}









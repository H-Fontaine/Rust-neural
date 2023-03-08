use std::ops::AddAssign;
use matrix::Matrix;
use num_traits::Float;
use rand::distributions::{Distribution, Open01, Standard, Uniform};
use rand::{Rng, thread_rng};
use rand_distr::StandardNormal;
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
    pub fn generate<D: Distribution<T>>(&self, nb_to_generate : usize, distribution : D) -> Matrix<T> where T : AddAssign {
        let mut rng = thread_rng();
        self.generative_network.down(Matrix::new_rand(nb_to_generate, self.get_input_size(), &mut rng, distribution))
    }
}

//USAGE
impl<T : Float> AdversarialNetworks<T> {

}


//GETTERS
impl<T : Float> AdversarialNetworks<T> {
    pub fn get_input_size(&self) -> usize {
        self.generative_network.bias.first().unwrap().lines()
    }

}
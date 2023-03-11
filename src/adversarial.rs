use matrix::Matrix;
use rand::distributions::{Distribution};
use rand::{Rng};
use crate::Network;

pub struct AdversarialNetworks {
    generative_network : Network,
    discriminator_network : Network,
}

//CONSTRUCTORS
impl<> AdversarialNetworks {
    pub fn new<R: ?Sized + Rng, D: Distribution<f64>> (generative_shape : Vec<usize>, discriminator_shape : Vec<usize>, learning_rate : f64, rng: &mut R, distribution: &D) -> AdversarialNetworks {
        assert_eq!(generative_shape.last().unwrap(), discriminator_shape.first().unwrap(), "The output size of the generative network must be equal to the input of the discriminator network");
        assert_eq!(*discriminator_shape.last().unwrap(), 2, "The output of the discriminator network must be 2");

        AdversarialNetworks {
            generative_network : Network::new(generative_shape, learning_rate, rng, distribution),
            discriminator_network : Network::new(discriminator_shape, learning_rate, rng, distribution),
        }
    }
}

//USAGE
impl AdversarialNetworks {
    pub fn generate<R: ?Sized + Rng, D: Distribution<f64>>(&self, nb_to_generate : usize, rng : &mut R, distribution : D) -> Matrix<f64> {
        self.generative_network.down(Matrix::new_rand(nb_to_generate, self.get_input_size(), rng, distribution))
    }

    pub fn training<R: ?Sized + Rng, D: Distribution<f64>>(&mut self, to_mimic : Matrix<f64>, nb_of_batch : usize, half_batch_size : usize, rng : &mut R, distribution : D) {
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
impl AdversarialNetworks {
    fn train_discriminator(&mut self, fakes : Matrix<f64>, real : Matrix<f64>) {
        let mut responses = Matrix::zeros(fakes.lines() + real.lines(), 2);
        for i in 0..fakes.lines() {
            responses[i][1] = 1f64; //If it is a fake we want index 1 to trigger
        }
        for i in fakes.lines()..responses.lines() {
            responses[i][0] = 1f64; //If it is a real one we want index 0 to trigger
        }
        let input = fakes.concatenate_lines(real);
        self.discriminator_network.simple_train(input, responses);
    }

    fn train_generative(&mut self, input : Matrix<f64>) {
        let correction_coef = self.get_learning_rate() / (input.lines() as f64);
        let mut responses = Matrix::zeros(input.lines(), 2);
        for i in 0..responses.lines() {
            responses[i][0] = 1f64; //We want to maximise discriminator error so we want 0 to trigger (because we want the input to be detected as a real one)
        }

        let (mut lasts_res_generative, mut lasts_cl_generative) = self.generative_network.propagation(input);
        let (lasts_res_discriminator, lasts_cl_discriminator) = self.discriminator_network.propagation(lasts_res_generative.pop().unwrap()); //Moving the result of the generative network into lasts_res_discriminator.first()
        let mut last_cl_discriminator_augmented = vec![lasts_cl_generative.pop().unwrap()];
        last_cl_discriminator_augmented.extend(lasts_cl_discriminator);
        let cost_generative = self.discriminator_network.compute_first_layer_error(responses,(lasts_res_discriminator, last_cl_discriminator_augmented));
        let gradient_generative = self.generative_network.gradient_from_layer_error(correction_coef,cost_generative, (lasts_res_generative, lasts_cl_generative));
        self.generative_network.correction(gradient_generative);
    }
}


//GETTERS
impl AdversarialNetworks {
    pub fn get_input_size(&self) -> usize {
        self.generative_network.weights.first().unwrap().lines()
    }

    pub fn get_learning_rate(&self) -> f64 {
        self.generative_network.learning_rate
    }
}









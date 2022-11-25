use matrix::Matrix;
use mnist;

fn main() {
    let images_test : Matrix<f32> = mnist::load_images("dataset/t10k-images.idx3-ubyte");
    let images_train : Matrix<f32> = mnist::load_images("dataset/train-images.idx3-ubyte");
    let labels_test = mnist::load_labels("dataset/t10k-labels.idx1-ubyte");
    let labels_train = mnist::load_labels("dataset/train-labels.idx1-ubyte");

    mnist::display(&images_train, &labels_train, 10);
}
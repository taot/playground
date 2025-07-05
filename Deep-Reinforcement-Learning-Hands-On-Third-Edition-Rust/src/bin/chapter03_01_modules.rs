use burn::{
    module::Module,
    nn::{Linear, LinearConfig, Dropout, DropoutConfig, Relu},
    tensor::{backend::Backend, Tensor},
};
use burn::tensor::activation::softmax;

#[derive(Module, Debug)]
pub struct OurModule<B: Backend> {
    linear1: Linear<B>,
    relu1: Relu,
    linear2: Linear<B>,
    relu2: Relu,
    linear3: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> OurModule<B> {
    pub fn new(device: &B::Device, num_inputs: usize, num_classes: usize, dropout_prob: f64) -> Self {
        let linear1 = LinearConfig::new(num_inputs, 5).init(device);
        let relu1 = Relu::new();
        let linear2 = LinearConfig::new(5, 20).init(device);
        let relu2 = Relu::new();
        let linear3 = LinearConfig::new(20, num_classes).init(device);
        let dropout = DropoutConfig::new(dropout_prob).init();

        Self {
            linear1,
            relu1,
            linear2,
            relu2,
            linear3,
            dropout,
        }
    }

    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let x = self.linear1.forward(input);
        let x = self.relu1.forward(x);
        let x = self.linear2.forward(x);
        let x = self.relu2.forward(x);
        let x = self.linear3.forward(x);
        let x = self.dropout.forward(x);
        softmax(x, 1)
    }
}

fn main() {
    use burn::backend::{ndarray::NdArray, Autodiff};

    type Backend = Autodiff<NdArray>;
    let device = Default::default();

    let model: OurModule<Backend> = OurModule::new(&device, 2, 3, 0.3);
    let input = Tensor::<Backend, 2>::from_floats([[2, 3]], &device);
    let output = model.forward(input);
    println!("{}", output);
}

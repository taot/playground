use anyhow::Result;
use burn::{
    backend::{ndarray::NdArray, Autodiff}, // Using NdArrayBackend for simplicity
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    tensor::{backend::Backend, Tensor},
};

use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::Int;
use burn::tensor::activation::softmax;
use burn::tensor::TensorData;
use burn_tch::LibTorch;
use deep_reinforcement_learning_hands_on_third_edition_rust::create_summary_writer;
use gym::environment::Environment;
use gym::space_template::SpaceTemplate;
use gym::{Action, State};
use gym::client::RenderMode;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::Rng;

const HIDDEN_SIZE: usize = 128;
const BATCH_SIZE: usize = 16;
const PERCENTILE: f64 = 70.0;


#[derive(Debug, Clone)]
pub struct EpisodeStep {
    pub observation: ndarray::Array<f64, ndarray::Ix1>,
    pub action: usize,
}

#[derive(Debug, Clone)]
pub struct Episode {
    pub reward: f64,
    pub steps: Vec<EpisodeStep>,
}

#[derive(Module, Debug)]
pub struct Net<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    relu: Relu,
}

#[derive(Config, Debug)]
pub struct NetConfig {
    obs_size: usize,
    hidden_size: usize,
    n_actions: usize,
}

impl NetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Net<B> {
        Net {
            linear1: LinearConfig::new(self.obs_size, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.n_actions).init(device),
            relu: Relu::new(),
        }
    }
}

impl<B: Backend> Net<B> {
    pub fn forward(&self, xs: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(xs);
        let x = self.relu.forward(x);
        self.linear2.forward(x)
    }
}

pub struct BatchRunner<'a> {
    env: &'a Environment<'a>,
    batch_size: usize,
    obs: ndarray::Array1<f64>,
}

impl<'a> BatchRunner<'a> {
    pub fn new(env: &'a Environment<'a>, batch_size: usize) -> Self {
        let (obs, _) = env.reset(None).unwrap();
        Self {
            env,
            batch_size,
            obs: obs.get_box().unwrap(),
        }
    }

    pub fn next_batch<B: Backend>(&mut self, net: &Net<B>, device: &B::Device) -> Vec<Episode> {
        let mut episode_reward = 0.0;
        let mut episode_steps = Vec::new();
        let mut batch = Vec::with_capacity(self.batch_size);

        loop {
            let obs_v: Tensor<B, 1> = Tensor::from_data(
                TensorData::new(self.obs.to_vec(), self.obs.shape().to_vec()),
                device,
            );
            // println!("obs_v: {:?}", obs_v);

            let act_probs_v = softmax(net.forward(obs_v.unsqueeze()), 1);
            // println!("act_probs_v: {:?}", act_probs_v);

            let act_probs = act_probs_v
                .squeeze::<1>(0)
                .to_data()
                .to_vec::<f32>()
                .unwrap();
            // println!("act_probs: {:?}", act_probs);

            let act_dist = WeightedIndex::new(&act_probs).unwrap();
            // println!("act_dist: {:?}", act_dist);

            let mut rng = rand::rng();
            let action = rng.sample(act_dist);
            // println!("action: {:?}", action);

            let State {
                observation: next_obs,
                reward,
                is_done,
                is_truncated,
            } = self.env.step(&Action::Discrete(action)).unwrap();
            // println!("next_obs: {:?}, reward: {:?}, is_done: {:?}, is_truncated: {:?}", next_obs, reward, is_done, is_truncated);

            episode_reward += reward;
            let step = EpisodeStep {
                observation: self.obs.clone(),
                action,
            };
            episode_steps.push(step);

            self.obs = next_obs.get_box().unwrap();

            if is_done || is_truncated {
                let episode = Episode {
                    reward: episode_reward,
                    steps: episode_steps.clone(),
                };
                batch.push(episode);

                let (obs, _) = self.env.reset(None).unwrap();
                self.obs = obs.get_box().unwrap();
                episode_reward = 0.0;
                episode_steps = Vec::new();


                if batch.len() == self.batch_size {
                    let r = batch.clone();
                    batch = Vec::with_capacity(self.batch_size);
                    return r;
                }
            }
        }
    }
}

fn compute_percentile(data: &[f64], percentile: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mut sorted_data = data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let index = (percentile / 100.0) * (sorted_data.len() - 1) as f64;
    let lower_index = index.floor() as usize;
    let upper_index = index.ceil() as usize;

    if lower_index == upper_index {
        sorted_data[lower_index]
    } else {
        let weight = index - lower_index as f64;
        sorted_data[lower_index] * (1.0 - weight) + sorted_data[upper_index] * weight
    }
}

fn compute_mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

fn filter_batch<B: Backend>(
    batch: Vec<Episode>,
    percentile: f64,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 1, Int>, f64, f64) {
    let rewards = batch.iter().map(|e| e.reward).collect::<Vec<f64>>();
    let reward_bound = compute_percentile(&rewards, percentile);
    let reward_mean = compute_mean(&rewards);

    let mut train_obs: Vec<Tensor<B, 1>> = Vec::new();
    let mut train_act: Vec<i64> = Vec::new();

    for episode in batch {
        if episode.reward < reward_bound {
            continue;
        }

        for step in episode.steps {
            let tensor_obs =
                Tensor::<B, 1>::from_data(step.observation.as_slice().unwrap(), device);
            train_obs.push(tensor_obs);
            train_act.push(step.action as i64);
        }
    }

    let train_obs_v = Tensor::stack(train_obs, 0);
    let train_act_v =
        Tensor::<B, 1, Int>::from_data(TensorData::from(train_act.as_slice()), device);

    (train_obs_v, train_act_v, reward_bound, reward_mean)
}

fn main() -> Result<()> {
    let gym_client = gym::client::GymClient::default();
    let env = gym_client
        .make(
            "CartPole-v1",
            Some(gym::client::MakeOptions {
                // render_mode: Some(RenderMode::Human),
                ..Default::default()
            }),
        )
        .expect("Unable to create environment");

    let obs_size = match env.observation_space() {
        obs_space @ SpaceTemplate::Box {
            high: _,
            low: _,
            shape,
        } => {
            println!("Observation space: {:#?}", obs_space);
            shape[0]
        }
        _ => {
            return Err(anyhow::anyhow!("Invalid observation space"));
        }
    };

    let n_actions = match env.action_space() {
        action_space @ SpaceTemplate::Discrete { n } => {
            println!("Action space: {:#?}", action_space);
            *n
        }
        _ => {
            return Err(anyhow::anyhow!("Invalid action space"));
        }
    } as usize;

    let (state, _) = env.reset(None)?;
    println!("Initial state: {:#?}", state);
    println!("{}", state.get_box()?);

    let net_config = NetConfig {
        obs_size,
        hidden_size: HIDDEN_SIZE,
        n_actions,
    };

    // type Backend = Autodiff<NdArray>;
    type Backend = Autodiff<LibTorch>;
    let device = Default::default();
    let mut net = net_config.init::<Backend>(&device);
    println!("Net: {}", net);

    let objective = CrossEntropyLossConfig::new().init::<Backend>(&device);
    let mut optimizer = AdamConfig::new().init::<Backend, Net<Backend>>();

    let mut writer = create_summary_writer();

    let mut batch_runner = BatchRunner::new(&env, BATCH_SIZE);
    let mut iter_no = 0;

    loop {
        let batch = batch_runner.next_batch(&net, &device);

        let (train_obs_v, train_act_v, reward_bound, reward_mean) =
            filter_batch::<Backend>(batch, PERCENTILE, &device);

        let action_scores_v = net.forward(train_obs_v);
        let loss_v = objective.forward(action_scores_v, train_act_v);
        let grads = GradientsParams::from_grads(loss_v.backward(), &net);
        net = optimizer.step(0.01, net, grads);

        iter_no += 1;
        println!(
            "Iter: {}, Reward mean: {}, Reward bound: {}, Loss: {}",
            iter_no,
            reward_mean,
            reward_bound,
            loss_v.to_data().to_vec::<f32>().unwrap()[0]
        );

        if reward_mean > 475.0 {
            break;
        }
    }

    Ok(())
}

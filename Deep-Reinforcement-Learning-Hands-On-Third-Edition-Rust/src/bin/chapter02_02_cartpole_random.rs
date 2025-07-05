use gym::client::MakeOptions;

extern crate gym;

fn main() {
    let gym = gym::client::GymClient::default();
    let env = gym
        .make(
            "CartPole-v1",
            Some(MakeOptions {
                render_mode: Some(gym::client::RenderMode::Human),
                ..Default::default()
            }),
        )
        .expect("Unable to create environment");

    let mut total_reward = 0.0;
    let mut total_steps = 0;

    let (obs, _) = env.reset(None).unwrap();
    println!("Fist observation: {:?}", obs);

    loop {
        let action = env.action_space().sample();
        let state = env.step(&action).unwrap();
        total_reward += state.reward;
        total_steps += 1;

        if state.is_done || state.is_truncated {
            let reason = if state.is_done {
                "done"
            } else {
                "truncated"
            };
            println!("Episode terminated ({})", reason);
            break;
        }
    }

    println!("Episode done in {} steps, total reward: {}", total_steps, total_reward);

    env.close();
}

use deep_reinforcement_learning_hands_on_third_edition_rust::create_summary_writer;

fn main() {
    let mut writer = create_summary_writer();

    for (i, angle) in (-360..720).enumerate() {
        let angle_rad = (angle as f32).to_radians();
        writer.add_scalar("sin", angle_rad.sin(), i);
        writer.add_scalar("cos", angle_rad.cos(), i);
        writer.add_scalar("tan", angle_rad.tan(), i);
    }

    writer.flush();
}

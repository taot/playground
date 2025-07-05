use tensorboard_rs::summary_writer::SummaryWriter;

pub fn create_summary_writer() -> SummaryWriter {
    let now = time::OffsetDateTime::now_local().unwrap();
    let s = format!(
        "{:04}-{:02}-{:02}_{:02}-{:02}-{:02}_{}",
        now.year(),
        now.month() as u8,
        now.day(),
        now.hour(),
        now.minute(),
        now.second(),
        hostname::get().unwrap_or_default().to_string_lossy()
    );
    let writer = SummaryWriter::new(format!("./runs/{}", s));
    writer
}

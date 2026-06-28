use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::{filter::LevelFilter, fmt, prelude::*, registry::Registry};

pub struct Logger {}

impl Logger {
    pub async fn init(trace: bool) {
        let file_appender =
            RollingFileAppender::new(Rotation::HOURLY, "./.astrophage/logs", "ap.log");
        let (non_blocking_file, _guard) = tracing_appender::non_blocking(file_appender);

        let level = if trace {
            LevelFilter::DEBUG
        } else {
            LevelFilter::INFO
        };

        let stdout_layer = fmt::layer().with_writer(std::io::stdout).with_filter(level);

        let file_layer = fmt::layer()
            .with_writer(non_blocking_file)
            .with_filter(LevelFilter::TRACE);

        let subscriber = Registry::default().with(stdout_layer).with(file_layer);

        tracing::subscriber::set_global_default(subscriber).expect("Failed to set subscriber");
    }
}

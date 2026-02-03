use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use nailbite::app::App;
use nailbite::config::NailbiteConfig;

#[derive(Parser, Debug)]
#[command(name = "nailbite", about = "BFRB detection and decoupling exercise system")]
struct Cli {
    /// Path to the configuration file
    #[arg(long, default_value = "config.yaml")]
    config: String,

    /// Show camera preview window on startup
    #[arg(long)]
    show_preview: bool,
}

fn main() -> Result<(), nailbite::errors::NailbiteError> {
    let cli = Cli::parse();

    let config = NailbiteConfig::load(&cli.config)?;

    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                format!("{},tower_http=info", config.general.log_level).into()
            }),
        )
        .init();

    tracing::info!("nailbite starting");
    tracing::debug!(?config, "loaded configuration");

    let app = App::new(config);
    app.run(cli.show_preview)?;

    Ok(())
}

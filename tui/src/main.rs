mod api;
mod app;
mod config;
mod crypto;
mod theme;
mod ui;

use std::io::stdout;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use ratatui::crossterm::event::{DisableBracketedPaste, EnableBracketedPaste};
use ratatui::crossterm::execute;
use ratatui_image::picker::{Picker, ProtocolType};

use crate::app::App;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ProtocolChoice {
    /// Detect the best protocol supported by the terminal.
    Auto,
    Kitty,
    Sixel,
    Iterm2,
    Halfblocks,
}

impl ProtocolChoice {
    fn forced(self) -> Option<ProtocolType> {
        match self {
            ProtocolChoice::Auto => None,
            ProtocolChoice::Kitty => Some(ProtocolType::Kitty),
            ProtocolChoice::Sixel => Some(ProtocolType::Sixel),
            ProtocolChoice::Iterm2 => Some(ProtocolType::Iterm2),
            ProtocolChoice::Halfblocks => Some(ProtocolType::Halfblocks),
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "imagineit-tui",
    version,
    about = "Generate images on an ImagineIt server from your terminal"
)]
struct Cli {
    /// ImagineIt server base URL.
    #[arg(
        short,
        long,
        env = "IMAGINEIT_URL",
        default_value = "http://localhost:4600"
    )]
    url: String,

    /// Force an image protocol instead of detecting it.
    #[arg(long, value_enum, default_value_t = ProtocolChoice::Auto)]
    protocol: ProtocolChoice,

    /// Account file used to persist the token and private key.
    #[arg(long, env = "IMAGINEIT_CONFIG")]
    config: Option<PathBuf>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("imagineit-tui: {err:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    let config_path = cli.config.unwrap_or_else(config::default_path);

    let mut terminal = ratatui::try_init().context("could not initialise the terminal")?;

    // Query graphics support after entering the alternate screen but before
    // crossterm starts consuming input events.
    let query = Picker::from_query_stdio();
    let query_error = query.as_ref().err().map(ToString::to_string);
    let mut picker = query.unwrap_or_else(|_| Picker::halfblocks());
    if let Some(protocol) = cli.protocol.forced() {
        picker.set_protocol_type(protocol);
    }
    let startup_notice = query_error.map(|err| match cli.protocol.forced() {
        Some(protocol) => format!(
            "Could not query terminal graphics support ({err}); using the forced \
             {protocol:?} protocol with a fallback cell size."
        ),
        None => format!(
            "Could not query terminal graphics support ({err}); using halfblocks. \
             Try --protocol kitty if images look wrong."
        ),
    });

    let _ = execute!(stdout(), EnableBracketedPaste);
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        let _ = execute!(stdout(), DisableBracketedPaste);
        previous_hook(panic_info);
    }));

    let run_result = match App::new(cli.url, config_path, picker, startup_notice) {
        Ok(mut app) => app.run(&mut terminal),
        Err(err) => Err(err),
    };

    let _ = execute!(stdout(), DisableBracketedPaste);
    let restore_result = ratatui::try_restore();
    run_result?;
    restore_result?;
    Ok(())
}

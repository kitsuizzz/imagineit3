use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use directories::{BaseDirs, UserDirs};
use rand::Rng;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::style::{Modifier, Style};
use ratatui::DefaultTerminal;
use ratatui_image::picker::Picker;
use ratatui_image::protocol::StatefulProtocol;
use ratatui_textarea::{CursorMove, Input, TextArea, WrapMode};
use tokio::runtime::{Handle, Runtime};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

use crate::api::{
    self, Account, AppEvent, GeneratedImage, GenerationParams, KaggleAction, KaggleProfile,
    ProgressUpdate, WorkerInfo, WorkerKeyMaterial,
};
use crate::config;
use crate::theme;

pub const SCHEDULERS: [&str; 9] = [
    "euler",
    "euler_a",
    "unipc",
    "dpm",
    "dpm_singlestep",
    "heun",
    "lms",
    "kdpm2",
    "kdpm2_a",
];

const WORKER_POLL_INTERVAL: Duration = Duration::from_secs(5);

pub struct App {
    pub screen: Screen,
    pub server: String,
    pub config_path: PathBuf,
    pub picker: Picker,
    pub protocol_label: String,
    pub account: Option<Account>,
    pub form: Form,
    pub preview: Option<Preview>,
    pub active_job: Option<ActiveJob>,
    pub status: Status,
    pub overlay: Overlay,
    pub auth_error: Option<String>,
    pub should_quit: bool,
    pub tick: u64,
    pub workers: Vec<WorkerInfo>,
    pub workers_loading: bool,
    pub workers_loaded: bool,
    pub workers_error: Option<String>,
    pub worker_key: Option<WorkerKeySetup>,
    pub worker_key_loading: bool,
    pub worker_key_error: Option<String>,
    pub kaggle: KaggleState,

    last_workers_poll: Option<Instant>,
    client: reqwest::Client,
    handle: Handle,
    events_tx: UnboundedSender<AppEvent>,
    events_rx: UnboundedReceiver<AppEvent>,
    _runtime: Runtime,
}

pub struct WorkerKeySetup {
    pub worker_id: String,
    pub key_id: String,
    pub path: PathBuf,
}

pub struct KaggleState {
    pub token: TextArea<'static>,
    pub profiles: Vec<KaggleProfile>,
    pub active: Option<String>,
    pub profiles_loading: bool,
    pub busy: Option<KaggleAction>,
    pub confirm_delete: bool,
    pub error: Option<String>,
    pub report: Option<String>,
}

pub enum Screen {
    Auth(AuthScreen),
    Main,
}

pub enum AuthScreen {
    Menu {
        selected: usize,
    },
    SignIn {
        token: Box<TextArea<'static>>,
        busy: bool,
    },
    Busy {
        message: String,
    },
}

pub enum Overlay {
    None,
    Help,
    Workers,
    Kaggle,
    Token(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusKind {
    Muted,
    Info,
    Success,
    Error,
}

pub struct Status {
    pub text: String,
    pub kind: StatusKind,
}

impl Status {
    fn new(text: impl Into<String>, kind: StatusKind) -> Self {
        Self {
            text: text.into(),
            kind,
        }
    }
}

pub struct Preview {
    pub protocol: StatefulProtocol,
    pub bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub seed: u64,
    pub scheduler: String,
    pub elapsed: Duration,
    pub promise_id: String,
}

pub struct ActiveJob {
    pub promise_id: Option<String>,
    pub started: Instant,
    pub params: GenerationParams,
    pub status: String,
    pub steps_completed: u64,
    pub steps_total: u64,
}

impl ActiveJob {
    pub fn ratio(&self) -> f64 {
        if self.steps_total == 0 {
            0.0
        } else {
            (self.steps_completed as f64 / self.steps_total as f64).clamp(0.0, 1.0)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Field {
    Prompt,
    Negative,
    Width,
    Height,
    Steps,
    Cfg,
    Seed,
    Clip,
    Scheduler,
    Generate,
}

impl Field {
    pub const ALL: [Field; 10] = [
        Field::Prompt,
        Field::Negative,
        Field::Width,
        Field::Height,
        Field::Steps,
        Field::Cfg,
        Field::Seed,
        Field::Clip,
        Field::Scheduler,
        Field::Generate,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Field::Prompt => "Prompt",
            Field::Negative => "Negative",
            Field::Width => "Width",
            Field::Height => "Height",
            Field::Steps => "Steps",
            Field::Cfg => "CFG",
            Field::Seed => "Seed",
            Field::Clip => "CLIP",
            Field::Scheduler => "Sampler",
            Field::Generate => "Generate",
        }
    }

    fn is_textarea(self) -> bool {
        !matches!(self, Field::Scheduler | Field::Generate)
    }
}

pub struct Form {
    pub prompt: TextArea<'static>,
    pub negative: TextArea<'static>,
    pub width: TextArea<'static>,
    pub height: TextArea<'static>,
    pub steps: TextArea<'static>,
    pub cfg: TextArea<'static>,
    pub seed: TextArea<'static>,
    pub clip: TextArea<'static>,
    pub scheduler: usize,
    pub focus: Field,
}

impl Form {
    fn new() -> Self {
        let mut prompt = TextArea::default();
        prompt.set_wrap_mode(WrapMode::WordOrGlyph);
        prompt.set_cursor_line_style(Style::default());
        prompt.set_cursor_style(Style::default().add_modifier(Modifier::REVERSED));
        prompt.set_placeholder_text("Describe the image you want to create…");
        prompt.set_placeholder_style(
            Style::default()
                .fg(theme::MUTED)
                .add_modifier(Modifier::ITALIC),
        );

        let mut negative = TextArea::default();
        negative.set_wrap_mode(WrapMode::WordOrGlyph);
        negative.set_cursor_line_style(Style::default());
        negative.set_cursor_style(Style::default());
        negative.set_placeholder_text("Things to avoid (optional)");
        negative.set_placeholder_style(
            Style::default()
                .fg(theme::MUTED)
                .add_modifier(Modifier::ITALIC),
        );

        Self {
            prompt,
            negative,
            width: single_line("1024", ""),
            height: single_line("1024", ""),
            steps: single_line("28", ""),
            cfg: single_line("5.0", ""),
            seed: single_line("", "random"),
            clip: single_line("2", ""),
            scheduler: SCHEDULERS
                .iter()
                .position(|name| *name == "euler_a")
                .unwrap_or(0),
            focus: Field::Prompt,
        }
    }

    pub fn refresh_cursor_styles(&mut self) {
        let focus = self.focus;
        for field in Field::ALL {
            let Some(textarea) = self.textarea_mut(field) else {
                continue;
            };
            let style = if focus == field {
                Style::default().add_modifier(Modifier::REVERSED)
            } else {
                Style::default()
            };
            textarea.set_cursor_style(style);
        }
    }

    fn textarea_mut(&mut self, field: Field) -> Option<&mut TextArea<'static>> {
        match field {
            Field::Prompt => Some(&mut self.prompt),
            Field::Negative => Some(&mut self.negative),
            Field::Width => Some(&mut self.width),
            Field::Height => Some(&mut self.height),
            Field::Steps => Some(&mut self.steps),
            Field::Cfg => Some(&mut self.cfg),
            Field::Seed => Some(&mut self.seed),
            Field::Clip => Some(&mut self.clip),
            Field::Scheduler | Field::Generate => None,
        }
    }

    pub fn textarea(&self, field: Field) -> Option<&TextArea<'static>> {
        match field {
            Field::Prompt => Some(&self.prompt),
            Field::Negative => Some(&self.negative),
            Field::Width => Some(&self.width),
            Field::Height => Some(&self.height),
            Field::Steps => Some(&self.steps),
            Field::Cfg => Some(&self.cfg),
            Field::Seed => Some(&self.seed),
            Field::Clip => Some(&self.clip),
            Field::Scheduler | Field::Generate => None,
        }
    }

    pub fn focus_next(&mut self, backwards: bool) {
        let index = Field::ALL
            .iter()
            .position(|field| *field == self.focus)
            .unwrap_or(0);
        let next = if backwards {
            (index + Field::ALL.len() - 1) % Field::ALL.len()
        } else {
            (index + 1) % Field::ALL.len()
        };
        self.focus = Field::ALL[next];
    }

    pub fn insert_paste(&mut self, text: &str) {
        let field = self.focus;
        let Some(textarea) = self.textarea_mut(field) else {
            return;
        };
        let multiline = matches!(field, Field::Prompt | Field::Negative);
        let clean = if multiline {
            text.to_string()
        } else {
            text.chars()
                .map(|c| if c == '\r' || c == '\n' { ' ' } else { c })
                .collect()
        };
        textarea.insert_str(clean);
    }

    pub fn cycle_scheduler(&mut self, forward: bool) {
        let len = SCHEDULERS.len();
        self.scheduler = if forward {
            (self.scheduler + 1) % len
        } else {
            (self.scheduler + len - 1) % len
        };
    }

    pub fn scheduler(&self) -> &'static str {
        SCHEDULERS[self.scheduler.min(SCHEDULERS.len() - 1)]
    }

    fn adjust_focused(&mut self, direction: i64) -> bool {
        let field = self.focus;
        let (min, max, step, decimals, fallback) = match field {
            Field::Width | Field::Height => (64.0, 2048.0, 64.0, 0, 1024.0),
            Field::Steps => (1.0, 150.0, 1.0, 0, 28.0),
            Field::Cfg => (0.0, 20.0, 0.1, 1, 5.0),
            Field::Clip => (1.0, 12.0, 1.0, 0, 2.0),
            Field::Seed => (0.0, 1_000_000_000.0, 1.0, 0, 0.0),
            _ => return false,
        };

        let Some(textarea) = self.textarea_mut(field) else {
            return false;
        };
        let current_text = one_line(textarea);
        if field == Field::Seed && current_text.trim().is_empty() {
            let seed = rand::thread_rng().gen_range(0..1_000_000_000u64);
            set_value(textarea, &seed.to_string());
            return true;
        }
        let current = current_text.parse::<f64>().unwrap_or(fallback);
        let next = (current + direction as f64 * step).clamp(min, max);
        let rendered = if decimals == 0 {
            format!("{next:.0}")
        } else {
            format!("{next:.1}")
        };
        set_value(textarea, &rendered);
        true
    }

    pub fn validate(&mut self) -> std::result::Result<GenerationParams, (Field, String)> {
        let prompt = text_of(&self.prompt).trim().to_string();
        if prompt.is_empty() {
            self.focus = Field::Prompt;
            return Err((Field::Prompt, "Write a prompt first.".to_string()));
        }

        let negative = text_of(&self.negative).trim().to_string();
        let negative_prompt = if negative.is_empty() {
            None
        } else {
            Some(negative)
        };

        let width = self
            .parse_dimension(Field::Width, 1024, "Width")
            .map_err(|message| (Field::Width, message))?;
        let height = self
            .parse_dimension(Field::Height, 1024, "Height")
            .map_err(|message| (Field::Height, message))?;
        let steps = self
            .parse_integer::<u16>(Field::Steps, 28, 1, 150, "Steps")
            .map_err(|message| (Field::Steps, message))?;
        let clip_skip = self
            .parse_integer::<u8>(Field::Clip, 2, 1, 12, "CLIP skip")
            .map_err(|message| (Field::Clip, message))?;

        let cfg = {
            let text = one_line(&self.cfg);
            let value = if text.trim().is_empty() {
                5.0
            } else {
                text.trim().parse::<f32>().map_err(|_| {
                    (
                        Field::Cfg,
                        "CFG must be a number between 0 and 20.".to_string(),
                    )
                })?
            };
            if !value.is_finite() || !(0.0..=20.0).contains(&value) {
                return Err((
                    Field::Cfg,
                    "CFG must be a number between 0 and 20.".to_string(),
                ));
            }
            value
        };

        let seed_text = one_line(&self.seed);
        let seed = if seed_text.trim().is_empty() {
            rand::thread_rng().gen_range(0..1_000_000_000u64)
        } else {
            let seed = seed_text.trim().parse::<u64>().map_err(|_| {
                (
                    Field::Seed,
                    "Seed must be a whole number, or empty for a random seed.".to_string(),
                )
            })?;
            if seed > 1_000_000_000 {
                return Err((
                    Field::Seed,
                    "Seed must be between 0 and 1,000,000,000.".to_string(),
                ));
            }
            seed
        };

        Ok(GenerationParams {
            prompt,
            negative_prompt,
            width,
            height,
            cfg,
            steps,
            seed,
            clip_skip,
            scheduler: self.scheduler().to_string(),
        })
    }

    fn parse_dimension(
        &mut self,
        field: Field,
        default: u16,
        name: &str,
    ) -> std::result::Result<u16, String> {
        let Some(textarea) = self.textarea_mut(field) else {
            return Err(format!("{name} is not editable."));
        };
        let text = one_line(textarea);
        let raw = if text.trim().is_empty() {
            default
        } else {
            text.trim()
                .parse::<u16>()
                .map_err(|_| format!("{name} must be a whole number from 64 to 2048."))?
        };
        if !(64..=2048).contains(&raw) {
            return Err(format!("{name} must be from 64 to 2048."));
        }
        let snapped = ((raw as f32 / 8.0).round() * 8.0) as u16;
        let snapped = snapped.clamp(64, 2048);
        if snapped != raw {
            set_value(textarea, &snapped.to_string());
        }
        Ok(snapped)
    }

    fn parse_integer<T>(
        &mut self,
        field: Field,
        default: T,
        min: T,
        max: T,
        name: &str,
    ) -> std::result::Result<T, String>
    where
        T: std::str::FromStr + Copy + PartialOrd + std::fmt::Display,
    {
        let Some(textarea) = self.textarea_mut(field) else {
            return Err(format!("{name} is not editable."));
        };
        let text = one_line(textarea);
        let value = if text.trim().is_empty() {
            default
        } else {
            text.trim()
                .parse::<T>()
                .map_err(|_| format!("{name} must be a whole number from {min} to {max}."))?
        };
        if value < min || value > max {
            return Err(format!("{name} must be from {min} to {max}."));
        }
        Ok(value)
    }
}

impl Default for Form {
    fn default() -> Self {
        Self::new()
    }
}

fn single_line(value: &str, placeholder: &str) -> TextArea<'static> {
    let mut textarea = TextArea::new(vec![value.to_string()]);
    textarea.set_wrap_mode(WrapMode::None);
    textarea.set_cursor_line_style(Style::default());
    textarea.set_cursor_style(Style::default());
    if !placeholder.is_empty() {
        textarea.set_placeholder_text(placeholder);
        textarea.set_placeholder_style(
            Style::default()
                .fg(theme::MUTED)
                .add_modifier(Modifier::ITALIC),
        );
    }
    textarea
}

fn token_input() -> TextArea<'static> {
    let mut textarea = single_line("", "imi_… paste your account token");
    textarea.set_cursor_style(Style::default().add_modifier(Modifier::REVERSED));
    textarea
}

fn kaggle_token_input() -> TextArea<'static> {
    let mut textarea = single_line("", "Paste a Kaggle API key (KGAT_…)");
    textarea.set_mask_char('•');
    textarea.set_cursor_style(Style::default().add_modifier(Modifier::REVERSED));
    textarea
}

impl KaggleState {
    fn new() -> Self {
        Self {
            token: kaggle_token_input(),
            profiles: Vec::new(),
            active: None,
            profiles_loading: false,
            busy: None,
            confirm_delete: false,
            error: None,
            report: None,
        }
    }

    fn apply_profiles(
        &mut self,
        profiles: Vec<KaggleProfile>,
        preferred_id: Option<String>,
        message: Option<String>,
        action: KaggleAction,
    ) {
        self.profiles = profiles
            .into_iter()
            .filter(|profile| {
                profile.kind.eq_ignore_ascii_case("kaggle") || profile.kind.is_empty()
            })
            .collect();
        self.profiles_loading = false;
        self.busy = None;
        self.confirm_delete = false;
        self.error = None;

        let preferred =
            preferred_id.filter(|id| self.profiles.iter().any(|profile| &profile.id == id));
        let current = self
            .active
            .clone()
            .filter(|id| self.profiles.iter().any(|profile| &profile.id == id));
        self.active = preferred.or(current).or_else(|| {
            self.profiles
                .iter()
                .max_by(|left, right| {
                    left.created_at
                        .as_deref()
                        .unwrap_or("")
                        .cmp(right.created_at.as_deref().unwrap_or(""))
                })
                .map(|profile| profile.id.clone())
        });

        if let Some(message) = message {
            self.report = Some(message);
        }
        if action == KaggleAction::Create {
            self.token.clear();
        }
        self.refresh_placeholder();
    }

    fn refresh_placeholder(&mut self) {
        self.token
            .set_placeholder_text(if self.profiles.is_empty() {
                "Paste a Kaggle API key (KGAT_…)"
            } else {
                "Paste a key to add another Kaggle profile"
            });
    }

    pub fn active_profile(&self) -> Option<&KaggleProfile> {
        self.active
            .as_deref()
            .and_then(|id| self.profiles.iter().find(|profile| profile.id == id))
            .or_else(|| self.profiles.first())
    }

    pub fn token_value(&self) -> String {
        self.token.lines().join("").trim().to_string()
    }
}

fn one_line(textarea: &TextArea<'_>) -> String {
    textarea.lines().join("")
}

fn text_of(textarea: &TextArea<'_>) -> String {
    textarea.lines().join("\n")
}

fn set_value(textarea: &mut TextArea<'_>, value: &str) {
    textarea.select_all();
    textarea.insert_str(value);
    textarea.move_cursor(CursorMove::End);
}

impl App {
    pub fn new(
        server: String,
        config_path: PathBuf,
        picker: Picker,
        startup_notice: Option<String>,
    ) -> Result<Self> {
        let server = normalise_server(&server);
        let protocol_label = format!("{:?}", picker.protocol_type()).to_lowercase();
        let client = api::build_client()?;
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .context("could not start the async runtime")?;
        let handle = runtime.handle().clone();
        let (events_tx, events_rx) = unbounded_channel();

        // The hosted deployment exposes the app through Caddy's `/ui-api`
        // bridge. Probe once so bare hostnames and explicit prefixes both work,
        // and so HTTP redirects are folded into the saved server URL.
        let server = runtime.block_on(api::resolve_server(&server, &client));

        let mut status = Status::new("Ready.", StatusKind::Muted);
        let mut auth_error = None;
        let mut screen = Screen::Auth(AuthScreen::Menu { selected: 0 });
        let mut account = None;

        match config::load(&config_path) {
            Ok(Some(stored)) => {
                if stored.server == server {
                    account = Some(stored);
                    screen = Screen::Main;
                } else {
                    auth_error = Some(format!(
                        "A saved account exists for {}, not {}.",
                        stored.server, server
                    ));
                }
            }
            Ok(None) => {}
            Err(err) => {
                auth_error = Some(format!("{err:#}"));
            }
        }

        if let Some(notice) = startup_notice {
            status = Status::new(notice, StatusKind::Info);
        }

        let mut app = Self {
            screen,
            server,
            config_path,
            picker,
            protocol_label,
            account,
            form: Form::new(),
            preview: None,
            active_job: None,
            status,
            overlay: Overlay::None,
            auth_error,
            should_quit: false,
            tick: 0,
            workers: Vec::new(),
            workers_loading: false,
            workers_loaded: false,
            workers_error: None,
            worker_key: None,
            worker_key_loading: false,
            worker_key_error: None,
            kaggle: KaggleState::new(),
            last_workers_poll: None,
            client,
            handle,
            events_tx,
            events_rx,
            _runtime: runtime,
        };

        if let Some(account) = app.account.clone() {
            let handle = app.handle.clone();
            let client = app.client.clone();
            let tx = app.events_tx.clone();
            handle.spawn(async move {
                api::validate_account(account, client, tx).await;
            });
        }
        app.form.refresh_cursor_styles();
        Ok(app)
    }

    pub fn run(&mut self, terminal: &mut DefaultTerminal) -> Result<()> {
        while !self.should_quit {
            self.tick = self.tick.wrapping_add(1);
            self.poll_workers_if_due();
            self.form.refresh_cursor_styles();
            terminal.draw(|frame| crate::ui::draw(frame, self))?;

            let timeout = if self.busy() {
                Duration::from_millis(80)
            } else {
                Duration::from_millis(250)
            };

            if event::poll(timeout)? {
                loop {
                    let event = event::read()?;
                    self.handle_terminal_event(event);
                    if !event::poll(Duration::ZERO)? {
                        break;
                    }
                }
            }

            while let Ok(event) = self.events_rx.try_recv() {
                self.handle_app_event(event);
            }
        }
        Ok(())
    }

    fn busy(&self) -> bool {
        self.active_job.is_some()
            || matches!(
                self.screen,
                Screen::Auth(AuthScreen::Busy { .. })
                    | Screen::Auth(AuthScreen::SignIn { busy: true, .. })
            )
    }

    fn poll_workers_if_due(&mut self) {
        if self.workers_loading || !matches!(&self.screen, Screen::Main) {
            return;
        }
        let Some(account) = self.account.clone() else {
            return;
        };
        let due = self
            .last_workers_poll
            .is_none_or(|last| last.elapsed() >= WORKER_POLL_INTERVAL);
        if !due {
            return;
        }

        self.last_workers_poll = Some(Instant::now());
        self.workers_loading = true;
        let tx = self.events_tx.clone();
        let client = self.client.clone();
        self.handle.spawn(async move {
            api::fetch_workers(account, client, tx).await;
        });
    }

    fn force_workers_refresh(&mut self) {
        if self.workers_loading {
            return;
        }
        self.last_workers_poll = None;
        self.poll_workers_if_due();
    }

    fn request_worker_key(&mut self) {
        if self.worker_key_loading {
            return;
        }
        let Some(account) = self.account.clone() else {
            return;
        };
        self.worker_key_loading = true;
        self.worker_key_error = None;
        let tx = self.events_tx.clone();
        let client = self.client.clone();
        self.handle.spawn(async move {
            api::ensure_worker_key(account, client, tx).await;
        });
    }

    fn reset_workers(&mut self) {
        self.workers.clear();
        self.workers_loading = false;
        self.workers_loaded = false;
        self.workers_error = None;
        self.worker_key = None;
        self.worker_key_loading = false;
        self.worker_key_error = None;
        self.last_workers_poll = None;
    }

    fn reset_kaggle(&mut self) {
        self.kaggle = KaggleState::new();
    }

    fn open_kaggle(&mut self) {
        if self.account.is_none() {
            return;
        }
        self.overlay = Overlay::Kaggle;
        self.kaggle.token.clear();
        self.kaggle.error = None;
        self.kaggle.confirm_delete = false;
        self.kaggle.refresh_placeholder();
        self.load_kaggle_profiles();
    }

    fn close_kaggle(&mut self) {
        self.kaggle.token.clear();
        self.kaggle.confirm_delete = false;
        if matches!(&self.overlay, Overlay::Kaggle) {
            self.overlay = Overlay::None;
        }
    }

    fn load_kaggle_profiles(&mut self) {
        if self.kaggle.busy.is_some() || self.kaggle.profiles_loading {
            return;
        }
        let Some(account) = self.account.clone() else {
            return;
        };
        self.kaggle.profiles_loading = true;
        self.kaggle.error = None;
        let tx = self.events_tx.clone();
        let client = self.client.clone();
        self.handle.spawn(async move {
            api::fetch_kaggle_profiles(account, client, tx).await;
        });
    }

    fn create_kaggle_profile(&mut self) {
        if self.kaggle.busy.is_some() {
            return;
        }
        let api_key = self.kaggle.token_value();
        if api_key.is_empty() {
            self.kaggle.error = Some("Paste a Kaggle API key first.".to_string());
            return;
        }
        if api_key.len() < 8 {
            self.kaggle.error = Some("That does not look like a Kaggle API key.".to_string());
            return;
        }
        let Some(account) = self.account.clone() else {
            return;
        };
        self.kaggle.busy = Some(KaggleAction::Create);
        self.kaggle.confirm_delete = false;
        self.kaggle.error = None;
        self.kaggle.report = Some("Validating and saving the Kaggle profile…".to_string());
        let tx = self.events_tx.clone();
        let client = self.client.clone();
        self.handle.spawn(async move {
            api::create_kaggle_profile(account, client, api_key, tx).await;
        });
    }

    fn start_kaggle(&mut self) {
        if self.kaggle.busy.is_some() {
            return;
        }
        let Some(profile) = self.kaggle.active_profile().cloned() else {
            self.kaggle.error = Some("Save a Kaggle API key first (press enter).".to_string());
            return;
        };
        if self.workers.iter().any(|worker| {
            worker.worker_id == profile.worker_id
                || worker
                    .worker_id
                    .starts_with(&format!("{}-", profile.worker_id))
        }) {
            self.kaggle.report = Some("This Kaggle worker is already connected.".to_string());
            self.kaggle.error = None;
            return;
        }
        let Some(account) = self.account.clone() else {
            return;
        };
        self.kaggle.busy = Some(KaggleAction::Start);
        self.kaggle.confirm_delete = false;
        self.kaggle.error = None;
        self.kaggle.report = Some(format!(
            "Starting {} on Kaggle… This can take up to two minutes.",
            profile.name
        ));
        let tx = self.events_tx.clone();
        let client = self.client.clone();
        let profile_id = profile.id;
        self.handle.spawn(async move {
            api::run_kaggle_profile_action(account, client, KaggleAction::Start, profile_id, tx)
                .await;
        });
    }

    fn stop_kaggle(&mut self) {
        if self.kaggle.busy.is_some() {
            return;
        }
        let Some(profile) = self.kaggle.active_profile().cloned() else {
            self.kaggle.error = Some("There is no Kaggle profile to stop.".to_string());
            return;
        };
        let Some(account) = self.account.clone() else {
            return;
        };
        self.kaggle.busy = Some(KaggleAction::Stop);
        self.kaggle.confirm_delete = false;
        self.kaggle.error = None;
        self.kaggle.report = Some("Asking the Kaggle worker to stop…".to_string());
        let tx = self.events_tx.clone();
        let client = self.client.clone();
        let profile_id = profile.id;
        self.handle.spawn(async move {
            api::run_kaggle_profile_action(account, client, KaggleAction::Stop, profile_id, tx)
                .await;
        });
    }

    fn delete_kaggle(&mut self) {
        if self.kaggle.busy.is_some() {
            return;
        }
        let Some(profile) = self.kaggle.active_profile().cloned() else {
            self.kaggle.error = Some("There is no Kaggle profile to delete.".to_string());
            return;
        };
        if !self.kaggle.confirm_delete {
            self.kaggle.confirm_delete = true;
            self.kaggle.error = None;
            self.kaggle.report = Some(format!(
                "Press F7 again to delete the profile “{}”.",
                profile.name
            ));
            return;
        }
        let Some(account) = self.account.clone() else {
            return;
        };
        self.kaggle.confirm_delete = false;
        self.kaggle.busy = Some(KaggleAction::Delete);
        self.kaggle.error = None;
        self.kaggle.report = Some("Deleting the Kaggle profile…".to_string());
        let tx = self.events_tx.clone();
        let client = self.client.clone();
        let profile_id = profile.id;
        self.handle.spawn(async move {
            api::run_kaggle_profile_action(account, client, KaggleAction::Delete, profile_id, tx)
                .await;
        });
    }

    fn install_worker_key(&mut self, material: WorkerKeyMaterial) {
        self.worker_key_loading = false;
        let setup = config::save_worker_key(&self.config_path, &material);
        match setup {
            Ok(path) => {
                self.worker_key = Some(WorkerKeySetup {
                    worker_id: material.worker_id,
                    key_id: material.key_id,
                    path: path.clone(),
                });
                self.worker_key_error = None;
                self.set_status(
                    format!("GPU worker key saved to {}", path.display()),
                    StatusKind::Success,
                );
            }
            Err(err) => {
                let message = format!("Could not save the GPU worker key: {err:#}");
                self.worker_key_error = Some(message.clone());
                self.set_status(message, StatusKind::Error);
            }
        }
    }

    pub fn worker_app_url(&self) -> String {
        let server = self
            .account
            .as_ref()
            .map(|account| account.server.as_str())
            .unwrap_or(&self.server);
        worker_websocket_url(server)
    }

    fn handle_terminal_event(&mut self, event: Event) {
        match event {
            Event::Key(key) if matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) => {
                self.handle_key(key)
            }
            Event::Paste(text) => self.handle_paste(&text),
            Event::Resize(_, _) => {}
            _ => {}
        }
    }

    fn handle_paste(&mut self, text: &str) {
        if !matches!(&self.screen, Screen::Main) {
            if let Screen::Auth(AuthScreen::SignIn { token, busy }) = &mut self.screen {
                if !*busy {
                    let clean: String = text.chars().filter(|c| !c.is_whitespace()).collect();
                    token.insert_str(clean);
                }
            }
            return;
        }
        if matches!(&self.overlay, Overlay::Kaggle) {
            if self.kaggle.busy.is_none() {
                let clean: String = text.chars().filter(|c| !c.is_whitespace()).collect();
                self.kaggle.token.insert_str(clean);
            }
            return;
        }
        if !matches!(&self.overlay, Overlay::None) {
            return;
        }
        self.form.insert_paste(text);
    }

    fn handle_key(&mut self, key: KeyEvent) {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        if ctrl && matches!(key.code, KeyCode::Char('q') | KeyCode::Char('c')) {
            self.should_quit = true;
            return;
        }

        if matches!(&self.overlay, Overlay::Token(_)) {
            if matches!(key.code, KeyCode::Enter | KeyCode::Esc) {
                self.overlay = Overlay::None;
            }
            return;
        }
        if matches!(&self.overlay, Overlay::Help) {
            if matches!(key.code, KeyCode::F(1) | KeyCode::Esc | KeyCode::Enter) {
                self.overlay = Overlay::None;
            }
            return;
        }
        if matches!(&self.overlay, Overlay::Workers) {
            match key.code {
                KeyCode::Esc | KeyCode::F(3) | KeyCode::Enter => {
                    self.overlay = Overlay::None;
                }
                KeyCode::Char('r') if !ctrl => self.force_workers_refresh(),
                KeyCode::Char('k') if !ctrl => self.request_worker_key(),
                KeyCode::Char('g') if !ctrl => self.open_kaggle(),
                _ => {}
            }
            return;
        }
        if matches!(&self.overlay, Overlay::Kaggle) {
            if key.code != KeyCode::F(7) {
                self.kaggle.confirm_delete = false;
            }
            match key.code {
                KeyCode::Esc | KeyCode::F(4) => self.close_kaggle(),
                _ if self.kaggle.busy.is_some() => {}
                KeyCode::Enter => self.create_kaggle_profile(),
                KeyCode::F(5) => self.start_kaggle(),
                KeyCode::F(6) => self.stop_kaggle(),
                KeyCode::F(7) => self.delete_kaggle(),
                KeyCode::Tab | KeyCode::BackTab => {}
                _ => {
                    let input: Input = key.into();
                    self.kaggle.token.input(input);
                }
            }
            return;
        }

        if matches!(&self.screen, Screen::Auth(_)) {
            self.handle_auth_key(key);
            return;
        }

        if key.code == KeyCode::F(1) {
            self.overlay = Overlay::Help;
            return;
        }
        if key.code == KeyCode::F(2) {
            self.save_image();
            return;
        }
        if key.code == KeyCode::F(3) {
            self.overlay = Overlay::Workers;
            self.force_workers_refresh();
            return;
        }
        if key.code == KeyCode::F(4) {
            self.open_kaggle();
            return;
        }
        if key.code == KeyCode::F(5)
            || (ctrl && matches!(key.code, KeyCode::Char('g') | KeyCode::Enter))
        {
            self.start_generation();
            return;
        }

        match key.code {
            KeyCode::Tab => {
                self.form.focus_next(false);
                return;
            }
            KeyCode::BackTab => {
                self.form.focus_next(true);
                return;
            }
            _ => {}
        }

        match self.form.focus {
            Field::Scheduler => match key.code {
                KeyCode::Left | KeyCode::Up => self.form.cycle_scheduler(false),
                KeyCode::Right | KeyCode::Down | KeyCode::Char(' ') => {
                    self.form.cycle_scheduler(true)
                }
                KeyCode::Enter => self.form.focus_next(false),
                _ => {}
            },
            Field::Generate => {
                if matches!(key.code, KeyCode::Enter | KeyCode::Char(' ')) {
                    self.start_generation();
                }
            }
            field if field.is_textarea() => {
                let single_line = !matches!(field, Field::Prompt | Field::Negative);
                if single_line {
                    match key.code {
                        KeyCode::Enter => {
                            self.form.focus_next(false);
                            return;
                        }
                        KeyCode::Up if key.modifiers.is_empty() => {
                            self.form.adjust_focused(1);
                            return;
                        }
                        KeyCode::Down if key.modifiers.is_empty() => {
                            self.form.adjust_focused(-1);
                            return;
                        }
                        _ => {}
                    }
                }

                if let Some(textarea) = self.form.textarea_mut(field) {
                    let input: Input = key.into();
                    textarea.input(input);
                }
            }
            _ => {}
        }
    }

    fn handle_auth_key(&mut self, key: KeyEvent) {
        enum Action {
            None,
            Create,
            GoSignIn,
            SubmitSignIn,
            Back,
            Input(KeyEvent),
        }

        let action = match &mut self.screen {
            Screen::Auth(AuthScreen::Menu { selected }) => match key.code {
                KeyCode::Up => {
                    *selected = selected.saturating_sub(1);
                    Action::None
                }
                KeyCode::Down => {
                    *selected = (*selected + 1).min(1);
                    Action::None
                }
                KeyCode::Enter => {
                    if *selected == 0 {
                        Action::Create
                    } else {
                        Action::GoSignIn
                    }
                }
                KeyCode::Char('n') => Action::Create,
                KeyCode::Char('s') => Action::GoSignIn,
                _ => Action::None,
            },
            Screen::Auth(AuthScreen::SignIn { busy, .. }) => {
                if *busy {
                    Action::None
                } else {
                    match key.code {
                        KeyCode::Enter => Action::SubmitSignIn,
                        KeyCode::Esc => Action::Back,
                        _ => Action::Input(key),
                    }
                }
            }
            Screen::Auth(AuthScreen::Busy { .. }) => Action::None,
            Screen::Main => Action::None,
        };

        match action {
            Action::None => {}
            Action::Create => self.start_create_account(),
            Action::GoSignIn => {
                self.auth_error = None;
                self.screen = Screen::Auth(AuthScreen::SignIn {
                    token: Box::new(token_input()),
                    busy: false,
                });
            }
            Action::Back => {
                self.auth_error = None;
                self.screen = Screen::Auth(AuthScreen::Menu { selected: 1 });
            }
            Action::SubmitSignIn => {
                let token = match &self.screen {
                    Screen::Auth(AuthScreen::SignIn { token, .. }) => one_line(token),
                    _ => String::new(),
                };
                if token.trim().is_empty() {
                    self.auth_error = Some("Paste your account token first.".to_string());
                    return;
                }
                self.auth_error = None;
                if let Screen::Auth(AuthScreen::SignIn { busy, .. }) = &mut self.screen {
                    *busy = true;
                }
                let tx = self.events_tx.clone();
                let client = self.client.clone();
                let server = self.server.clone();
                self.handle.spawn(async move {
                    api::sign_in(server, token, client, tx).await;
                });
            }
            Action::Input(key) => {
                if let Screen::Auth(AuthScreen::SignIn { token, busy }) = &mut self.screen {
                    if !*busy {
                        let input: Input = key.into();
                        token.input(input);
                    }
                }
            }
        }
    }

    fn start_create_account(&mut self) {
        self.auth_error = None;
        self.overlay = Overlay::None;
        self.screen = Screen::Auth(AuthScreen::Busy {
            message: "Creating your account and encryption key…".to_string(),
        });
        let tx = self.events_tx.clone();
        let client = self.client.clone();
        let server = self.server.clone();
        self.handle.spawn(async move {
            api::create_account(server, client, tx).await;
        });
    }

    fn start_generation(&mut self) {
        if self.active_job.is_some() {
            self.set_status("A generation is already running.", StatusKind::Info);
            return;
        }
        if self.account.is_none() {
            self.set_status("Sign in before generating.", StatusKind::Error);
            return;
        }

        let params = match self.form.validate() {
            Ok(params) => params,
            Err((field, message)) => {
                self.form.focus = field;
                self.set_status(message, StatusKind::Error);
                return;
            }
        };

        self.active_job = Some(ActiveJob {
            promise_id: None,
            started: Instant::now(),
            steps_completed: 0,
            steps_total: params.steps as u64,
            status: "submitting".to_string(),
            params: params.clone(),
        });
        self.set_status("Submitting prompt…", StatusKind::Info);

        let account = self.account.clone().expect("checked above");
        let tx = self.events_tx.clone();
        let client = self.client.clone();
        self.handle.spawn(async move {
            api::generate(account, client, params, tx).await;
        });
    }

    fn handle_app_event(&mut self, event: AppEvent) {
        match event {
            AppEvent::AccountCreated {
                account,
                token,
                warning,
            } => {
                self.install_account(account, true, warning);
                self.overlay = Overlay::Token(token);
            }
            AppEvent::SignedIn { account } => {
                self.install_account(account, false, None);
            }
            AppEvent::AuthFailed(message) => {
                match &mut self.screen {
                    Screen::Auth(AuthScreen::SignIn { busy, .. }) => *busy = false,
                    Screen::Auth(AuthScreen::Busy { .. }) => {
                        self.screen = Screen::Auth(AuthScreen::Menu { selected: 0 });
                    }
                    _ => {}
                }
                self.auth_error = Some(message);
            }
            AppEvent::Validated { valid, message } => {
                if !valid && matches!(&self.screen, Screen::Main) {
                    self.account = None;
                    self.reset_workers();
                    self.reset_kaggle();
                    self.overlay = Overlay::None;
                    self.screen = Screen::Auth(AuthScreen::Menu { selected: 1 });
                    self.auth_error = message;
                    self.set_status("Sign in to continue.", StatusKind::Error);
                }
            }
            AppEvent::Notice { message, error } => {
                self.set_status(
                    message,
                    if error {
                        StatusKind::Error
                    } else {
                        StatusKind::Info
                    },
                );
            }
            AppEvent::Submitted { promise_id } => {
                if let Some(job) = &mut self.active_job {
                    job.promise_id = Some(promise_id.clone());
                    job.status = "queued".to_string();
                    self.set_status(
                        format!("Queued · {}", short_id(&promise_id)),
                        StatusKind::Info,
                    );
                }
            }
            AppEvent::Progress { promise_id, update } => {
                self.apply_progress(&promise_id, update);
            }
            AppEvent::ImageReady(generated) => {
                self.install_image(generated);
            }
            AppEvent::Failed {
                promise_id,
                message,
            } => {
                let matches_active = match (&promise_id, &self.active_job) {
                    (None, _) => true,
                    (Some(failed), Some(job)) => {
                        job.promise_id.as_deref().is_none()
                            || job.promise_id.as_deref() == Some(failed.as_str())
                    }
                    (Some(_), None) => false,
                };
                if matches_active {
                    self.active_job = None;
                }
                self.set_status(message, StatusKind::Error);
            }
            AppEvent::WorkersUpdated {
                account_id,
                workers,
            } => {
                if self.account.as_ref().map(|account| &account.account_id) != Some(&account_id) {
                    return;
                }
                self.workers = workers;
                self.workers_loading = false;
                self.workers_loaded = true;
                self.workers_error = None;
                if self.workers.is_empty()
                    && self.active_job.is_none()
                    && self.preview.is_none()
                    && matches!(self.status.kind, StatusKind::Success | StatusKind::Muted)
                {
                    self.set_status(
                        "No GPU worker connected · press F3 to set one up.",
                        StatusKind::Info,
                    );
                }
            }
            AppEvent::WorkersFailed {
                account_id,
                message,
            } => {
                if self.account.as_ref().map(|account| &account.account_id) == Some(&account_id) {
                    self.workers_loading = false;
                    self.workers_error = Some(message);
                }
            }
            AppEvent::WorkerKeyReady {
                account_id,
                material,
            } => {
                if self.account.as_ref().map(|account| &account.account_id) == Some(&account_id) {
                    self.install_worker_key(material);
                }
            }
            AppEvent::WorkerKeyFailed {
                account_id,
                message,
            } => {
                if self.account.as_ref().map(|account| &account.account_id) == Some(&account_id) {
                    self.worker_key_loading = false;
                    self.worker_key_error = Some(message.clone());
                    self.set_status(message, StatusKind::Error);
                }
            }
            AppEvent::KaggleProfilesUpdated {
                account_id,
                action,
                profiles,
                preferred_id,
                message,
            } => {
                if self.account.as_ref().map(|account| &account.account_id) != Some(&account_id) {
                    return;
                }
                let status_message = message.clone();
                self.kaggle
                    .apply_profiles(profiles, preferred_id, message, action);
                if let Some(message) = status_message {
                    let kind = match action {
                        KaggleAction::Create | KaggleAction::Delete => StatusKind::Success,
                        _ => StatusKind::Info,
                    };
                    self.set_status(message, kind);
                }
                if matches!(
                    action,
                    KaggleAction::Create
                        | KaggleAction::Start
                        | KaggleAction::Stop
                        | KaggleAction::Delete
                ) {
                    self.force_workers_refresh();
                }
            }
            AppEvent::KaggleFailed {
                account_id,
                action,
                message,
            } => {
                if self.account.as_ref().map(|account| &account.account_id) != Some(&account_id) {
                    return;
                }
                self.kaggle.profiles_loading = false;
                if action == KaggleAction::Refresh {
                    if matches!(&self.overlay, Overlay::Kaggle) {
                        self.kaggle.error = Some(message);
                    }
                } else {
                    self.kaggle.busy = None;
                    self.kaggle.error = Some(message.clone());
                    self.kaggle.report = None;
                    self.set_status(message, StatusKind::Error);
                }
            }
        }
    }

    fn install_account(&mut self, account: Account, created: bool, warning: Option<String>) {
        let save_error = config::save(&self.config_path, &account).err();
        self.account = Some(account);
        self.reset_workers();
        self.reset_kaggle();
        self.screen = Screen::Main;
        self.auth_error = None;
        self.overlay = Overlay::None;

        match (save_error, warning) {
            (Some(err), _) => {
                self.set_status(
                    format!("Account ready, but it could not be saved: {err:#}"),
                    StatusKind::Error,
                );
            }
            (None, Some(warning)) => {
                self.set_status(warning, StatusKind::Info);
            }
            (None, None) if created => {
                self.set_status("Account created.", StatusKind::Success);
            }
            (None, None) => {
                self.set_status("Signed in.", StatusKind::Success);
            }
        }
    }

    fn apply_progress(&mut self, promise_id: &str, update: ProgressUpdate) {
        let no_workers = self.workers_loaded && self.workers.is_empty();
        let Some(job) = &mut self.active_job else {
            return;
        };
        if job
            .promise_id
            .as_deref()
            .is_some_and(|known| known != promise_id)
        {
            return;
        }

        job.status = update.status.clone();
        job.steps_completed = update.steps_completed;
        if let Some(total) = update.steps_total {
            job.steps_total = total;
        }

        let kind = match update.status.as_str() {
            "failed" | "cancelled" => StatusKind::Error,
            "succeeded" => StatusKind::Success,
            _ => StatusKind::Info,
        };
        let message = match update.status.as_str() {
            "queued" if no_workers => {
                "Queued · no GPU worker connected — press F3 to set one up.".to_string()
            }
            "queued" => "Queued · waiting for an available GPU…".to_string(),
            "dispatched" => "Dispatched · starting inference…".to_string(),
            "running" => format!(
                "Generating · {}/{} steps",
                job.steps_completed, job.steps_total
            ),
            "succeeded" => "Decrypting your image…".to_string(),
            "failed" | "cancelled" => update
                .error
                .unwrap_or_else(|| format!("Inference {}.", update.status)),
            other => format!(
                "{} · {}/{} steps",
                capitalise(other),
                job.steps_completed,
                job.steps_total
            ),
        };
        self.set_status(message, kind);
    }

    fn install_image(&mut self, generated: GeneratedImage) {
        if let Some(job) = &self.active_job {
            if let Some(promise_id) = &job.promise_id {
                if promise_id != &generated.promise_id {
                    return;
                }
            }
        }
        let elapsed = self
            .active_job
            .as_ref()
            .map(|job| job.started.elapsed())
            .unwrap_or_default();
        let width = generated.image.width();
        let height = generated.image.height();
        let protocol = self.picker.new_resize_protocol(generated.image);

        self.preview = Some(Preview {
            protocol,
            bytes: generated.jpeg,
            width,
            height,
            seed: generated.seed,
            scheduler: generated.scheduler,
            elapsed,
            promise_id: generated.promise_id.clone(),
        });
        self.active_job = None;
        self.set_status(
            format!(
                "Image ready · {}×{} · seed {} · {:.1}s",
                width,
                height,
                generated.seed,
                elapsed.as_secs_f32()
            ),
            StatusKind::Success,
        );
    }

    fn set_status(&mut self, text: impl Into<String>, kind: StatusKind) {
        let text = text.into().replace(['\r', '\n'], " ");
        self.status = Status::new(text, kind);
    }

    fn save_image(&mut self) {
        let Some(preview) = &self.preview else {
            self.set_status("There is no image to save yet.", StatusKind::Info);
            return;
        };

        let dir = UserDirs::new()
            .and_then(|dirs| dirs.picture_dir().map(Path::to_path_buf))
            .or_else(|| BaseDirs::new().map(|dirs| dirs.home_dir().join("Pictures")))
            .unwrap_or_else(|| PathBuf::from("."))
            .join("imagineit");
        if let Err(err) = std::fs::create_dir_all(&dir) {
            self.set_status(
                format!("Could not create {}: {err}", dir.display()),
                StatusKind::Error,
            );
            return;
        }

        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_secs())
            .unwrap_or(0);
        let path = dir.join(format!(
            "imagineit-{}-{}-{}.jpg",
            preview.seed,
            stamp,
            short_id(&preview.promise_id)
        ));
        match std::fs::write(&path, &preview.bytes) {
            Ok(()) => self.set_status(format!("Saved {}", path.display()), StatusKind::Success),
            Err(err) => self.set_status(
                format!("Could not save {}: {err}", path.display()),
                StatusKind::Error,
            ),
        }
    }

    pub fn server_label(&self) -> String {
        self.server
            .trim_start_matches("https://")
            .trim_start_matches("http://")
            .trim_end_matches('/')
            .to_string()
    }
}

fn normalise_server(server: &str) -> String {
    let server = server.trim().trim_end_matches('/');
    if server.is_empty() {
        return "http://localhost:4600".to_string();
    }
    if server.contains("://") {
        server.to_string()
    } else {
        format!("http://{server}")
    }
}

fn worker_websocket_url(server: &str) -> String {
    let (scheme, rest) = match server.split_once("://") {
        Some((scheme, rest)) => (scheme, rest),
        None => ("http", server),
    };
    let rest = rest.trim_end_matches('/');
    let rest = rest.strip_suffix("/ui-api").unwrap_or(rest);
    let scheme = match scheme {
        "https" => "wss",
        "http" => "ws",
        other => other,
    };
    format!("{scheme}://{rest}/internal/v1/workers/ws")
}

fn short_id(id: &str) -> String {
    id.chars().take(8).collect()
}

fn capitalise(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ready_form() -> Form {
        let mut form = Form::new();
        form.prompt.insert_str("a quiet garden at dusk");
        form
    }

    #[test]
    fn default_form_validates() {
        let mut form = ready_form();
        let params = form.validate().expect("valid defaults");
        assert_eq!(params.prompt, "a quiet garden at dusk");
        assert_eq!(params.negative_prompt, None);
        assert_eq!(params.width, 1024);
        assert_eq!(params.height, 1024);
        assert_eq!(params.steps, 28);
        assert_eq!(params.cfg, 5.0);
        assert_eq!(params.clip_skip, 2);
        assert_eq!(params.scheduler, "euler_a");
        assert!(params.seed < 1_000_000_000);
    }

    #[test]
    fn dimensions_snap_to_a_multiple_of_eight() {
        let mut form = ready_form();
        set_value(&mut form.width, "100");
        let params = form.validate().expect("valid dimensions");
        assert_eq!(params.width, 104);
        assert_eq!(one_line(&form.width), "104");
    }

    #[test]
    fn rejects_an_empty_prompt_and_focuses_it() {
        let mut form = Form::new();
        let error = form.validate().expect_err("empty prompt");
        assert_eq!(error.0, Field::Prompt);
        assert_eq!(form.focus, Field::Prompt);
    }

    #[test]
    fn keeps_an_explicit_seed() {
        let mut form = ready_form();
        set_value(&mut form.seed, "12345");
        assert_eq!(form.validate().unwrap().seed, 12345);
    }

    #[test]
    fn cycles_schedulers_in_both_directions() {
        let mut form = Form::new();
        let start = form.scheduler();
        form.cycle_scheduler(true);
        assert_ne!(form.scheduler(), start);
        form.cycle_scheduler(false);
        assert_eq!(form.scheduler(), start);
    }

    fn kaggle_profile(id: &str, kind: &str, created_at: &str) -> KaggleProfile {
        KaggleProfile {
            id: id.to_string(),
            kind: kind.to_string(),
            name: "kaggle profile".to_string(),
            worker_id: format!("worker-{id}"),
            has_key: true,
            created_at: Some(created_at.to_string()),
        }
    }

    #[test]
    fn kaggle_profiles_filter_and_select_the_newest() {
        let mut state = KaggleState::new();
        state.apply_profiles(
            vec![
                kaggle_profile("old", "kaggle", "2026-01-01T00:00:00"),
                kaggle_profile("local", "byog", "2026-02-01T00:00:00"),
                kaggle_profile("new", "kaggle", "2026-03-01T00:00:00"),
            ],
            None,
            None,
            KaggleAction::Refresh,
        );
        assert_eq!(state.profiles.len(), 2);
        assert_eq!(state.active.as_deref(), Some("new"));

        state.apply_profiles(
            state.profiles.clone(),
            Some("old".to_string()),
            None,
            KaggleAction::Start,
        );
        assert_eq!(state.active.as_deref(), Some("old"));
    }

    #[test]
    fn builds_worker_websocket_urls() {
        assert_eq!(
            worker_websocket_url("https://imagineit.kitsui.co/ui-api"),
            "wss://imagineit.kitsui.co/internal/v1/workers/ws"
        );
        assert_eq!(
            worker_websocket_url("http://localhost:4600"),
            "ws://localhost:4600/internal/v1/workers/ws"
        );
        assert_eq!(
            worker_websocket_url("http://localhost:4600/"),
            "ws://localhost:4600/internal/v1/workers/ws"
        );
    }
}

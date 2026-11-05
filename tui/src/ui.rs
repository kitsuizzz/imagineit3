use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, BorderType, Borders, Clear, LineGauge, Padding, Paragraph, Wrap};
use ratatui::Frame;
use ratatui_image::{FilterType, Resize, StatefulImage};

use crate::api::KaggleAction;
use crate::app::{ActiveJob, App, AuthScreen, Field, Form, Overlay, Screen, StatusKind};
use crate::theme;

const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const MIN_WIDTH: u16 = 80;
const MIN_HEIGHT: u16 = 24;

pub fn draw(frame: &mut Frame, app: &mut App) {
    let area = frame.area();
    if area.width < MIN_WIDTH || area.height < MIN_HEIGHT {
        draw_too_small(frame, area);
        return;
    }

    if matches!(&app.screen, Screen::Main) {
        draw_main(frame, app);
    } else {
        draw_auth(frame, app);
    }
    draw_overlay(frame, app);
}

fn draw_main(frame: &mut Frame, app: &mut App) {
    let [header_area, body_area, footer_area] = Layout::vertical([
        Constraint::Length(2),
        Constraint::Min(0),
        Constraint::Length(2),
    ])
    .areas(frame.area());

    draw_header(frame, header_area, app);

    let form_width = ((body_area.width as u32 * 46 / 100) as u16).clamp(40, 58);
    let [form_area, output_area] =
        Layout::horizontal([Constraint::Length(form_width), Constraint::Min(0)])
            .spacing(2)
            .areas(body_area);

    draw_form(frame, form_area, app);
    draw_output(frame, output_area, app);
    draw_footer(frame, footer_area, app);
}

fn draw_header(frame: &mut Frame, area: Rect, app: &App) {
    let [left, right] =
        Layout::horizontal([Constraint::Min(20), Constraint::Length(32)]).areas(area);
    let title_rect = Rect::new(left.x, left.y, left.width, 1);
    let info_rect = Rect::new(left.x, left.y + 1, left.width, 1);
    let account_rect = Rect::new(right.x, right.y, right.width, 1);
    let security_rect = Rect::new(right.x, right.y + 1, right.width, 1);

    frame.render_widget(Paragraph::new(logo_line(Some("terminal"))), title_rect);
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(app.server_label(), Style::default().fg(theme::MUTED)),
            Span::styled("  ·  ", Style::default().fg(theme::DIM)),
            Span::styled(
                app.protocol_label.clone(),
                Style::default().fg(theme::ACCENT_SECONDARY),
            ),
        ])),
        info_rect,
    );

    let account_line = if let Some(account) = &app.account {
        Line::from(vec![
            Span::styled("● ", Style::default().fg(theme::SUCCESS)),
            Span::styled("account ", Style::default().fg(theme::MUTED)),
            Span::styled(
                short(&account.account_id, 8),
                Style::default().fg(theme::TEXT),
            ),
        ])
    } else {
        Line::from(vec![
            Span::styled("● ", Style::default().fg(theme::ERROR)),
            Span::styled("signed out", Style::default().fg(theme::MUTED)),
        ])
    };
    frame.render_widget(
        Paragraph::new(account_line).alignment(Alignment::Right),
        account_rect,
    );
    let (gpu_dot, gpu_dot_colour, gpu_text, gpu_colour) =
        if app.workers_loading && !app.workers_loaded {
            ("◌ ", theme::MUTED, "checking gpu".to_string(), theme::MUTED)
        } else if !app.workers.is_empty() {
            let busy = app
                .workers
                .iter()
                .filter(|worker| {
                    worker.has_active_job
                        || matches!(worker.status.as_str(), "busy" | "dispatching" | "loading")
                })
                .count();
            let text = if busy > 0 {
                format!("{} gpu · {} busy", app.workers.len(), busy)
            } else {
                format!("{} gpu ready", app.workers.len())
            };
            ("● ", theme::SUCCESS, text, theme::SUCCESS)
        } else if app.workers_loaded {
            (
                "○ ",
                theme::ACCENT_SECONDARY,
                "no gpu · f3 setup".to_string(),
                theme::MUTED,
            )
        } else if app.workers_error.is_some() {
            (
                "? ",
                theme::ACCENT_SECONDARY,
                "gpu status · f3 setup".to_string(),
                theme::MUTED,
            )
        } else {
            (
                "◌ ",
                theme::MUTED,
                "gpu status unknown".to_string(),
                theme::MUTED,
            )
        };
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(gpu_dot, Style::default().fg(gpu_dot_colour)),
            Span::styled(gpu_text, Style::default().fg(gpu_colour)),
        ]))
        .alignment(Alignment::Right),
        security_rect,
    );
}

fn draw_form(frame: &mut Frame, area: Rect, app: &mut App) {
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(theme::panel())
        .title(Line::from(vec![Span::styled(
            " CREATE ",
            Style::default().fg(theme::MUTED),
        )]))
        .padding(Padding::horizontal(1));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let fixed = 1 + 6 + 2;
    let extra = inner.height.saturating_sub(fixed + 5 + 3);
    let prompt_height = 5 + (extra * 2 / 3);
    let negative_height = 3 + extra - (extra * 2 / 3);

    let [prompt_area, negative_area, heading_area, params_area, bottom_area, _] =
        Layout::vertical([
            Constraint::Length(prompt_height),
            Constraint::Length(negative_height),
            Constraint::Length(1),
            Constraint::Length(6),
            Constraint::Length(2),
            Constraint::Min(0),
        ])
        .areas(inner);

    draw_text_field(
        frame,
        prompt_area,
        &app.form,
        Field::Prompt,
        " Prompt ",
        "required",
    );
    draw_text_field(
        frame,
        negative_area,
        &app.form,
        Field::Negative,
        " Negative ",
        "optional",
    );

    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                "PARAMETERS",
                Style::default().fg(theme::DIM).add_modifier(Modifier::BOLD),
            ),
            Span::styled("   ↑↓ adjusts values", Style::default().fg(theme::DIM)),
        ])),
        heading_area,
    );

    let rows: [Rect; 3] = Layout::vertical([
        Constraint::Length(2),
        Constraint::Length(2),
        Constraint::Length(2),
    ])
    .areas(params_area);
    let fields = [
        [Field::Width, Field::Height],
        [Field::Steps, Field::Cfg],
        [Field::Seed, Field::Clip],
    ];
    for (row_area, pair) in rows.iter().zip(fields) {
        let [left, right] = Layout::horizontal([Constraint::Fill(1), Constraint::Fill(1)])
            .spacing(2)
            .areas(*row_area);
        draw_compact_field(frame, left, app, pair[0]);
        draw_compact_field(frame, right, app, pair[1]);
    }

    let [sampler_area, button_area] =
        Layout::horizontal([Constraint::Min(14), Constraint::Length(13)])
            .spacing(2)
            .areas(bottom_area);
    draw_sampler(frame, sampler_area, app);
    draw_generate_button(frame, button_area, app);
}

fn draw_text_field(
    frame: &mut Frame,
    area: Rect,
    form: &Form,
    field: Field,
    title: &str,
    hint: &str,
) {
    let focused = form.focus == field;
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(theme::field_border(focused))
        .padding(Padding::horizontal(1))
        .title(Line::from(vec![
            Span::styled(title.to_string(), theme::field_title(focused)),
            Span::styled(hint.to_string(), Style::default().fg(theme::DIM)),
        ]));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if let Some(textarea) = form.textarea(field) {
        frame.render_widget(textarea, inner);
    }
}

fn draw_compact_field(frame: &mut Frame, area: Rect, app: &App, field: Field) {
    let focused = app.form.focus == field;
    let [label_area, value_area] =
        Layout::horizontal([Constraint::Length(7), Constraint::Min(0)]).areas(area);

    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            field.label(),
            theme::label(focused),
        )))
        .alignment(Alignment::Right),
        label_area,
    );

    let block = Block::default()
        .borders(Borders::BOTTOM)
        .border_style(theme::field_border(focused))
        .padding(Padding::left(1));
    let inner = block.inner(value_area);
    frame.render_widget(block, value_area);
    if let Some(textarea) = app.form.textarea(field) {
        frame.render_widget(textarea, inner);
    }
}

fn draw_sampler(frame: &mut Frame, area: Rect, app: &App) {
    let focused = app.form.focus == Field::Scheduler;
    let [label_area, value_area] =
        Layout::horizontal([Constraint::Length(7), Constraint::Min(0)]).areas(area);

    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            Field::Scheduler.label(),
            theme::label(focused),
        )))
        .alignment(Alignment::Right),
        label_area,
    );

    let block = Block::default()
        .borders(Borders::BOTTOM)
        .border_style(theme::field_border(focused))
        .padding(Padding::left(1));
    let inner = block.inner(value_area);
    frame.render_widget(block, value_area);

    let available = inner.width.saturating_sub(if focused { 4 } else { 0 }) as usize;
    let value = short(scheduler_display(app.form.scheduler()), available);
    let line = if focused {
        Line::from(vec![
            Span::styled("‹ ", Style::default().fg(theme::ACCENT)),
            Span::styled(value, Style::default().fg(theme::ACCENT_STRONG)),
            Span::styled(" ›", Style::default().fg(theme::ACCENT)),
        ])
    } else {
        Line::from(Span::styled(value, Style::default().fg(theme::TEXT)))
    };
    frame.render_widget(Paragraph::new(line), inner);
}

fn draw_generate_button(frame: &mut Frame, area: Rect, app: &App) {
    let focused = app.form.focus == Field::Generate;
    let running = app.active_job.is_some();
    let (background, foreground) = if running {
        (theme::BORDER, theme::MUTED)
    } else if focused {
        (theme::ACCENT, theme::BUTTON_TEXT)
    } else {
        (theme::BUTTON_IDLE, theme::TEXT)
    };

    let block = Block::default().style(Style::default().bg(background));
    frame.render_widget(block, area);

    let (first, second, second_style) = if running {
        (
            format!("{} Working…", spinner(app.tick)),
            "one at a time".to_string(),
            Style::default().fg(theme::MUTED),
        )
    } else {
        (
            "✦ Generate".to_string(),
            "ctrl+g".to_string(),
            Style::default().fg(if focused {
                theme::BUTTON_TEXT
            } else {
                theme::ACCENT_STRONG
            }),
        )
    };

    frame.render_widget(
        Paragraph::new(Text::from(vec![
            Line::from(Span::styled(
                first,
                Style::default().fg(foreground).add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(second, second_style)),
        ]))
        .alignment(Alignment::Center),
        area,
    );
}

fn draw_output(frame: &mut Frame, area: Rect, app: &mut App) {
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(theme::panel())
        .title(Line::from(vec![Span::styled(
            " OUTPUT ",
            Style::default().fg(theme::MUTED),
        )]));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.width == 0 || inner.height == 0 {
        return;
    }

    let generating = app.active_job.is_some();
    let [image_area, gauge_area, meta_area] = Layout::vertical([
        Constraint::Min(1),
        Constraint::Length(if generating { 1 } else { 0 }),
        Constraint::Length(1),
    ])
    .areas(inner);

    if let Some(preview) = app.preview.as_mut() {
        frame.render_stateful_widget(
            StatefulImage::new().resize(Resize::Fit(Some(FilterType::Triangle))),
            image_area,
            &mut preview.protocol,
        );
    } else {
        draw_output_placeholder(frame, image_area, app);
    }

    if generating {
        if let Some(job) = &app.active_job {
            draw_progress(frame, gauge_area, job, app.tick);
        }
    }

    draw_meta_line(frame, meta_area, app);
}

fn draw_output_placeholder(frame: &mut Frame, area: Rect, app: &App) {
    let mut lines: Vec<Line<'static>> = Vec::new();

    if let Some(job) = &app.active_job {
        let phase = match job.status.as_str() {
            "queued" => "Waiting for an available GPU",
            "dispatched" => "Starting inference",
            "running" => "Denoising",
            "succeeded" => "Decrypting image",
            _ => "Preparing",
        };
        lines.push(Line::from(Span::styled(
            "✦",
            Style::default()
                .fg(theme::ACCENT)
                .add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled(
                format!("{}  ", spinner(app.tick)),
                Style::default().fg(theme::ACCENT_STRONG),
            ),
            Span::styled(phase.to_string(), Style::default().fg(theme::TEXT)),
        ]));
        lines.push(Line::from(Span::styled(
            format!(
                "{}×{}  ·  {} steps  ·  {}",
                job.params.width, job.params.height, job.params.steps, job.params.scheduler
            ),
            Style::default().fg(theme::MUTED),
        )));
        if job.status == "queued" && app.workers_loaded && app.workers.is_empty() {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "No GPU worker connected · press F3 to set one up",
                Style::default().fg(theme::ACCENT_SECONDARY),
            )));
        }
    } else if app.status.kind == StatusKind::Error {
        lines.push(Line::from(Span::styled(
            "✖",
            Style::default()
                .fg(theme::ERROR)
                .add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            app.status.text.clone(),
            Style::default().fg(theme::ERROR),
        )));
    } else {
        lines.push(Line::from(Span::styled(
            "✦",
            Style::default()
                .fg(theme::ACCENT)
                .add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Nothing rendered yet",
            Style::default()
                .fg(theme::TEXT)
                .add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(Span::styled(
            "Write a prompt, then press ctrl+g",
            Style::default().fg(theme::MUTED),
        )));
    }

    let top = usize::from(area.height.saturating_sub(lines.len() as u16)) / 2;
    let mut text = Vec::with_capacity(top + lines.len());
    text.extend((0..top).map(|_| Line::from("")));
    text.extend(lines);

    frame.render_widget(
        Paragraph::new(Text::from(text))
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: true }),
        area,
    );
}

fn draw_progress(frame: &mut Frame, area: Rect, job: &ActiveJob, tick: u64) {
    let percent = (job.ratio() * 100.0).round() as u16;
    let spinner = spinner(tick);
    let label = Line::from(vec![
        Span::styled(format!(" {spinner} "), Style::default().fg(theme::ACCENT)),
        Span::styled(
            format!(
                "{percent:>3}%  {}/{} ",
                job.steps_completed, job.steps_total
            ),
            Style::default().fg(theme::MUTED),
        ),
    ]);
    frame.render_widget(
        LineGauge::default()
            .filled_style(Style::default().fg(theme::ACCENT))
            .unfilled_style(Style::default().fg(theme::BORDER))
            .filled_symbol("━")
            .unfilled_symbol("─")
            .ratio(job.ratio())
            .label(label),
        area,
    );
}

fn draw_meta_line(frame: &mut Frame, area: Rect, app: &App) {
    let line = if let Some(job) = &app.active_job {
        Line::from(vec![
            Span::styled(
                format!("{}  ", spinner(app.tick)),
                Style::default().fg(theme::ACCENT),
            ),
            Span::styled(
                match job.status.as_str() {
                    "queued" => "queued · waiting for a GPU",
                    "dispatched" => "dispatched · starting inference",
                    "running" => "running inference",
                    "succeeded" => "succeeded · downloading",
                    other => other,
                }
                .to_string(),
                Style::default().fg(theme::MUTED),
            ),
            Span::styled(
                format!("  ·  seed {}", job.params.seed),
                Style::default().fg(theme::DIM),
            ),
        ])
    } else if let Some(preview) = &app.preview {
        let dimensions = format!("{}×{}", preview.width, preview.height);
        if area.width < 52 {
            Line::from(vec![
                Span::styled(dimensions, Style::default().fg(theme::TEXT)),
                Span::styled(
                    format!("  ·  seed {}", preview.seed),
                    Style::default().fg(theme::MUTED),
                ),
                Span::styled(
                    format!("  ·  {:.1}s", preview.elapsed.as_secs_f32()),
                    Style::default().fg(theme::MUTED),
                ),
            ])
        } else {
            Line::from(vec![
                Span::styled(dimensions, Style::default().fg(theme::TEXT)),
                Span::styled(
                    format!(
                        "  ·  seed {}  ·  {}  ·  {:.1}s",
                        preview.seed,
                        preview.scheduler,
                        preview.elapsed.as_secs_f32()
                    ),
                    Style::default().fg(theme::MUTED),
                ),
                Span::styled("  ·  f2 save", Style::default().fg(theme::DIM)),
            ])
        }
    } else {
        Line::from(Span::styled("ready", Style::default().fg(theme::DIM)))
    };
    frame.render_widget(Paragraph::new(line), area);
}

fn draw_footer(frame: &mut Frame, area: Rect, app: &App) {
    let [status_area, hints_area] =
        Layout::vertical([Constraint::Length(1), Constraint::Length(1)]).areas(area);

    let icon = match app.status.kind {
        StatusKind::Muted => "·",
        StatusKind::Info => "◌",
        StatusKind::Success => "✔",
        StatusKind::Error => "✖",
    };
    let colour = theme::status_colour(app.status.kind);
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(format!(" {icon} "), Style::default().fg(colour)),
            Span::styled(app.status.text.clone(), Style::default().fg(colour)),
        ])),
        status_area,
    );

    let compact = area.width < 105;
    let hints = if compact {
        vec![
            ("tab", "focus"),
            ("^g", "run"),
            ("f2", "save"),
            ("f3", "gpu"),
            ("f4", "kaggle"),
            ("^q", "quit"),
        ]
    } else {
        vec![
            ("tab / shift+tab", "focus"),
            ("↑↓", "adjust numbers"),
            ("ctrl+g", "generate"),
            ("f2", "save image"),
            ("f3", "gpu setup"),
            ("f4", "kaggle"),
            ("f1", "help"),
            ("ctrl+q", "quit"),
        ]
    };
    frame.render_widget(Paragraph::new(hints_line(&hints)), hints_area);
}

fn draw_auth(frame: &mut Frame, app: &App) {
    let area = frame.area();
    let card_width = 64.min(area.width.saturating_sub(4));
    let card_height = 14;

    let total_height = card_height + 1 + 2 + 1;
    let card_x = area.x + area.width.saturating_sub(card_width) / 2;
    let card_y = area.y + area.height.saturating_sub(total_height) / 2;

    let card = Rect::new(card_x, card_y, card_width, card_height);
    let notice = Rect::new(card.x, card.y + card.height + 1, card.width, 2);
    let hint = Rect::new(card.x, notice.y + notice.height, card.width, 1);

    let suffix = match &app.screen {
        Screen::Auth(AuthScreen::SignIn { .. }) => " / sign in",
        Screen::Auth(AuthScreen::Busy { .. }) => " / setup",
        _ => " / welcome",
    };
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme::BORDER))
        .padding(Padding::horizontal(2))
        .title(logo_line(Some(suffix.trim_start_matches(" / "))));
    let inner = block.inner(card);
    frame.render_widget(block, card);

    match &app.screen {
        Screen::Auth(AuthScreen::Menu { selected }) => {
            draw_auth_menu(frame, inner, *selected, app);
        }
        Screen::Auth(AuthScreen::SignIn { token, busy }) => {
            draw_auth_sign_in(frame, inner, token.as_ref(), *busy, app);
        }
        Screen::Auth(AuthScreen::Busy { message }) => {
            draw_auth_busy(frame, inner, message, app);
        }
        Screen::Main => {}
    }

    let (notice_text, notice_style, hint_text) = match &app.screen {
        Screen::Auth(AuthScreen::Menu { .. }) => (
            app.auth_error.clone().unwrap_or_else(|| {
                "Your private key stays on this machine. The server only stores the public half."
                    .to_string()
            }),
            if app.auth_error.is_some() {
                Style::default().fg(theme::ERROR)
            } else {
                Style::default().fg(theme::MUTED)
            },
            "↑↓ choose   ·   enter select   ·   ctrl+q quit",
        ),
        Screen::Auth(AuthScreen::SignIn { .. }) => (
            app.auth_error.clone().unwrap_or_else(|| {
                "The token restores the server-side key escrow. It is the only way back in."
                    .to_string()
            }),
            if app.auth_error.is_some() {
                Style::default().fg(theme::ERROR)
            } else {
                Style::default().fg(theme::MUTED)
            },
            "enter sign in   ·   esc back   ·   ctrl+q quit",
        ),
        Screen::Auth(AuthScreen::Busy { .. }) => (
            app.auth_error
                .clone()
                .unwrap_or_else(|| "Creating a P-256 key pair…".to_string()),
            if app.auth_error.is_some() {
                Style::default().fg(theme::ERROR)
            } else {
                Style::default().fg(theme::MUTED)
            },
            "please wait   ·   ctrl+q quit",
        ),
        Screen::Main => (String::new(), Style::default(), ""),
    };

    frame.render_widget(
        Paragraph::new(notice_text)
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: true })
            .style(notice_style),
        notice,
    );
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            hint_text,
            Style::default().fg(theme::DIM),
        )))
        .alignment(Alignment::Center),
        hint,
    );
}

fn draw_auth_menu(frame: &mut Frame, inner: Rect, selected: usize, app: &App) {
    let [tagline_area, _, first_area, _, second_area, _, server_area, _] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(2),
        Constraint::Length(1),
        Constraint::Length(2),
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Min(0),
    ])
    .areas(inner);

    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "Text to image, from your terminal.",
            Style::default().fg(theme::TEXT),
        )))
        .alignment(Alignment::Center),
        tagline_area,
    );

    draw_auth_option(
        frame,
        first_area,
        selected == 0,
        "Create a new account",
        "Generates a fresh P-256 key and saves it here",
    );
    draw_auth_option(
        frame,
        second_area,
        selected == 1,
        "Sign in with an existing token",
        "Restores the account's escrowed private key",
    );

    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("server  ", Style::default().fg(theme::DIM)),
            Span::styled(app.server.clone(), Style::default().fg(theme::MUTED)),
        ]))
        .alignment(Alignment::Center),
        server_area,
    );
}

fn draw_auth_option(frame: &mut Frame, area: Rect, selected: bool, title: &str, description: &str) {
    let marker = if selected { "●" } else { "○" };
    let marker_style = Style::default().fg(if selected { theme::ACCENT } else { theme::DIM });
    let title_style = Style::default()
        .fg(if selected {
            theme::ACCENT_STRONG
        } else {
            theme::TEXT
        })
        .add_modifier(if selected {
            Modifier::BOLD
        } else {
            Modifier::empty()
        });
    let description_style = Style::default().fg(if selected { theme::MUTED } else { theme::DIM });

    frame.render_widget(
        Paragraph::new(Text::from(vec![
            Line::from(vec![
                Span::styled(format!("{marker}  "), marker_style),
                Span::styled(title.to_string(), title_style),
            ]),
            Line::from(vec![
                Span::raw("   "),
                Span::styled(description.to_string(), description_style),
            ]),
        ])),
        area,
    );
}

fn draw_auth_sign_in(
    frame: &mut Frame,
    inner: Rect,
    token: &ratatui_textarea::TextArea<'_>,
    busy: bool,
    app: &App,
) {
    let [lead_area, _, label_area, token_area, _, server_area, _] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(3),
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Min(0),
    ])
    .areas(inner);

    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "Sign in with the token created when this account was made.",
            Style::default().fg(theme::TEXT),
        )))
        .alignment(Alignment::Center),
        lead_area,
    );

    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                "Account token",
                Style::default().fg(if busy {
                    theme::DIM
                } else {
                    theme::ACCENT_STRONG
                }),
            ),
            Span::styled(
                if busy { "  checking…" } else { "" },
                Style::default().fg(theme::MUTED),
            ),
        ])),
        label_area,
    );

    let token_block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(if busy { theme::BORDER } else { theme::ACCENT }))
        .padding(Padding::horizontal(1));
    let token_inner = token_block.inner(token_area);
    frame.render_widget(token_block, token_area);
    frame.render_widget(token, token_inner);

    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("server  ", Style::default().fg(theme::DIM)),
            Span::styled(app.server.clone(), Style::default().fg(theme::MUTED)),
        ])),
        server_area,
    );
}

fn draw_auth_busy(frame: &mut Frame, inner: Rect, message: &str, app: &App) {
    let [_, spinner_area, message_area, _, server_area, _] = Layout::vertical([
        Constraint::Min(0),
        Constraint::Length(1),
        Constraint::Length(2),
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Min(0),
    ])
    .areas(inner);

    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                format!("{}  ", spinner(app.tick)),
                Style::default().fg(theme::ACCENT),
            ),
            Span::styled(
                "Working…",
                Style::default()
                    .fg(theme::TEXT)
                    .add_modifier(Modifier::BOLD),
            ),
        ]))
        .alignment(Alignment::Center),
        spinner_area,
    );
    frame.render_widget(
        Paragraph::new(message.to_string())
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: true })
            .style(Style::default().fg(theme::MUTED)),
        message_area,
    );
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("server  ", Style::default().fg(theme::DIM)),
            Span::styled(app.server.clone(), Style::default().fg(theme::MUTED)),
        ]))
        .alignment(Alignment::Center),
        server_area,
    );
}

fn draw_overlay(frame: &mut Frame, app: &App) {
    match &app.overlay {
        Overlay::None => {}
        Overlay::Help => {
            frame.render_widget(Clear, frame.area());
            draw_help(frame, app);
        }
        Overlay::Workers => {
            frame.render_widget(Clear, frame.area());
            draw_workers(frame, app);
        }
        Overlay::Kaggle => {
            frame.render_widget(Clear, frame.area());
            draw_kaggle(frame, app);
        }
        Overlay::Token(token) => {
            frame.render_widget(Clear, frame.area());
            draw_token_overlay(frame, app, token);
        }
    }
}

fn draw_workers(frame: &mut Frame, app: &App) {
    let area = frame.area();
    let width = 88.min(area.width.saturating_sub(4));
    let height = 24.min(area.height);
    let popup = centered_rect(area, width, height);
    frame.render_widget(Clear, popup);

    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme::ACCENT))
        .padding(Padding::horizontal(2))
        .title(Line::from(vec![
            Span::styled(" GPU WORKERS ", Style::default().fg(theme::ACCENT)),
            Span::styled("  ·  ", Style::default().fg(theme::DIM)),
            Span::styled("imagineit", Style::default().fg(theme::TEXT)),
        ]))
        .title_bottom(Line::from(Span::styled(
            " r refresh  ·  k worker key  ·  g kaggle  ·  esc close ",
            Style::default().fg(theme::DIM),
        )));
    let inner = block.inner(popup);
    frame.render_widget(block, popup);

    let mut lines: Vec<Line<'static>> = Vec::new();
    section(&mut lines, "CONNECTED");

    if app.workers_loading && !app.workers_loaded {
        lines.push(Line::from(vec![
            Span::styled(
                format!("{}  ", spinner(app.tick)),
                Style::default().fg(theme::ACCENT),
            ),
            Span::styled("Checking workers…", Style::default().fg(theme::MUTED)),
        ]));
    } else if !app.workers.is_empty() {
        for worker in app.workers.iter().take(4) {
            let stale = worker.last_seen_seconds_ago > 30;
            let status = if stale {
                "stale".to_string()
            } else {
                worker.status.clone()
            };
            let status_colour = worker_status_colour(&status);
            let name = if worker.name.trim().is_empty() {
                worker.worker_id.clone()
            } else {
                worker.name.clone()
            };
            lines.push(Line::from(vec![
                Span::styled("● ", Style::default().fg(status_colour)),
                Span::styled(
                    short(&name, 28),
                    Style::default()
                        .fg(theme::TEXT)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(format!("  {status}"), Style::default().fg(status_colour)),
                Span::styled(
                    format!(
                        "  ·  {}{}  ·  up {}s  ·  seen {}s ago",
                        short(
                            if worker.model.trim().is_empty() {
                                &worker.device
                            } else {
                                &worker.model
                            },
                            22
                        ),
                        if worker.kind.is_empty() {
                            String::new()
                        } else {
                            format!(" [{}]", worker.kind)
                        },
                        worker.connected_seconds,
                        worker.last_seen_seconds_ago
                    ),
                    Style::default().fg(theme::MUTED),
                ),
            ]));
        }
        if app.workers.len() > 4 {
            lines.push(Line::from(Span::styled(
                format!("  +{} more", app.workers.len() - 4),
                Style::default().fg(theme::DIM),
            )));
        }
    } else {
        lines.push(Line::from(Span::styled(
            "○  No GPU worker connected for this account.",
            Style::default().fg(theme::ACCENT_SECONDARY),
        )));
    }

    if let Some(error) = &app.workers_error {
        lines.push(Line::from(Span::styled(
            format!("Worker status: {error}"),
            Style::default().fg(theme::ERROR),
        )));
    }

    lines.push(Line::from(""));
    section(&mut lines, "WORKER KEY");

    if let Some(setup) = &app.worker_key {
        lines.push(Line::from(vec![
            Span::styled("● ", Style::default().fg(theme::SUCCESS)),
            Span::styled(
                format!("{} · {}", setup.worker_id, setup.key_id),
                Style::default().fg(theme::TEXT),
            ),
        ]));
        lines.push(Line::from(vec![
            Span::styled("Saved to  ", Style::default().fg(theme::MUTED)),
            Span::styled(
                setup.path.display().to_string(),
                Style::default().fg(theme::ACCENT_STRONG),
            ),
        ]));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Run this on the GPU host (from the ImagineIt checkout):",
            Style::default().fg(theme::TEXT),
        )));
        lines.push(Line::from(Span::styled(
            "python gpu_worker.py \\",
            Style::default().fg(theme::ACCENT_SECONDARY),
        )));
        lines.push(Line::from(Span::styled(
            format!("  --app-url {} \\", app.worker_app_url()),
            Style::default().fg(theme::ACCENT_SECONDARY),
        )));
        lines.push(Line::from(Span::styled(
            format!("  --key-file {} \\", setup.path.display()),
            Style::default().fg(theme::ACCENT_SECONDARY),
        )));
        lines.push(Line::from(Span::styled(
            "  --gpus auto",
            Style::default().fg(theme::ACCENT_SECONDARY),
        )));
        lines.push(Line::from(Span::styled(
            "Copy the key file to the GPU host if it is a different machine.",
            Style::default().fg(theme::MUTED),
        )));
        lines.push(Line::from(Span::styled(
            "GPU host: pip install -e '.[gpu]' from the repo",
            Style::default().fg(theme::MUTED),
        )));
    } else if app.worker_key_loading {
        lines.push(Line::from(vec![
            Span::styled(
                format!("{}  ", spinner(app.tick)),
                Style::default().fg(theme::ACCENT),
            ),
            Span::styled(
                "Fetching a worker key from the server…",
                Style::default().fg(theme::MUTED),
            ),
        ]));
    } else if let Some(error) = &app.worker_key_error {
        lines.push(Line::from(Span::styled(
            format!("✖ {error}"),
            Style::default().fg(theme::ERROR),
        )));
        lines.push(Line::from(Span::styled(
            "Press k to try again.",
            Style::default().fg(theme::MUTED),
        )));
    } else {
        lines.push(Line::from(Span::styled(
            "Press k to fetch or create a GPU worker key for this account.",
            Style::default().fg(theme::TEXT),
        )));
        lines.push(Line::from(Span::styled(
            "The key is saved locally and can be copied to a GPU machine.",
            Style::default().fg(theme::MUTED),
        )));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            if app.workers.is_empty() {
                "Until a worker connects, generation jobs stay queued."
            } else {
                "A worker is connected. Press k to save its key for another GPU host."
            },
            Style::default().fg(theme::DIM),
        )));
    }

    if app.worker_key.is_some() {
        if let Some(error) = &app.worker_key_error {
            lines.push(Line::from(Span::styled(
                format!("Worker key error: {error}"),
                Style::default().fg(theme::ERROR),
            )));
        }
    }

    frame.render_widget(
        Paragraph::new(Text::from(lines))
            .style(Style::default().fg(theme::TEXT))
            .wrap(Wrap { trim: false }),
        inner,
    );
}

fn worker_status_colour(status: &str) -> Color {
    match status {
        "ready" => theme::SUCCESS,
        "busy" | "dispatching" | "loading" => theme::INFO,
        "stale" => theme::ACCENT_SECONDARY,
        _ => theme::ERROR,
    }
}

fn draw_kaggle(frame: &mut Frame, app: &App) {
    let area = frame.area();
    let width = 86.min(area.width.saturating_sub(4));
    let height = 24.min(area.height);
    let popup = centered_rect(area, width, height);
    frame.render_widget(Clear, popup);

    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme::ACCENT))
        .padding(Padding::horizontal(2))
        .title(Line::from(vec![
            Span::styled(" KAGGLE GPU ", Style::default().fg(theme::ACCENT)),
            Span::styled("  ·  ", Style::default().fg(theme::DIM)),
            Span::styled("imagineit", Style::default().fg(theme::TEXT)),
        ]))
        .title_bottom(Line::from(Span::styled(
            " enter save  ·  f5 start  ·  f6 stop  ·  f7 delete  ·  esc close ",
            Style::default().fg(theme::DIM),
        )));
    let inner = block.inner(popup);
    frame.render_widget(block, popup);

    let active_id = app.kaggle.active.as_deref();
    let mut head: Vec<Line<'static>> = Vec::new();
    section(&mut head, "KAGGLE PROFILES");

    if app.kaggle.profiles_loading && app.kaggle.profiles.is_empty() {
        head.push(Line::from(vec![
            Span::styled(
                format!("{}  ", spinner(app.tick)),
                Style::default().fg(theme::ACCENT),
            ),
            Span::styled(
                "Checking saved profiles…",
                Style::default().fg(theme::MUTED),
            ),
        ]));
    } else if app.kaggle.profiles.is_empty() {
        head.push(Line::from(Span::styled(
            "○  No Kaggle GPU profile saved yet.",
            Style::default().fg(theme::ACCENT_SECONDARY),
        )));
    } else {
        for profile in app.kaggle.profiles.iter().take(3) {
            let active = active_id == Some(profile.id.as_str());
            let worker = app.workers.iter().find(|worker| {
                worker.worker_id == profile.worker_id
                    || worker
                        .worker_id
                        .starts_with(&format!("{}-", profile.worker_id))
            });
            let mut spans = vec![
                Span::styled(
                    if active { "● " } else { "○ " },
                    Style::default().fg(if active { theme::ACCENT } else { theme::DIM }),
                ),
                Span::styled(
                    short(&profile.name, 26),
                    Style::default().fg(theme::TEXT).add_modifier(if active {
                        Modifier::BOLD
                    } else {
                        Modifier::empty()
                    }),
                ),
                Span::styled(
                    format!("  {}", short(&profile.worker_id, 22)),
                    Style::default().fg(theme::MUTED),
                ),
            ];
            if let Some(worker) = worker {
                spans.push(Span::styled(
                    format!("  {} connected", worker.status),
                    Style::default().fg(theme::SUCCESS),
                ));
            } else if profile.has_key {
                spans.push(Span::styled(
                    "  key saved",
                    Style::default().fg(theme::MUTED),
                ));
            } else {
                spans.push(Span::styled("  no key", Style::default().fg(theme::ERROR)));
            }
            head.push(Line::from(spans));
        }
        if app.kaggle.profiles.len() > 3 {
            head.push(Line::from(Span::styled(
                format!("  +{} more", app.kaggle.profiles.len() - 3),
                Style::default().fg(theme::DIM),
            )));
        }
    }
    head.push(Line::from(""));

    let head_height = head.len() as u16;
    let [head_area, input_area, hint_area, status_area] = Layout::vertical([
        Constraint::Length(head_height),
        Constraint::Length(3),
        Constraint::Length(2),
        Constraint::Min(0),
    ])
    .areas(inner);

    frame.render_widget(Paragraph::new(Text::from(head)), head_area);

    let input_block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(if app.kaggle.busy.is_some() {
            theme::BORDER
        } else {
            theme::ACCENT
        }))
        .padding(Padding::horizontal(1))
        .title(Line::from(Span::styled(
            " KAGGLE API KEY ",
            Style::default().fg(if app.kaggle.busy.is_some() {
                theme::DIM
            } else {
                theme::ACCENT_STRONG
            }),
        )));
    let input_inner = input_block.inner(input_area);
    frame.render_widget(input_block, input_area);
    frame.render_widget(&app.kaggle.token, input_inner);

    frame.render_widget(
        Paragraph::new(Text::from(vec![
            Line::from(Span::styled(
                "Kaggle → Settings → API → Create New Token",
                Style::default().fg(theme::MUTED),
            )),
            Line::from(Span::styled(
                "Enter saves a new profile; the key is stored encrypted on the server.",
                Style::default().fg(theme::DIM),
            )),
        ])),
        hint_area,
    );

    let mut status_lines: Vec<Line<'static>> = Vec::new();
    if let Some(action) = app.kaggle.busy {
        let message = match action {
            KaggleAction::Create => "Saving the Kaggle profile…",
            KaggleAction::Start => "Starting the ImagineIt notebook on Kaggle…",
            KaggleAction::Stop => "Asking the Kaggle worker to stop…",
            KaggleAction::Delete => "Deleting the Kaggle profile…",
            KaggleAction::Refresh => "Refreshing Kaggle profiles…",
        };
        status_lines.push(Line::from(vec![
            Span::styled(
                format!("{}  ", spinner(app.tick)),
                Style::default().fg(theme::ACCENT),
            ),
            Span::styled(message.to_string(), Style::default().fg(theme::TEXT)),
        ]));
        if action == KaggleAction::Start {
            status_lines.push(Line::from(Span::styled(
                "The notebook can take up to two minutes to appear on Kaggle.",
                Style::default().fg(theme::MUTED),
            )));
        }
    } else if let Some(error) = &app.kaggle.error {
        status_lines.push(Line::from(Span::styled(
            format!("✖ {error}"),
            Style::default().fg(theme::ERROR),
        )));
    } else if let Some(report) = &app.kaggle.report {
        status_lines.push(Line::from(Span::styled(
            if app.kaggle.confirm_delete {
                format!("⚠ {report}")
            } else {
                format!("✔ {report}")
            },
            Style::default().fg(if app.kaggle.confirm_delete {
                theme::ACCENT_SECONDARY
            } else {
                theme::SUCCESS
            }),
        )));
    } else {
        status_lines.push(Line::from(Span::styled(
            "Enter your Kaggle API key and press Enter to create a profile.",
            Style::default().fg(theme::DIM),
        )));
    }

    frame.render_widget(
        Paragraph::new(Text::from(status_lines))
            .style(Style::default().fg(theme::TEXT))
            .wrap(Wrap { trim: true }),
        status_area,
    );
}

fn draw_help(frame: &mut Frame, app: &App) {
    let area = frame.area();
    let width = 78.min(area.width.saturating_sub(4));
    let height = 24.min(area.height);
    let popup = centered_rect(area, width, height);
    frame.render_widget(Clear, popup);

    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme::ACCENT))
        .padding(Padding::uniform(1))
        .title(Line::from(vec![
            Span::styled(" KEYS ", Style::default().fg(theme::ACCENT)),
            Span::styled("  ·  ", Style::default().fg(theme::DIM)),
            Span::styled("imagineit", Style::default().fg(theme::TEXT)),
        ]));
    let inner = block.inner(popup);
    frame.render_widget(block, popup);

    let mut lines: Vec<Line<'static>> = Vec::new();
    section(&mut lines, "INPUT");
    key_line(&mut lines, "tab / shift+tab", "move between fields");
    key_line(&mut lines, "↑ / ↓", "nudge numbers, or move inside prompts");
    key_line(
        &mut lines,
        "enter",
        "newline in prompts, next field elsewhere",
    );
    key_line(&mut lines, "paste", "insert clipboard text");
    lines.push(Line::from(""));
    section(&mut lines, "GENERATE");
    key_line(&mut lines, "ctrl+g / f5", "start a generation");
    key_line(&mut lines, "ctrl+enter", "start a generation");
    key_line(&mut lines, "f2", "save the current image");
    key_line(&mut lines, "f3", "GPU workers and local worker keys");
    key_line(&mut lines, "f4", "Kaggle GPU profiles and API key");
    key_line(&mut lines, "f1 / esc", "close this help");
    key_line(&mut lines, "ctrl+q / ctrl+c", "quit");
    lines.push(Line::from(""));
    section(&mut lines, "THIS SESSION");
    info_line(&mut lines, "server", &app.server);
    info_line(
        &mut lines,
        "account",
        &app.account
            .as_ref()
            .map(|account| account.account_id.clone())
            .unwrap_or_else(|| "signed out".to_string()),
    );
    info_line(
        &mut lines,
        "key file",
        &app.config_path.display().to_string(),
    );
    info_line(
        &mut lines,
        "images",
        &format!(
            "decrypted locally · rendered with the {} protocol",
            app.protocol_label
        ),
    );

    frame.render_widget(
        Paragraph::new(Text::from(lines))
            .style(Style::default().fg(theme::TEXT))
            .wrap(Wrap { trim: false }),
        inner,
    );
}

fn draw_token_overlay(frame: &mut Frame, app: &App, token: &str) {
    let area = frame.area();
    let width = 74.min(area.width.saturating_sub(4));
    let height = 13.min(area.height.saturating_sub(2));
    let popup = centered_rect(area, width, height);
    frame.render_widget(Clear, popup);

    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme::ACCENT))
        .padding(Padding::horizontal(2))
        .title(Line::from(vec![Span::styled(
            " ACCOUNT CREATED ",
            Style::default().fg(theme::ACCENT),
        )]));
    let inner = block.inner(popup);
    frame.render_widget(block, popup);

    let [message_area, token_area, footer_area] = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(3),
        Constraint::Min(0),
    ])
    .areas(inner);

    frame.render_widget(
        Paragraph::new(Text::from(vec![
            Line::from(Span::styled(
                "Save this token. It is the only credential for this account.",
                Style::default()
                    .fg(theme::TEXT)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                format!("It is also stored in {}.", app.config_path.display()),
                Style::default().fg(theme::MUTED),
            )),
        ]))
        .wrap(Wrap { trim: true }),
        message_area,
    );

    let token_block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme::BORDER))
        .padding(Padding::horizontal(1))
        .title(Line::from(Span::styled(
            " token ",
            Style::default().fg(theme::MUTED),
        )));
    let token_inner = token_block.inner(token_area);
    frame.render_widget(token_block, token_area);
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            token.to_string(),
            Style::default()
                .fg(theme::ACCENT_STRONG)
                .add_modifier(Modifier::BOLD),
        ))),
        token_inner,
    );

    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "press enter to continue",
            Style::default().fg(theme::DIM),
        )))
        .alignment(Alignment::Center),
        footer_area,
    );
}

fn draw_too_small(frame: &mut Frame, area: Rect) {
    frame.render_widget(Clear, area);
    let lines = vec![
        Line::from(Span::styled(
            "✦ imagineit",
            Style::default()
                .fg(theme::ACCENT)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(Span::styled(
            format!(
                "This terminal is {}×{}. Resize to at least {}×{}.",
                area.width, area.height, MIN_WIDTH, MIN_HEIGHT
            ),
            Style::default().fg(theme::MUTED),
        )),
    ];
    let height = lines.len() as u16;
    let rect = Rect::new(
        area.x,
        area.y + area.height.saturating_sub(height) / 2,
        area.width,
        height.min(area.height),
    );
    frame.render_widget(
        Paragraph::new(Text::from(lines)).alignment(Alignment::Center),
        rect,
    );
}

fn logo_line(suffix: Option<&str>) -> Line<'static> {
    let mut spans = vec![Span::styled(
        "✦ ",
        Style::default()
            .fg(theme::ACCENT)
            .add_modifier(Modifier::BOLD),
    )];
    spans.extend(gradient("imagineit", (255, 122, 156), (255, 184, 108)));
    if let Some(suffix) = suffix {
        spans.push(Span::styled(
            format!("  {suffix}"),
            Style::default()
                .fg(theme::MUTED)
                .add_modifier(Modifier::ITALIC),
        ));
    }
    Line::from(spans)
}

fn gradient(text: &str, from: (u8, u8, u8), to: (u8, u8, u8)) -> Vec<Span<'static>> {
    let chars: Vec<char> = text.chars().collect();
    let denominator = chars.len().saturating_sub(1).max(1) as f32;
    chars
        .into_iter()
        .enumerate()
        .map(|(index, character)| {
            let ratio = index as f32 / denominator;
            let mix = |start: u8, end: u8| {
                (start as f32 + (end as f32 - start as f32) * ratio).round() as u8
            };
            let colour = Color::Rgb(mix(from.0, to.0), mix(from.1, to.1), mix(from.2, to.2));
            Span::styled(
                character.to_string(),
                Style::default().fg(colour).add_modifier(Modifier::BOLD),
            )
        })
        .collect()
}

fn hints_line(hints: &[(&str, &str)]) -> Line<'static> {
    let mut spans = Vec::new();
    for (index, (key, description)) in hints.iter().enumerate() {
        if index > 0 {
            spans.push(Span::styled("  ·  ", Style::default().fg(theme::DIM)));
        }
        spans.push(Span::styled(
            (*key).to_string(),
            Style::default()
                .fg(theme::ACCENT_SECONDARY)
                .add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::raw(" "));
        spans.push(Span::styled(
            (*description).to_string(),
            Style::default().fg(theme::MUTED),
        ));
    }
    Line::from(spans)
}

fn section(lines: &mut Vec<Line<'static>>, title: &str) {
    lines.push(Line::from(Span::styled(
        title.to_string(),
        Style::default()
            .fg(theme::ACCENT)
            .add_modifier(Modifier::BOLD),
    )));
}

fn key_line(lines: &mut Vec<Line<'static>>, keys: &str, description: &str) {
    lines.push(Line::from(vec![
        Span::styled(
            format!("  {keys:<18}"),
            Style::default().fg(theme::ACCENT_SECONDARY),
        ),
        Span::styled(description.to_string(), Style::default().fg(theme::TEXT)),
    ]));
}

fn info_line(lines: &mut Vec<Line<'static>>, label: &str, value: &str) {
    lines.push(Line::from(vec![
        Span::styled(format!("  {label:<10}"), Style::default().fg(theme::MUTED)),
        Span::styled(value.to_string(), Style::default().fg(theme::TEXT)),
    ]));
}

fn centered_rect(area: Rect, width: u16, height: u16) -> Rect {
    let width = width.min(area.width);
    let height = height.min(area.height);
    Rect::new(
        area.x + area.width.saturating_sub(width) / 2,
        area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    )
}

fn short(text: &str, max: usize) -> String {
    if max == 0 {
        return String::new();
    }
    if text.chars().count() <= max {
        text.to_string()
    } else {
        format!(
            "{}…",
            text.chars().take(max.saturating_sub(1)).collect::<String>()
        )
    }
}

fn scheduler_display(scheduler: &str) -> &str {
    match scheduler {
        "dpm_singlestep" => "dpm_1step",
        "kdpm2_a" => "kdpm2a",
        other => other,
    }
}

fn spinner(tick: u64) -> &'static str {
    SPINNER[(tick as usize) % SPINNER.len()]
}

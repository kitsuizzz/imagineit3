use ratatui::style::{Color, Modifier, Style};

pub const ACCENT: Color = Color::Rgb(255, 122, 156);
pub const ACCENT_STRONG: Color = Color::Rgb(255, 157, 181);
pub const ACCENT_SECONDARY: Color = Color::Rgb(255, 184, 108);
pub const TEXT: Color = Color::Rgb(214, 217, 224);
pub const MUTED: Color = Color::Rgb(120, 124, 140);
pub const DIM: Color = Color::Rgb(88, 92, 108);
pub const BORDER: Color = Color::Rgb(56, 61, 78);
pub const SUCCESS: Color = Color::Rgb(158, 206, 106);
pub const ERROR: Color = Color::Rgb(247, 118, 142);
pub const INFO: Color = Color::Rgb(125, 207, 255);
pub const BUTTON_TEXT: Color = Color::Rgb(24, 24, 32);
pub const BUTTON_IDLE: Color = Color::Rgb(112, 62, 80);

pub fn field_border(focused: bool) -> Style {
    Style::default().fg(if focused { ACCENT } else { BORDER })
}

pub fn field_title(focused: bool) -> Style {
    Style::default()
        .fg(if focused { ACCENT } else { MUTED })
        .add_modifier(Modifier::BOLD)
}

pub fn label(focused: bool) -> Style {
    Style::default().fg(if focused { ACCENT_STRONG } else { MUTED })
}

pub fn panel() -> Style {
    Style::default().fg(BORDER)
}

pub fn status_colour(kind: crate::app::StatusKind) -> Color {
    match kind {
        crate::app::StatusKind::Muted => MUTED,
        crate::app::StatusKind::Info => INFO,
        crate::app::StatusKind::Success => SUCCESS,
        crate::app::StatusKind::Error => ERROR,
    }
}

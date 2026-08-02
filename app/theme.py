"""Navy blue / cyan glassmorphic theme for the Gemini Live Agent GUI.

Qt Widgets has no real backdrop-blur, so the glass look is approximated the
standard way: semi-transparent white fills, a faint light border, generous
border radii, and a soft drop shadow behind each "glass" panel over a deep
navy gradient.
"""

from __future__ import annotations

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QGraphicsDropShadowEffect, QWidget

# -- Palette -----------------------------------------------------------------

NAVY_900 = "#060D1F"  # window background base
NAVY_800 = "#0A1430"
NAVY_700 = "#101E42"
GLASS_FILL = "rgba(148, 197, 255, 0.07)"
GLASS_FILL_STRONG = "rgba(148, 197, 255, 0.12)"
GLASS_BORDER = "rgba(125, 211, 252, 0.22)"
CYAN_400 = "#22D3EE"
CYAN_300 = "#67E8F9"
CYAN_DIM = "rgba(34, 211, 238, 0.35)"
TEXT_PRIMARY = "#E8F3FF"
TEXT_MUTED = "#8FA8CC"
DANGER = "#FB7185"
SUCCESS = "#34D399"

FONT_STACK = "'Segoe UI', 'Inter', 'Helvetica Neue', Arial, sans-serif"


def _panel_qss(fill: str = GLASS_FILL, radius: int = 12) -> str:
    return (
        f"background-color: {fill};"
        f"border: 1px solid {GLASS_BORDER};"
        f"border-radius: {radius}px;"
    )


MAIN_QSS = f"""
* {{
    font-family: {FONT_STACK};
    font-size: 13px;
    color: {TEXT_PRIMARY};
    outline: none;
}}

QMainWindow, QDialog {{
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 1,
        stop: 0 {NAVY_900}, stop: 0.55 {NAVY_800}, stop: 1 {NAVY_700}
    );
}}

QWidget#centralRoot {{
    background: transparent;
}}

QMenuBar {{
    background: transparent;
    border-bottom: 1px solid {GLASS_BORDER};
    padding: 2px;
}}
QMenuBar::item {{
    padding: 4px 10px;
    border-radius: 6px;
    background: transparent;
}}
QMenuBar::item:selected {{ background: {GLASS_FILL_STRONG}; }}
QMenu {{
    background: {NAVY_800};
    border: 1px solid {GLASS_BORDER};
    border-radius: 8px;
    padding: 4px;
}}
QMenu::item {{ padding: 6px 22px; border-radius: 6px; }}
QMenu::item:selected {{ background: {CYAN_DIM}; }}

/* -- Glass panels -------------------------------------------------------- */
QWidget#glassPanel {{
    {_panel_qss()}
}}
QFrame#glassCard {{
    {_panel_qss(GLASS_FILL_STRONG, 10)}
}}

QLabel {{ background: transparent; }}
QLabel#heading {{
    font-size: 15px;
    font-weight: 600;
    color: {CYAN_300};
    letter-spacing: 0.4px;
}}
QLabel#muted {{ color: {TEXT_MUTED}; }}

/* -- Status chip ---------------------------------------------------------- */
QLabel#statusChip {{
    {_panel_qss(GLASS_FILL_STRONG, 10)}
    padding: 4px 12px;
    font-weight: 600;
    color: {CYAN_300};
}}

/* -- Buttons -------------------------------------------------------------- */
QPushButton {{
    {_panel_qss(GLASS_FILL_STRONG, 10)}
    padding: 7px 16px;
    font-weight: 600;
}}
QPushButton:hover {{
    border-color: {CYAN_400};
    background: rgba(34, 211, 238, 0.16);
}}
QPushButton:pressed {{ background: rgba(34, 211, 238, 0.28); }}
QPushButton:disabled {{
    color: {TEXT_MUTED};
    border-color: rgba(148, 197, 255, 0.10);
    background: rgba(148, 197, 255, 0.04);
}}
QPushButton#accentButton {{
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 0,
        stop: 0 #0EA5E9, stop: 1 {CYAN_400}
    );
    border: none;
    color: #04121F;
}}
QPushButton#accentButton:hover {{
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 0,
        stop: 0 #38BDF8, stop: 1 {CYAN_300}
    );
}}

/* -- Inputs --------------------------------------------------------------- */
QLineEdit, QComboBox, QDoubleSpinBox {{
    {_panel_qss("rgba(6, 13, 31, 0.55)", 9)}
    padding: 6px 10px;
    selection-background-color: {CYAN_DIM};
}}
QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus {{
    border: 1px solid {CYAN_400};
}}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox QAbstractItemView {{
    background: {NAVY_800};
    border: 1px solid {GLASS_BORDER};
    border-radius: 8px;
    selection-background-color: {CYAN_DIM};
    padding: 4px;
}}
QCheckBox {{ spacing: 8px; background: transparent; }}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border-radius: 5px;
    border: 1px solid {GLASS_BORDER};
    background: rgba(6, 13, 31, 0.55);
}}
QCheckBox::indicator:checked {{
    background: {CYAN_400};
    border-color: {CYAN_400};
    image: none;
}}

/* -- Transcript / activity ------------------------------------------------- */
QTextEdit, QListWidget {{
    {_panel_qss("rgba(6, 13, 31, 0.55)")}
    padding: 8px;
}}
QTabWidget::pane {{
    {_panel_qss()}
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    color: {TEXT_MUTED};
    padding: 8px 18px;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    margin-right: 4px;
}}
QTabBar::tab:selected {{
    background: {GLASS_FILL_STRONG};
    color: {CYAN_300};
    font-weight: 600;
}}
QTabBar::tab:hover {{ color: {TEXT_PRIMARY}; }}

/* -- Meters ----------------------------------------------------------------- */
QProgressBar {{
    {_panel_qss("rgba(6, 13, 31, 0.55)", 8)}
    text-align: center;
    color: {TEXT_MUTED};
    max-height: 18px;
}}
QProgressBar::chunk {{
    border-radius: 7px;
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 0,
        stop: 0 #0EA5E9, stop: 1 {CYAN_400}
    );
}}

/* -- Scrollbars ------------------------------------------------------------- */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 4px 2px;
}}
QScrollBar::handle:vertical {{
    background: {CYAN_DIM};
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: {CYAN_400}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 2px 4px;
}}
QScrollBar::handle:horizontal {{
    background: {CYAN_DIM};
    border-radius: 5px;
    min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{ background: {CYAN_400}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* -- Scroll areas ------------------------------------------------------------ */
QScrollArea {{
    background: transparent;
    border: none;
}}
QScrollArea > QWidget > QWidget {{
    background: transparent;
}}
QWidget#toolContainer {{
    background: transparent;
}}

/* -- Status bar / tooltips -------------------------------------------------- */
QStatusBar {{
    background: transparent;
    border-top: 1px solid {GLASS_BORDER};
    color: {TEXT_MUTED};
}}
QToolTip {{
    background: {NAVY_800};
    border: 1px solid {GLASS_BORDER};
    border-radius: 6px;
    padding: 6px 8px;
    color: {TEXT_PRIMARY};
}}

/* -- Share banner ------------------------------------------------------------- */
QLabel#shareBanner {{
    background: rgba(251, 113, 133, 0.12);
    border: 1px solid rgba(251, 113, 133, 0.45);
    border-radius: 10px;
    padding: 5px 12px;
    color: {DANGER};
    font-weight: 600;
}}
"""


def apply_glass_shadow(widget: QWidget, *, blur: int = 24, dy: int = 6) -> None:
    """Add a soft drop shadow so a panel reads as floating glass."""
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(blur)
    shadow.setOffset(0, dy)
    shadow.setColor(QColor(0, 0, 0, 140))
    widget.setGraphicsEffect(shadow)

"""Темы ML IDE (тёмная и светлая): единая палитра, скругления, аккуратные скроллбары и подсказки."""
import os

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

THEMES = {
    "dark": dict(BG="#17191e", PANEL="#1f2228", SURFACE="#2a2e37", SURFACE_HI="#343945", BORDER="#363b47", TEXT="#e6e8ee",
                 MUTED="#8b92a1", ACCENT="#3d6df0", ACCENT_HI="#6b93ff", CANVAS="#17191e", CHART="#1c1d21", DISABLED="#7b8294",
                 CARD="#232730", CARD_PRIMARY="#2a3a66"),
    "light": dict(BG="#ffffff", PANEL="#f1f3f7", SURFACE="#e6e9ef", SURFACE_HI="#d9dee7", BORDER="#c7cdd8", TEXT="#1c2230",
                  MUTED="#667085", ACCENT="#2a5ee8", ACCENT_HI="#1f4fd0", CANVAS="#e4e8ee", CHART="#fafbfd", DISABLED="#7a8394",
                  CARD="#ffffff", CARD_PRIMARY="#dce7ff"),
}
_current = "dark"

# Цвета предупреждений и ошибок в тексте (HTML и таблицы стилей по всему приложению заданы под тёмную тему). В светлой теме они
# заменяются на более тёмные, чтобы контраст с белым фоном был не ниже 4.5:1.
LIGHT_FIX = {"#ff6b6b": "#c62828", "#f0a030": "#8a5a00", "#e0a030": "#8a5a00", "#8b92a1": "#596273", "#3fb950": "#1b7f3b",
             "#58a6ff": "#0b5fcc", "#2ecc71": "#177a43", "#ff9f43": "#a65400", "#a29bfe": "#5b4fd6", "#fd79a8": "#c2306a",
             "#4da3ff": "#0b5fcc", "#feca57": "#8a6d00", "#00d2d3": "#007c7d"}


def recolor(text):
    """В светлой теме подменяет «тёмные» цвета в тексте/таблице стилей на читаемые на белом."""
    if _current != "light" or not isinstance(text, str) or "#" not in text:
        return text
    low = text.lower()
    for a, b in LIGHT_FIX.items():
        if a in low:
            text = text.replace(a, b).replace(a.upper(), b)
    return text


def chart_color(hex_color: str) -> str:
    return LIGHT_FIX.get(str(hex_color).lower(), hex_color) if _current == "light" else hex_color


def _make_images() -> dict:
    """Стрелки полей ввода и галочка чекбокса рисуются в PNG под текущую тему (стили Qt не умеют рисовать их сами)."""
    import tempfile
    from pathlib import Path

    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QPainter, QPixmap, QPolygonF
    d = Path(tempfile.gettempdir()) / f"mlide_style_{_current}"
    d.mkdir(parents=True, exist_ok=True)
    out = {}

    def save(name, w, h, draw):
        px = QPixmap(w * 3, h * 3)
        px.fill(Qt.transparent)
        p = QPainter(px)
        p.setRenderHint(QPainter.Antialiasing)
        p.scale(3, 3)
        draw(p)
        p.end()
        f = d / name
        px.save(str(f), "PNG")
        out[name[:-4]] = f.as_posix()

    col = QColor(TEXT)

    def arrow(up):
        def draw(p):
            p.setPen(Qt.NoPen)
            p.setBrush(col)
            pts = [QPointF(1, 6), QPointF(9, 6), QPointF(5, 1.5)] if up else [QPointF(1, 1.5), QPointF(9, 1.5), QPointF(5, 6)]
            p.drawPolygon(QPolygonF(pts))
        return draw

    def check(p):
        pen = p.pen()
        from PySide6.QtGui import QPen
        p.setPen(QPen(QColor("white"), 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawPolyline([QPointF(3.2, 8.2), QPointF(6.4, 11.4), QPointF(12.8, 4.6)])

    save("arrow_up.png", 10, 8, arrow(True))
    save("arrow_down.png", 10, 8, arrow(False))
    save("check.png", 16, 16, check)
    return out
# текущие цвета (заполняются apply_theme; значения по умолчанию - тёмная тема)
globals().update(THEMES["dark"])


def current_theme() -> str:
    return _current


def detect_theme() -> str:
    env = os.environ.get("MLIDE_THEME")
    if env in THEMES:
        return env
    try:
        from .. import settings
        v = settings.get().value("theme", "dark")
        return v if v in THEMES else "dark"
    except Exception:
        return "dark"


def save_theme(name: str):
    from .. import settings
    settings.get().setValue("theme", name)


def apply_dark(app: QApplication):
    apply_theme(app, "dark")


def apply_theme(app: QApplication, name: str = "dark"):
    global _current
    _current = name if name in THEMES else "dark"
    globals().update(THEMES[_current])
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    p = QPalette()
    c = QColor
    p.setColor(QPalette.Window, c(PANEL))
    p.setColor(QPalette.WindowText, c(TEXT))
    p.setColor(QPalette.Base, c(BG))
    p.setColor(QPalette.AlternateBase, c(PANEL))
    p.setColor(QPalette.Text, c(TEXT))
    p.setColor(QPalette.Button, c(SURFACE))
    p.setColor(QPalette.ButtonText, c(TEXT))
    p.setColor(QPalette.ToolTipBase, c(SURFACE))
    p.setColor(QPalette.ToolTipText, c(TEXT))
    p.setColor(QPalette.Highlight, c(ACCENT))
    p.setColor(QPalette.HighlightedText, c("white"))
    p.setColor(QPalette.PlaceholderText, c(MUTED))
    p.setColor(QPalette.Link, c(ACCENT_HI))
    p.setColor(QPalette.Disabled, QPalette.Text, c(DISABLED))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, c(DISABLED))
    p.setColor(QPalette.Disabled, QPalette.WindowText, c(DISABLED))
    app.setPalette(p)
    IMG = _make_images()
    app.setStyleSheet(f"""
        QMainWindow, QDialog {{ background: {PANEL}; }}
        QToolTip {{ background: {SURFACE}; color: {TEXT}; border: 1px solid {ACCENT}; border-radius: 6px;
                    padding: 6px 8px; }}

        /* панели инструментов */
        QToolBar {{ spacing: 4px; padding: 5px 8px; border: none; background: {BG};
                    border-bottom: 1px solid {BORDER}; }}
        QToolBar#toolPalette {{ border-bottom: none; border-right: 1px solid {BORDER}; padding: 8px 5px; spacing: 6px; }}
        QToolBar#toolPalette QToolButton {{ padding: 8px; border-radius: 9px; }}
        QToolBar::separator {{ width: 1px; background: {BORDER}; margin: 4px 6px; }}
        QToolButton {{ padding: 5px 10px; border-radius: 7px; border: 1px solid transparent; }}
        QToolButton:hover {{ background: {SURFACE_HI}; }}
        QToolButton:checked {{ background: {ACCENT}; color: white; }}
        QToolButton:pressed {{ background: {ACCENT}; }}

        /* меню */
        QMenuBar {{ background: {BG}; padding: 2px 4px; }}
        QMenuBar::item {{ padding: 5px 10px; border-radius: 6px; background: transparent; }}
        QMenuBar::item:selected {{ background: {SURFACE_HI}; }}
        QMenu {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px; padding: 5px; }}
        QMenu::item {{ padding: 6px 26px 6px 14px; border-radius: 5px; }}
        QMenu::item:selected {{ background: {ACCENT}; color: white; }}
        QMenu::separator {{ height: 1px; background: {BORDER}; margin: 5px 8px; }}

        /* док-панели */
        QDockWidget {{ font-weight: 600; }}
        QDockWidget::title {{ background: {BG}; padding: 8px 10px; border-bottom: 1px solid {BORDER}; }}
        QMainWindow::separator {{ background: {BG}; width: 3px; height: 3px; }}
        QMainWindow::separator:hover {{ background: {ACCENT}; }}

        /* кнопки */
        QPushButton {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 7px; padding: 6px 14px; }}
        QPushButton:hover {{ background: {SURFACE_HI}; border-color: {MUTED}; }}
        QPushButton:pressed {{ background: {BORDER}; }}
        QPushButton:default {{ background: {ACCENT}; border-color: {ACCENT}; color: white; }}
        QPushButton:default:hover {{ background: {ACCENT_HI}; }}
        QPushButton:disabled {{ color: {DISABLED}; background: {PANEL}; }}
        QPushButton[primary="true"] {{ background: {ACCENT}; border-color: {ACCENT}; color: white; }}
        QPushButton[primary="true"]:hover {{ background: {ACCENT_HI}; }}

        /* поля ввода */
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextBrowser {{
            background: {BG}; border: 1px solid {BORDER}; border-radius: 7px; padding: 5px 8px;
            selection-background-color: {ACCENT}; }}
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
        QComboBox::drop-down {{ border: none; width: 22px; }}
        QComboBox QAbstractItemView {{ background: {PANEL}; border: 1px solid {BORDER}; selection-background-color: {ACCENT}; }}
        QCheckBox {{ spacing: 8px; }}
        QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid {MUTED}; background: {BG}; }}
        QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; image: url({IMG["check"]}); }}
        QSpinBox, QDoubleSpinBox {{ padding-right: 22px; }}
        QSpinBox::up-button, QDoubleSpinBox::up-button {{ subcontrol-origin: border; subcontrol-position: top right; width: 20px;
            border-left: 1px solid {BORDER}; border-bottom: 1px solid {BORDER}; border-top-right-radius: 6px; background: {SURFACE}; }}
        QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-origin: border; subcontrol-position: bottom right; width: 20px;
            border-left: 1px solid {BORDER}; border-bottom-right-radius: 6px; background: {SURFACE}; }}
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover, QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover
            {{ background: {SURFACE_HI}; }}
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url({IMG["arrow_up"]}); width: 9px; height: 7px; }}
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url({IMG["arrow_down"]}); width: 9px; height: 7px; }}

        /* списки */
        QListView {{ background: {BG}; border: none; outline: none; }}
        QListView::item {{ padding: 5px 8px; border-radius: 6px; margin: 1px 4px; }}
        QListView::item:hover {{ background: {SURFACE}; }}
        QListView::item:selected {{ background: {ACCENT}; color: white; }}

        /* группы и вкладки */
        QGroupBox {{ border: 1px solid {BORDER}; border-radius: 9px; margin-top: 12px; padding: 12px 8px 8px 8px; }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; color: {MUTED}; font-weight: 600; }}
        QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 8px; top: -1px; }}
        QTabBar::tab {{ background: transparent; padding: 8px 16px; margin-right: 2px; color: {MUTED};
                        border-bottom: 2px solid transparent; }}
        QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
        QTabBar::tab:hover:!selected {{ color: {TEXT}; }}

        /* прогресс и статус */
        QProgressBar {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 7px; text-align: center; height: 16px; }}
        QProgressBar::chunk {{ background: {ACCENT}; border-radius: 6px; }}
        QStatusBar {{ background: {BG}; color: {MUTED}; border-top: 1px solid {BORDER}; }}
        QStatusBar QLabel {{ color: {MUTED}; }}
        QSplitter::handle {{ background: {BG}; }}
        QSplitter::handle:hover {{ background: {ACCENT}; }}

        /* скроллбары */
        QScrollBar:vertical {{ background: transparent; width: 11px; margin: 0; }}
        QScrollBar::handle:vertical {{ background: {SURFACE_HI}; border-radius: 5px; min-height: 30px; margin: 2px; }}
        QScrollBar::handle:vertical:hover {{ background: {MUTED}; }}
        QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 0; }}
        QScrollBar::handle:horizontal {{ background: {SURFACE_HI}; border-radius: 5px; min-width: 30px; margin: 2px; }}
        QScrollBar::handle:horizontal:hover {{ background: {MUTED}; }}
        QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
        QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
        QScrollArea {{ border: none; }}
    """)

"""Тёмная тема ML IDE: единая палитра, скругления, аккуратные скроллбары и подсказки."""
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

# палитра
BG = "#17191e"          # фон окна / холст
PANEL = "#1f2228"       # панели
SURFACE = "#2a2e37"     # кнопки, поля
SURFACE_HI = "#343945"
BORDER = "#363b47"
TEXT = "#e6e8ee"
MUTED = "#8b92a1"
ACCENT = "#5b8cff"
ACCENT_HI = "#7aa2ff"


def apply_dark(app: QApplication):
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
    p.setColor(QPalette.Disabled, QPalette.Text, c("#5d6370"))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, c("#5d6370"))
    p.setColor(QPalette.Disabled, QPalette.WindowText, c("#5d6370"))
    app.setPalette(p)
    app.setStyleSheet(f"""
        QMainWindow, QDialog {{ background: {PANEL}; }}
        QToolTip {{ background: {SURFACE}; color: {TEXT}; border: 1px solid {ACCENT}; border-radius: 6px;
                    padding: 6px 8px; }}

        /* панели инструментов */
        QToolBar {{ spacing: 4px; padding: 5px 8px; border: none; background: {BG};
                    border-bottom: 1px solid {BORDER}; }}
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
        QPushButton:default {{ background: {ACCENT}; border-color: {ACCENT}; color: white; font-weight: 600; }}
        QPushButton:default:hover {{ background: {ACCENT_HI}; }}
        QPushButton:disabled {{ color: #5d6370; background: {PANEL}; }}
        QPushButton[primary="true"] {{ background: {ACCENT}; border-color: {ACCENT}; color: white; font-weight: 600; }}
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
        QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

        /* списки */
        QListWidget {{ background: {BG}; border: none; outline: none; }}
        QListWidget::item {{ padding: 5px 8px; border-radius: 6px; margin: 1px 4px; }}
        QListWidget::item:hover {{ background: {SURFACE}; }}
        QListWidget::item:selected {{ background: {ACCENT}; color: white; }}

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

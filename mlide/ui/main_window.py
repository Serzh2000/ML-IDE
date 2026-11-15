"""Main window of ML IDE."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtCore import QSize, QStandardPaths, Qt, QTimer
from PySide6.QtGui import (QAction, QActionGroup, QColor, QIcon, QImage, QImageReader, QKeySequence,
                           QPixmap, QShortcut)
from PySide6.QtWidgets import (QApplication, QColorDialog, QComboBox, QDockWidget, QFileDialog, QHBoxLayout,
                               QInputDialog, QLabel, QListView, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QStackedWidget,
                               QPushButton, QToolBar, QVBoxLayout, QWidget)

from .. import __version__, settings, autolabel, classtools, formats, imgio, importers, sam as sam_mod
from ..augment import get_all_presets
from ..model import Annotation, ImageEntry, Project
from .canvas import Canvas
from .dialogs import AutoLabelDialog, ExportDialog, PresetDialog, StatsDialog
from .learn import AdvisorDialog, HelpDialog, TrainDialog
from .model_export import ModelExportDialog
from .model_viewer import ModelViewerDialog
from .results import ResultsDialog
from .tour import TourOverlay, rect_of
from .examples_dialog import ExamplesDialog
from .sam_download import ensure_sam
from .compare_dialog import CompareDialog
from .dedupe_dialog import DedupeDialog
from .help_dialogs import CodeSnippetsDialog, GpuHelpDialog, UpdateDialog
from .live_test import LiveTestDialog
from .review_dialog import ReviewDialog
from .video_dialog import VideoDialog
from .keypoints_dialog import KeypointsDialog
from .welcome import WelcomePage
from . import icons
from .imagelist import FILTERS, ImageListModel
from .. import paths
from .style import apply_dark

SHORTCUTS_HTML = """
<h3>Мышь</h3>
<b>Колесо</b> масштаб &nbsp; <b>Средняя / правая кнопка</b> или <b>Пробел+перетаскивание</b> - сдвиг<br>
<b>Рамка:</b> протяните прямоугольник &nbsp; <b>Полигон:</b> щёлкайте по точкам, замкнуть - <b>двойной щелчок / Enter</b>
или щелчок по первой точке, <b>Backspace</b> - убрать последнюю точку, <b>Esc</b> - отмена<br>
<b>Умный контур:</b> протяните грубую рамку - контур объекта найдётся автоматически (GrabCut)<br>
<b>Выбор:</b> щелчок - выделить, перетаскивание - переместить, маркеры - изменить размер/форму;
<b>двойной щелчок</b> по ребру полигона добавляет вершину, <b>Alt+щелчок</b> по вершине удаляет её;
<b>стрелки</b> сдвигают на 1 px
<h3>Клавиатура</h3>
<b>V</b> выбор &nbsp; <b>W</b> рамка &nbsp; <b>P</b> полигон &nbsp; <b>S</b> умный контур<br>
<b>1-9</b> выбрать класс (и сменить класс выделенного объекта) &nbsp; <b>Del</b> удалить выделенное<br>
<b>A / D</b> (или PgUp / PgDn) предыдущее / следующее изображение &nbsp; <b>R</b> отметить проверенным и дальше<br>
<b>Y</b> принять подсказки авторазметки &nbsp; <b>Ctrl+D</b> копировать разметку с предыдущего<br>
<b>Ctrl+Z / Ctrl+Shift+Z</b> отмена / повтор &nbsp; <b>Ctrl+S</b> сохранить &nbsp; <b>F</b> вписать в окно &nbsp; <b>Ctrl+E</b> экспорт
"""


def color_icon(hexcol, size=14):
    pm = QPixmap(size, size)
    pm.fill(QColor(hexcol))
    return QIcon(pm)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1400, 860)
        self.project = Project()
        self.idx = -1
        self.undo_stack: dict[str, list] = {}
        self.redo_stack: dict[str, list] = {}
        self._cv_cache = (None, None)

        self.canvas = Canvas(lambda: self.project)
        self.welcome = WelcomePage()
        self.stack = QStackedWidget()
        self.stack.addWidget(self.welcome)
        self.stack.addWidget(self.canvas)
        self.setCentralWidget(self.stack)
        self.welcome.importImages.connect(self.import_images)
        self.welcome.importFolder.connect(self.import_folder)
        self.welcome.openProject.connect(self.open_project)
        self.welcome.openExamples.connect(self.open_examples)
        self.welcome.startTour.connect(self.start_tour)
        self.welcome.openRecent.connect(self._open_recent)
        self.setAcceptDrops(True)
        self._build_docks()
        self._build_actions()
        self.canvas.beforeChange.connect(self.push_undo)
        self.canvas.changed.connect(self.on_changed)
        self.canvas.selectionChanged.connect(self.on_canvas_selection)
        self.canvas.smartBox.connect(self.on_smart_box)
        self.canvas.samPoints.connect(self.on_sam_points)
        self.canvas.hint.connect(lambda t: self.statusBar().showMessage(t, 15000))
        self._sam = None
        self.canvas.cursorMoved.connect(lambda x, y: self.pos_label.setText(f"x {x:.0f}  y {y:.0f}"))
        self.canvas.zoomChanged.connect(lambda s: self.zoom_label.setText(f"{s * 100:.0f}%"))

        self.pos_label, self.zoom_label, self.info_label = QLabel(""), QLabel(""), QLabel("")
        for w in (self.info_label, self.pos_label, self.zoom_label):
            self.statusBar().addPermanentWidget(w)
            w.setMinimumWidth(110)

        self.autosave = QTimer(self)
        self.autosave.timeout.connect(self._autosave)
        self.autosave.start(30000)
        self.refresh_all()
        self.set_tool("box")

    # ================================================================ UI construction
    def showEvent(self, e):
        super().showEvent(e)
        if not getattr(self, "_docks_sized", False):
            self._docks_sized = True
            QTimer.singleShot(0, self._fix_docks)

    def _fix_docks(self):
        self.resizeDocks([self.dock_images, self.dock_classes], [270, 300], Qt.Horizontal)

    def _build_docks(self):
        # images
        self.img_list = QListView()
        self.img_model = ImageListModel(lambda: self.project)
        self.img_list.setModel(self.img_model)
        self.img_list.setUniformItemSizes(True)
        self.img_list.setIconSize(QSize(16, 16))
        self.img_list.setSelectionMode(QListView.SingleSelection)
        self.img_filter = QComboBox()
        self.img_filter.addItems(FILTERS)
        self.img_filter.currentIndexChanged.connect(self.refresh_images)
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)
        v.addWidget(self.img_filter)
        v.addWidget(self.img_list, 1)
        for t, fn in (("Импорт изображений…", self.import_images), ("Импорт папки…", self.import_folder)):
            b = QPushButton(t)
            b.clicked.connect(fn)
            v.addWidget(b)
        d = QDockWidget("Изображения", self)
        d.setWidget(w)
        d.setObjectName("images")
        self.addDockWidget(Qt.LeftDockWidgetArea, d)
        self.img_list.selectionModel().currentChanged.connect(self._on_img_item)
        self.dock_images = d

        # classes
        self.cls_list = QListWidget()
        self.cls_list.currentRowChanged.connect(self._on_class_row)
        self.cls_list.itemDoubleClicked.connect(lambda _: self.rename_class())
        self.cls_list.itemClicked.connect(self._class_clicked)
        self.cls_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.cls_list.customContextMenuRequested.connect(self._class_menu)
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)
        v.addWidget(self.cls_list, 1)
        row = QHBoxLayout()
        for t, ico, fn in (("Класс", "plus", self.add_class), ("Имя", "pencil", self.rename_class), ("Цвет", "palette", self.recolor_class),
                           ("", "trash", self.remove_class)):
            b = QPushButton(t)
            b.setIcon(icons.icon(ico))
            if not t:
                b.setObjectName("removeClass")
            b.clicked.connect(fn)
            row.addWidget(b)
        v.addLayout(row)
        b = QPushButton("Импорт списка классов…")
        b.setToolTip("Файл txt / names / yaml со списком классов")
        b.clicked.connect(self.import_classes)
        v.addWidget(b)
        d = QDockWidget("Классы", self)
        d.setWidget(w)
        d.setObjectName("classes")
        self.addDockWidget(Qt.RightDockWidgetArea, d)
        self.dock_classes = d
        tips = {"Класс": "Добавить новый класс (то, что должна искать модель)",
                "Имя": "Переименовать выбранный класс", "Цвет": "Изменить цвет выбранного класса",
                "Импорт изображений…": "Добавить файлы изображений в проект",
                "Импорт папки…": "Добавить все изображения из папки (включая вложенные)"}
        for b in w.findChildren(QPushButton):
            if b.text() in tips:
                b.setToolTip(tips[b.text()])
            elif b.objectName() == "removeClass":
                b.setToolTip("Удалить выбранный класс вместе с его разметкой")

        # annotations
        self.ann_list = QListWidget()
        self.ann_list.setIconSize(QSize(16, 16))
        self.ann_list.currentRowChanged.connect(self._on_ann_row)
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)
        v.addWidget(self.ann_list, 1)
        row = QHBoxLayout()
        for t, fn in (("Принять подсказки (Y)", self.accept_suggestions), ("Удалить", self.canvas.delete_selected)):
            b = QPushButton(t)
            b.clicked.connect(fn)
            row.addWidget(b)
        v.addLayout(row)
        d = QDockWidget("Объекты на изображении", self)
        d.setWidget(w)
        d.setObjectName("objects")
        self.addDockWidget(Qt.RightDockWidgetArea, d)
        self.dock_objects = d

    def _act(self, text, slot, shortcut=None, checkable=False, tip=None):
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.setCheckable(checkable)
        if tip:
            a.setToolTip(tip)
        a.triggered.connect(slot)
        self.addAction(a)
        return a

    def _build_actions(self):
        A = self._act
        self.a_new = A("Новый проект", self.new_project, "Ctrl+N")
        self.a_open = A("Открыть проект…", self.open_project, "Ctrl+O")
        self.a_examples = A("Примеры (скачать и открыть)…", self.open_examples, "Ctrl+Shift+E")
        self.a_save = A("Сохранить", self.save_project, "Ctrl+S")
        self.a_saveas = A("Сохранить как…", self.save_project_as, "Ctrl+Shift+S")
        self.a_imp = A("Импорт изображений…", self.import_images, "Ctrl+I")
        self.a_impf = A("Импорт папки…", self.import_folder, "Ctrl+Shift+I")
        self.a_export = A("Экспорт датасета…", self.export, "Ctrl+E")
        self.a_undo = A("Отменить", self.undo, "Ctrl+Z")
        self.a_redo = A("Повторить", self.redo, "Ctrl+Shift+Z")
        self.a_del = A("Удалить выбранное", lambda: self.canvas.delete_selected(), "Delete")
        self.a_copyprev = A("Копировать разметку с предыдущего изображения", self.copy_prev, "Ctrl+D")
        self.a_accept = A("Принять подсказки на этом изображении", self.accept_suggestions, "Y")
        self.a_review = A("Отметить проверенным и дальше", self.mark_reviewed_next, "R")
        self.a_prev = A("Назад", lambda: self.step(-1), "A")
        self.a_next = A("Вперёд", lambda: self.step(1), "D")
        QShortcut(QKeySequence("PgUp"), self, activated=lambda: self.step(-1))
        QShortcut(QKeySequence("PgDown"), self, activated=lambda: self.step(1))
        self.a_fit = A("Вписать в окно", lambda: self.canvas.fit(), "F")
        self.a_100 = A("Масштаб 100%", lambda: self.canvas.zoom_to(1.0), "Ctrl+1")
        self.a_auto = A("Авторазметка…", self.auto_annotate, "Ctrl+L")
        self.a_presets = A("Пресеты аугментаций…", self.presets, "Ctrl+G")
        self.a_stats = A("Статистика и проверка датасета…", self.stats, "Ctrl+T")
        self.a_labels = A("Показывать подписи", self._toggle_labels, checkable=True)
        self.a_labels.setChecked(True)
        self.a_help = A("Горячие клавиши", self.show_help, "Shift+F1")
        self.a_ref = A("Справочник…", self.show_reference, "F1")
        self.a_adv = A("Советник по настройкам…", self.advisor, "Ctrl+H")
        self.a_train = A("Обучение модели…", self.train, "Ctrl+Shift+T")
        self.a_results = A("Результаты обучения…", self.results, "Ctrl+Shift+R")
        self.a_modelexp = A("Экспорт модели (ONNX, ESP32…)…", self.export_model, "Ctrl+Shift+M")
        self.a_viewer = A("Просмотр модели (схема ONNX)…", self.view_model, "Ctrl+Shift+G")
        self.a_imp_coco = A("Импорт разметки COCO (JSON)…", self.import_coco)
        self.a_imp_voc = A("Импорт разметки Pascal VOC…", self.import_voc)
        self.a_video = A("Кадры из видео…", self.extract_video, "Ctrl+Shift+V")
        self.a_snap = A("Сохранить снимок разметки", self.save_snapshot)
        self.a_opensnap = A("Открыть снимок разметки…", self.open_snapshot)
        self.a_dups = A("Дубли и похожие кадры…", self.find_duplicates)
        self.a_review_model = A("Проверка разметки моделью…", self.review_with_model)
        self.a_compare = A("Сравнение запусков обучения…", self.compare_runs)
        self.a_live = A("Тест модели на видео и камере…", self.live_test)
        self.a_gpu = A("Видеокарта и ускорение (CUDA)…", self.gpu_help)
        self.a_code = A("Примеры кода для запуска модели…", self.code_examples)
        self.a_update = A("Проверить обновления…", self.check_updates)
        self.a_kpts = A("Настроить ключевые точки…", self.configure_keypoints)
        self.a_tour = A("Тур по приложению", self.start_tour)
        self.a_exit = A("Выход", self.close, "Ctrl+Q")

        # короткие подписи для кнопок панели (в меню остаются полные названия)
        for a, t in ((self.a_prev, "Назад"), (self.a_next, "Вперёд"), (self.a_review, "Проверено"),
                     (self.a_auto, "Авторазметка"), (self.a_accept, "Принять (Y)"), (self.a_presets, "Аугментации"),
                     (self.a_stats, "Статистика"), (self.a_adv, "Советник"), (self.a_export, "Экспорт"),
                     (self.a_train, "Обучение"), (self.a_results, "Результаты"), (self.a_ref, "Справка (F1)")):
            a.setIconText(t)

        for a, t in ((self.a_prev, "Предыдущее изображение (A)"), (self.a_next, "Следующее изображение (D)"),
                     (self.a_review, "Отметить изображение проверенным и перейти к следующему (R)"),
                     (self.a_auto, "Авторазметка: обученная модель предложит рамки на ваших картинках (Ctrl+L)"),
                     (self.a_accept, "Принять подсказки модели на этом изображении (Y)"),
                     (self.a_presets, "Пресеты аугментаций: искажения для обучения, с превью (Ctrl+G)"),
                     (self.a_stats, "Статистика датасета и поиск проблем в разметке (Ctrl+T)"),
                     (self.a_adv, "Советник: подберёт настройки обучения под ваш датасет (Ctrl+H)"),
                     (self.a_export, "Экспорт датасета и train.py (Ctrl+E)"),
                     (self.a_train, "Запустить обучение и следить за ним (Ctrl+Shift+T)"),
                     (self.a_results, "Результаты обучения: метрики, графики, проверка модели (Ctrl+Shift+R)"),
                     (self.a_ref, "Справочник: руководства и словарь терминов (F1)")):
            a.setToolTip(t)

        for a, ico in ((self.a_prev, "prev"), (self.a_next, "next"), (self.a_review, "reviewed"), (self.a_auto, "auto"),
                       (self.a_accept, "accept"), (self.a_presets, "augment"), (self.a_stats, "stats"), (self.a_adv, "advisor"),
                       (self.a_export, "export"), (self.a_train, "train"), (self.a_results, "results"), (self.a_ref, "help"),
                       (self.a_open, "open"), (self.a_examples, "examples"), (self.a_imp, "images"), (self.a_impf, "folder"),
                       (self.a_video, "video"), (self.a_live, "camera"), (self.a_viewer, "model"), (self.a_compare, "stats"),
                       (self.a_tour, "tour"), (self.a_update, "refresh"), (self.a_gpu, "info"), (self.a_dups, "copy")):
            a.setIcon(icons.icon(ico))
        grp = QActionGroup(self)
        self.tool_actions = {}
        for key, text, sc, tip in (("select", "Выбор", "V", "Выбор / перемещение / изменение размера (V)"),
                                   ("box", "Рамка", "W", "Рисовать рамки (W)"),
                                   ("poly", "Полигон", "P", "Рисовать полигоны для сегментации (P)"),
                                   ("smart", "Умный контур", "S", "Обведите объект грубой рамкой - контур найдётся сам (S)"),
                                   ("sam", "Клик (SAM)", "G", "Кликните по объекту - нейросеть найдёт точный контур. Ctrl+клик - исключить область, Enter - принять (G)"),
                                   ("obb", "Повёрнутая рамка", "O", "Протяните линию вдоль объекта, затем щёлкните, задав ширину. [ и ] - повернуть (O)"),
                                   ("pose", "Ключевые точки", "K", "Нарисуйте рамку, затем ставьте точки по очереди (K). Список точек: Правка → Настроить ключевые точки"),
                                   ("tag", "Класс кадра", "T", "Классификация: присвоить класс изображению целиком клавишами 1-9 или щелчком по классу (T)")):
            a = A(text, lambda _=False, k=key: self.set_tool(k), sc, checkable=True, tip=tip)
            a.setIcon(icons.icon(key))
            a.setToolTip(f"<b>{text}</b> ({sc})<br>{tip}")
            grp.addAction(a)
            self.tool_actions[key] = a

        for i in range(9):
            QShortcut(QKeySequence(str(i + 1)), self, activated=lambda i=i: self.pick_class(i))

        mb = self.menuBar()
        m = mb.addMenu("&Файл")
        for a in (self.a_new, self.a_open, self.a_examples):
            m.addAction(a)
        self.recent_menu = m.addMenu("Недавние проекты")
        for a in (self.a_save, self.a_saveas, self.a_snap, self.a_opensnap, None, self.a_imp, self.a_impf, self.a_video, None,
                  self.a_imp_coco, self.a_imp_voc, None, self.a_export, None, self.a_exit):
            m.addSeparator() if a is None else m.addAction(a)
        self._fill_recent()
        m = mb.addMenu("&Правка")
        for a in (self.a_undo, self.a_redo, None, self.a_del, self.a_copyprev, self.a_accept, self.a_review, None, self.a_kpts):
            m.addSeparator() if a is None else m.addAction(a)
        m = mb.addMenu("&Инструменты")
        for a in (self.a_auto, self.a_review_model, self.a_dups, self.a_presets, self.a_stats, self.a_adv, None,
                  self.a_train, self.a_results, self.a_compare, self.a_modelexp, self.a_viewer, self.a_live, self.a_code, self.a_gpu):
            m.addSeparator() if a is None else m.addAction(a)
        m = mb.addMenu("&Вид")
        for a in (self.a_fit, self.a_100, self.a_labels):
            m.addAction(a)
        m.addSeparator()
        for d in (self.dock_images, self.dock_classes, self.dock_objects):
            m.addAction(d.toggleViewAction())
        tm = m.addMenu("Тема оформления")
        for code, label in (("dark", "Тёмная"), ("light", "Светлая")):
            tm.addAction(label, lambda c=code: self.change_theme(c))
        lm = m.addMenu("Язык / Language")
        for code, label in (("ru", "Русский"), ("en", "English")):
            lm.addAction(label, lambda c=code: self.change_language(c))
        hm = mb.addMenu("&Справка")
        hm.addAction(self.a_tour)
        hm.addAction(self.a_update)
        hm.addAction(self.a_ref)
        hm.addAction(self.a_help)

        # слева - палитра инструментов разметки (только значки, название и клавиша в подсказке), сверху - одна строка действий
        pal = QToolBar("Инструменты разметки")
        pal.setObjectName("toolPalette")
        pal.setOrientation(Qt.Vertical)
        pal.setMovable(False)
        pal.setIconSize(QSize(22, 22))
        pal.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.addToolBar(Qt.LeftToolBarArea, pal)
        for k in self.tool_actions:
            pal.addAction(self.tool_actions[k])
        self.tools_bar = pal

        tb = QToolBar("Действия")
        self.toolbar = tb
        tb.setMovable(False)
        tb.setIconSize(QSize(18, 18))
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(Qt.TopToolBarArea, tb)
        for a in (self.a_prev, self.a_next, self.a_review):
            tb.addAction(a)
        tb.addSeparator()
        for a in (self.a_auto, self.a_accept, self.a_presets, self.a_stats, self.a_adv):
            tb.addAction(a)
        tb.addSeparator()
        for a in (self.a_export, self.a_train, self.a_results):
            tb.addAction(a)
        tb.addSeparator()
        tb.addAction(self.a_ref)
        for a in (self.a_prev, self.a_next, self.a_review, self.a_ref):  # навигация и справка - только значки: экономит место
            btn = tb.widgetForAction(a)
            if btn is not None:
                btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.toolbar2 = tb                                           # совместимость: раньше было две строки

    # ================================================================ state refresh
    @property
    def entry(self) -> ImageEntry | None:
        return self.project.images[self.idx] if 0 <= self.idx < len(self.project.images) else None

    def refresh_all(self):
        self.stack.setCurrentIndex(1 if self.project.images else 0)
        self.refresh_classes()
        self.refresh_images()
        self.refresh_anns()
        self._title()

    def _title(self):
        name = self.project.name + ("*" if self.project.dirty else "")
        self.setWindowTitle(f"{name} - ML IDE {__version__}")

    def refresh_images(self):
        sm = self.img_list.selectionModel()
        sm.blockSignals(True)
        self.img_model.rebuild(self.img_filter.currentIndex(), self.idx)
        row = self.img_model.row_of(self.idx)
        if row >= 0:
            self.img_list.setCurrentIndex(self.img_model.index(row))
        sm.blockSignals(False)
        self.dock_images.setWindowTitle(f"Изображения ({len(self.project.images)})")

    def _update_img_item(self):
        self.img_model.touch(self.idx)

    def refresh_classes(self):
        self.cls_list.blockSignals(True)
        row = self.cls_list.currentRow()
        self.cls_list.clear()
        from collections import Counter
        cnt = Counter(a.cls for e in self.project.images for a in e.anns if not a.suggested)
        cnt.update(e.tag for e in self.project.images if e.tag is not None)          # классификация: класс кадра
        for i, c in enumerate(self.project.classes):
            counts = cnt.get(i, 0)
            it = QListWidgetItem(color_icon(c["color"]), f"{i + 1 if i < 9 else '·'}  {c['name']}   ({counts})")
            self.cls_list.addItem(it)
        self.cls_list.setCurrentRow(min(max(row, 0), self.cls_list.count() - 1) if self.cls_list.count() else -1)
        self.cls_list.blockSignals(False)
        self.canvas.cur_class = max(0, self.cls_list.currentRow())
        self.canvas.update()

    def refresh_anns(self):
        self.ann_list.blockSignals(True)
        self.ann_list.clear()
        e = self.entry
        sel_row = -1
        if e:
            for r, a in enumerate(e.anns):
                nm = self.project.classes[a.cls]["name"] if a.cls < len(self.project.classes) else f"#{a.cls}"
                col = self.project.classes[a.cls]["color"] if a.cls < len(self.project.classes) else "#888"
                kind_icon = {"box": "box", "poly": "polygon", "obb": "obb", "pose": "pose"}.get(a.kind, "box")
                it = QListWidgetItem(icons.tinted(kind_icon, col, 16), f"{nm}"
                                     f"{'   ? ' + format(a.conf, '.2f') if a.suggested and a.conf is not None else ''}")
                self.ann_list.addItem(it)
                if a.id == self.canvas.selected:
                    sel_row = r
        self.ann_list.setCurrentRow(sel_row)
        self.ann_list.blockSignals(False)

    # ================================================================ image navigation
    def _on_img_item(self, cur, _prev):
        if cur is not None and cur.isValid():
            self.load_image(self.img_model.image_index(cur.row()))

    def load_image(self, i):
        if not (0 <= i < len(self.project.images)):
            self.idx = -1
            self.canvas.set_entry(None, None)
            self.refresh_anns()
            return
        if i != self.idx and self.project.path and self.project.dirty:
            self._autosave()
        self.idx = i
        e = self.entry
        rd = QImageReader(e.path)
        rd.setAutoTransform(True)
        img = rd.read()
        if img.isNull():
            self.statusBar().showMessage(f"Не удалось прочитать {e.path}", 5000)
            self.canvas.set_entry(None, None)
            return
        self.canvas.set_entry(e, img)
        self.info_label.setText(f"{e.w}×{e.h}  [{i + 1}/{len(self.project.images)}]")
        self.refresh_anns()
        self._update_img_item()
        row = self.img_model.row_of(i)
        if row >= 0 and self.img_list.currentIndex().row() != row:
            sm = self.img_list.selectionModel()
            sm.blockSignals(True)
            self.img_list.setCurrentIndex(self.img_model.index(row))
            sm.blockSignals(False)
        self.canvas.setFocus()

    def step(self, d):
        n = len(self.project.images)
        if not n:
            return
        self.load_image(min(max(self.idx + d, 0), n - 1))

    def mark_reviewed_next(self):
        e = self.entry
        if e:
            e.reviewed = True
            self.accept_suggestions(silent=True)
            self.project.dirty = True
            self._update_img_item()
            self.step(1)
            self._title()

    # ================================================================ tools / classes
    def set_tool(self, k):
        self.canvas.tool = k
        self.tool_actions[k].setChecked(True)
        self.canvas.cancel_poly()
        self.canvas.setCursor(Qt.ArrowCursor if k == "select" else Qt.CrossCursor)

    def _toggle_labels(self):
        self.canvas.show_labels = self.a_labels.isChecked()
        self.canvas.update()

    def _on_class_row(self, row):
        self.canvas.cur_class = max(0, row)

    def pick_class(self, i):
        if i >= len(self.project.classes):
            return
        if self.canvas.tool == "tag":
            self.set_image_tag(i)
            return
        self.cls_list.setCurrentRow(i)
        a = self.canvas.selected_ann()
        if a and a.cls != i:
            self.push_undo()
            a.cls = i
            a.suggested = False
            self.on_changed()

    def set_image_tag(self, cls):
        """Классификация: класс изображения целиком; после назначения - следующий кадр."""
        e = self.entry
        if not e:
            return
        self.push_undo()
        e.tag = cls
        self.project.dirty = True
        self.on_changed()
        if self.idx < len(self.project.images) - 1:
            self.step(1)

    def _class_clicked(self, item):
        if self.canvas.tool == "tag":
            self.set_image_tag(self.cls_list.row(item))

    def configure_keypoints(self):
        if KeypointsDialog(self.project, self).exec():
            self.refresh_anns()
            self.canvas.update()
            self._title()

    def add_class(self):
        name, ok = QInputDialog.getText(self, "Новый класс", "Название класса:")
        if ok and name.strip():
            self.project.add_class(name.strip())
            self.refresh_classes()
            self.cls_list.setCurrentRow(len(self.project.classes) - 1)
            self._title()

    def rename_class(self):
        r = self.cls_list.currentRow()
        if r < 0:
            return
        name, ok = QInputDialog.getText(self, "Переименовать класс", "Название:", text=self.project.classes[r]["name"])
        if ok and name.strip():
            self.project.classes[r]["name"] = name.strip()
            self.project.dirty = True
            self.refresh_classes()
            self.refresh_anns()
            self._title()

    def recolor_class(self):
        r = self.cls_list.currentRow()
        if r < 0:
            return
        c = QColorDialog.getColor(QColor(self.project.classes[r]["color"]), self, "Цвет класса")
        if c.isValid():
            self.project.classes[r]["color"] = c.name()
            self.project.dirty = True
            self.refresh_classes()
            self.refresh_anns()

    def remove_class(self):
        r = self.cls_list.currentRow()
        if r < 0:
            return
        n = sum(1 for e in self.project.images for a in e.anns if a.cls == r)
        msg = f"Удалить класс '{self.project.classes[r]['name']}'"
        msg += f" и его размеченные объекты ({n})?" if n else "?"
        if QMessageBox.question(self, "Удаление класса", msg) == QMessageBox.Yes:
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.project.remove_class(r)
            self.canvas.select(None)
            self.refresh_all()

    def import_classes(self):
        p, _ = QFileDialog.getOpenFileName(self, "Список классов", "", "Файлы классов (*.txt *.names *.yaml *.yml)")
        if p:
            try:
                for n in formats.read_class_names(p):
                    self.project.add_class(str(n))
            except Exception as ex:  # noqa
                QMessageBox.warning(self, "Импорт", str(ex))
            self.refresh_classes()
            self._title()

    # ================================================================ annotation list <-> canvas
    def _on_ann_row(self, row):
        e = self.entry
        if e and 0 <= row < len(e.anns):
            self.canvas.select(e.anns[row].id)

    def on_canvas_selection(self, ann_id):
        e = self.entry
        self.ann_list.blockSignals(True)
        row = next((r for r, a in enumerate(e.anns) if a.id == ann_id), -1) if e else -1
        self.ann_list.setCurrentRow(row)
        self.ann_list.blockSignals(False)
        a = self.canvas.selected_ann()
        if a and a.cls < self.cls_list.count():
            self.cls_list.blockSignals(True)
            self.cls_list.setCurrentRow(a.cls)
            self.cls_list.blockSignals(False)
            self.canvas.cur_class = a.cls

    def on_changed(self):
        self.project.dirty = True
        self.refresh_anns()
        self._update_img_item()
        self.canvas.update()
        self._title()
        self.refresh_class_counts_soon()

    def refresh_class_counts_soon(self):
        if not hasattr(self, "_cc_timer"):
            self._cc_timer = QTimer(self)
            self._cc_timer.setSingleShot(True)
            self._cc_timer.timeout.connect(self.refresh_classes)
        self._cc_timer.start(400)

    # ================================================================ undo / redo
    def push_undo(self):
        e = self.entry
        if not e:
            return
        self.undo_stack.setdefault(e.path, []).append(e.snapshot())
        del self.undo_stack[e.path][:-100]
        self.redo_stack[e.path] = []

    def undo(self):
        e = self.entry
        st = self.undo_stack.get(e.path) if e else None
        if st:
            self.redo_stack.setdefault(e.path, []).append(e.snapshot())
            e.restore(st.pop())
            self.canvas.select(None)
            self.on_changed()

    def redo(self):
        e = self.entry
        st = self.redo_stack.get(e.path) if e else None
        if st:
            self.undo_stack.setdefault(e.path, []).append(e.snapshot())
            e.restore(st.pop())
            self.canvas.select(None)
            self.on_changed()

    # ================================================================ labeling helpers
    def accept_suggestions(self, silent=False):
        e = self.entry
        if not e:
            return
        sel = self.canvas.selected_ann()
        targets = [sel] if sel and sel.suggested else [a for a in e.anns if a.suggested]
        if not targets:
            return
        self.push_undo()
        for a in targets:
            a.suggested = False
        self.on_changed()

    def copy_prev(self):
        e = self.entry
        if not e or self.idx == 0:
            return
        prev = self.project.images[self.idx - 1]
        src = [a for a in prev.anns if not a.suggested]
        if not src:
            self.statusBar().showMessage("На предыдущем изображении нет разметки", 3000)
            return
        self.push_undo()
        for a in src:
            pts = [[x / prev.w * e.w, y / prev.h * e.h] for x, y in a.pts]
            e.anns.append(Annotation(a.cls, a.kind, pts))
        self.on_changed()

    def on_sam_points(self, pts):
        e = self.entry
        if not e or not pts:
            return
        if self._sam is None:
            if not ensure_sam(self):
                self.canvas.cancel_sam()
                return
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                self._sam = sam_mod.SamModel()
            finally:
                QApplication.restoreOverrideCursor()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if self._cv_cache[0] != e.path:
                self._cv_cache = (e.path, imgio.imread(e.path))
            self._sam.set_image(self._cv_cache[1], key=e.path)
            mask, iou = self._sam.predict(pts)
        except Exception as ex:  # noqa
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Разметка по клику", f"{type(ex).__name__}: {ex}")
            self.canvas.cancel_sam()
            return
        finally:
            QApplication.restoreOverrideCursor()
        poly = sam_mod.mask_to_polygon(mask)
        self.canvas.set_sam_preview(poly)
        self.statusBar().showMessage(
            ("Контур найден" if poly else "Контур не найден - кликните по объекту") +
            ". Enter - принять полигон, Shift+Enter - принять рамку, Ctrl+клик - исключить область, Backspace - убрать точку, Esc - отмена", 12000)

    def on_smart_box(self, x1, y1, x2, y2):
        e = self.entry
        if not e:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if self._cv_cache[0] != e.path:
                self._cv_cache = (e.path, imgio.imread(e.path))
            poly = autolabel.smart_polygon(self._cv_cache[1], (x1, y1, x2, y2))
        finally:
            QApplication.restoreOverrideCursor()
        if poly:
            self.canvas.add_polygon(poly)
        else:
            self.statusBar().showMessage("Не удалось найти контур - попробуйте рамку поточнее или инструмент «Полигон»",
                                         4000)

    # ================================================================ import
    def import_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Импорт изображений", "",
                                                "Изображения (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)")
        if files:
            self._add_paths(files)

    def import_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Импорт папки (рекурсивно)")
        if d:
            self._add_paths([d])

    def _collect(self, paths):
        out = []
        for p in map(Path, paths):
            if p.is_dir():
                out += [str(q) for q in sorted(p.rglob("*")) if q.is_file() and imgio.is_image(q)]
            elif p.is_file():
                out.append(str(p))
        return out

    def _add_paths(self, paths):
        files = self._collect(paths)
        before = {i.path for i in self.project.images}
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            n = self.project.add_images(files)
        finally:
            QApplication.restoreOverrideCursor()
        new = [e for e in self.project.images if e.path not in before]
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._offer_label_import(new, paths)
        cur_path = self.entry.path if self.entry else None
        self.idx = next((i for i, e in enumerate(self.project.images) if e.path == cur_path), -1)
        self.refresh_all()
        if self.idx < 0 and self.project.images:
            self.load_image(0)
        self.statusBar().showMessage(f"Импортировано изображений: {n}", 4000)

    def _offer_label_import(self, new, roots):
        with_labels = [e for e in new if formats.find_label_file(e.path)]
        if not with_labels:
            return
        if QMessageBox.question(self, "Найдена готовая разметка",
                                f"Найдены файлы разметки YOLO для {len(with_labels)} изображений. Импортировать?") != QMessageBox.Yes:
            return
        if not self.project.classes:
            names = None
            for r in map(Path, roots):
                base = r if r.is_dir() else r.parent
                for up in [base, base.parent, base.parent.parent]:
                    for fn in ("data.yaml", "dataset.yaml", "classes.txt", "obj.names", "labels.txt"):
                        if (up / fn).exists() and names is None:
                            try:
                                names = formats.read_class_names(up / fn)
                            except Exception:
                                pass
            if names is None:
                p, _ = QFileDialog.getOpenFileName(self, "Названия классов для этой разметки (txt / names / yaml)", "",
                                                   "Файлы классов (*.txt *.names *.yaml *.yml)")
                names = formats.read_class_names(p) if p else []
            for n in names:
                self.project.add_class(str(n))
        total = sum(formats.import_yolo_labels(e, len(self.project.classes), len(self.project.keypoints)) for e in with_labels)
        self.project.dirty = True
        self.statusBar().showMessage(f"Импортировано объектов: {total}", 4000)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if paths:
            if len(paths) == 1 and paths[0].lower().endswith(".mlide.json"):
                if self._maybe_save():
                    self._open_path(paths[0])
            else:
                self._add_paths(paths)

    # ================================================================ project files
    def _maybe_save(self) -> bool:
        """True = the current project may be replaced / closed."""
        if self.project.dirty:
            r = QMessageBox.question(self, "Несохранённые изменения", "Сохранить изменения в проекте?",
                                     QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
            if r == QMessageBox.Cancel:
                return False
            if r == QMessageBox.Save and not self.save_project():
                return False
        self._clear_recovery()          # saved or deliberately discarded: nothing left to recover
        return True

    # ---- crash recovery for projects that were never saved to a file
    @staticmethod
    def _recovery_path() -> Path:
        return Path(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)) / "recovery.mlide.json"

    def _clear_recovery(self):
        try:
            self._recovery_path().unlink()
        except OSError:
            pass

    def offer_recovery(self) -> bool:
        """At start-up: restore an unsaved project left behind by a crash. True if restored."""
        rp = self._recovery_path()
        if not rp.exists():
            return False
        r = QMessageBox.question(self, "Восстановление", "Найдена автосохранённая копия несохранённого проекта "
                                 "(программа была закрыта некорректно). Восстановить её?")
        if r != QMessageBox.Yes:
            self._clear_recovery()
            return False
        try:
            pr = Project.load(rp)
        except Exception as ex:  # noqa
            QMessageBox.critical(self, "Восстановление", f"Не удалось прочитать копию:\n{type(ex).__name__}: {ex}")
            return False
        pr.path = None                  # still unsaved: the user picks a real location on Ctrl+S
        pr.dirty = True
        self.project = pr
        self.idx = -1
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.refresh_all()
        if self.project.images:
            self.load_image(0)
        return True

    def open_examples(self):
        if not self._maybe_save():
            return
        dlg = ExamplesDialog(self)
        if dlg.exec() == dlg.DialogCode.Accepted and dlg.result_path:
            self._open_path(dlg.result_path)

    def new_project(self):
        if not self._maybe_save():
            return
        self.project = Project()
        self.idx = -1
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.canvas.set_entry(None, None)
        self.refresh_all()

    def open_project(self):
        if not self._maybe_save():
            return
        p, _ = QFileDialog.getOpenFileName(self, "Открыть проект", "", "Проект ML IDE (*.mlide.json)")
        if p:
            self._open_path(p)

    def _open_path(self, p):
        try:
            self.project = Project.load(p)
        except Exception as ex:  # noqa
            QMessageBox.critical(self, "Открыть проект", f"{type(ex).__name__}: {ex}")
            return
        if getattr(self.project, "recovered_from_backup", False):
            QMessageBox.warning(self, "Проект восстановлен", "Файл проекта был повреждён: открыта его резервная копия (.bak). "
                                "Последние изменения могли не сохраниться.")
        missing = self.project.missing_images()
        self._remember(p)
        self.idx = -1
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.refresh_all()
        if self.project.images:
            self.load_image(0)
        if missing:
            r = QMessageBox.question(self, "Нет изображений", f"Не найдено файлов изображений: {len(missing)} "
                                     f"(перемещены или удалены), например:\n{missing[0].path}\n\n"
                                     "Указать папку, где они лежат сейчас?")
            if r == QMessageBox.Yes:
                d = QFileDialog.getExistingDirectory(self, "Папка с изображениями (ищу по имени файла)")
                if d:
                    fixed, left = self.project.relink(d)
                    self.refresh_all()
                    if self.idx >= 0:
                        self.load_image(self.idx)
                    self.statusBar().showMessage(f"Найдено файлов: {fixed}, не найдено: {left}", 8000)

    def save_project(self) -> bool:
        if not self.project.path:
            return self.save_project_as()
        try:
            self.project.save()
        except Exception as ex:  # noqa - disk full, file locked, read-only folder...
            self._save_failed(self.project.path, ex)
            return False
        self._title()
        self.statusBar().showMessage(f"Сохранено: {self.project.path}", 3000)
        return True

    def save_project_as(self) -> bool:
        p, _ = QFileDialog.getSaveFileName(self, "Сохранить проект", f"{self.project.name}.mlide.json",
                                           "Проект ML IDE (*.mlide.json)")
        if not p:
            return False
        if not p.lower().endswith(".mlide.json"):
            p += ".mlide.json"
        try:
            self.project.save(p)
        except Exception as ex:  # noqa
            self._save_failed(p, ex)
            return False
        self.project.name = Path(p).name[:-len(".mlide.json")]
        self._remember(p)
        self._clear_recovery()
        self._title()
        return True

    def _save_failed(self, path, ex):
        QMessageBox.critical(self, "Сохранение", f"Не удалось сохранить проект в\n{path}\n\n"
                             f"{type(ex).__name__}: {ex}\n\nИзменения остаются несохранёнными. Освободите место "
                             "или сохраните проект в другое место («Сохранить как»).")

    def _autosave(self):
        if not self.project.dirty:
            return
        try:
            if self.project.path:
                self.project.save()
                self._title()
            elif self.project.images:           # never saved: keep a recovery copy in the app data folder
                rp = self._recovery_path()
                rp.parent.mkdir(parents=True, exist_ok=True)
                self.project.write_to(rp)
        except Exception as ex:  # noqa
            self.statusBar().showMessage(f"Автосохранение не удалось: {type(ex).__name__}: {ex}", 15000)

    def closeEvent(self, e):
        e.ignore()                      # stays ignored if anything below raises
        if not self._maybe_save():
            return
        td = getattr(self, "train_dlg", None)
        if td is not None and td.running:
            if QMessageBox.question(self, "Идёт обучение", "Обучение ещё идёт. Остановить его и выйти?") != QMessageBox.Yes:
                return
            td.stop(wait=True)
        e.accept()

    # ================================================================ dialogs
    def export(self):
        if not self.project.images:
            QMessageBox.information(self, "Экспорт", "Сначала импортируйте и разметьте изображения.")
            return
        ExportDialog(self.project, self).exec()
        self._title()

    def presets(self):
        PresetDialog(self.project, self.entry, self).exec()
        self._title()

    def auto_annotate(self):
        if not self.project.images:
            QMessageBox.information(self, "Авторазметка", "Сначала импортируйте изображения.")
            return
        dlg = AutoLabelDialog(self.project, self.entry, self)
        dlg.exec()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.refresh_all()
        self.canvas.update()

    def stats(self):
        StatsDialog(self.project, self).exec()

    # ================================================================ тур
    def _tb(self, *actions):
        return rect_of(self, *[(self.tools_bar.widgetForAction(a) or self.toolbar.widgetForAction(a)) for a in actions])

    def tour_steps(self):
        S = [
            (None, "Добро пожаловать в ML IDE",
             "Это программа для подготовки данных и обучения нейросетей, которые <b>находят объекты на фото</b> "
             "(YOLO). Весь путь в одном окне:<br><b>импорт → разметка → проверка → экспорт → обучение → результаты</b>."
             "<br><br>Тур займёт минуту. Стрелки ← → или Enter - переключение, Esc - выход."),
            (lambda: rect_of(self, self.dock_images), "1. Ваши изображения",
             "Сюда добавляются фото: кнопки внизу, меню <i>Файл</i> или просто перетащите файлы/папку в окно. "
             "Значок перед именем: пустой круг - без разметки, точка - размечено, галочка - проверено. Фильтр сверху поможет найти "
             "неразмеченные картинки или те, где есть подсказки модели."),
            (lambda: rect_of(self, self.dock_classes), "2. Классы",
             "Классы - это то, что должна искать модель (например «зебра», «слон»). Добавьте их здесь. "
             "Клавиши <b>1-9</b> быстро выбирают класс. Рядом в скобках - сколько объектов уже размечено."),
            (lambda: self._tb(*self.tool_actions.values()), "3. Инструменты разметки",
             "<b>Выбор (V)</b> - двигать и менять размер.<br><b>Рамка (W)</b> - прямоугольник вокруг объекта.<br>"
             "<b>Полигон (P)</b> - точный контур для сегментации.<br><b>Умный контур (S)</b> - обведите объект "
             "грубой рамкой, программа найдёт границу сама.<br><b>Клик (SAM, G)</b> - кликните по объекту, нейросеть найдёт "
             "точный контур; Ctrl+клик исключает область, Enter принимает.<br><b>Повёрнутая рамка (O)</b>, <b>Ключевые точки (K)</b> и "
             "<b>Класс кадра (T)</b> - для наклонных объектов, поз и классификации целых изображений."),
            (lambda: rect_of(self, self.stack), "4. Рабочая область",
             "Рисуйте прямо на картинке. <b>Колесо мыши</b> - масштаб, <b>пробел + перетаскивание</b> или средняя кнопка - "
             "сдвиг. Размечайте <b>все</b> нужные объекты на кадре: пропущенный объект учит модель его игнорировать."),
            (lambda: rect_of(self, self.dock_objects), "5. Объекты на изображении",
             "Список разметки текущего кадра. Выбор здесь подсвечивает объект на картинке. Пунктирные объекты "
             "с «?» - это подсказки авторазметки, которые ждут вашей проверки."),
            (lambda: self._tb(self.a_prev, self.a_next, self.a_review), "6. Навигация и проверка",
             "<b>A / D</b> - предыдущее и следующее фото. <b>R</b> - отметить кадр проверенным и перейти дальше. "
             "<b>Ctrl+D</b> копирует разметку с предыдущего кадра (удобно для видео)."),
            (lambda: self._tb(self.a_auto, self.a_accept), "7. Авторазметка - компьютер подсказывает",
             "Загрузите обученную модель (.onnx или .pt), и она сама предложит рамки на остальных фото. "
             "Вы только проверяете: <b>Y</b> - принять подсказки, лишнее удалите. Так разметка в разы быстрее."),
            (lambda: self._tb(self.a_presets, self.a_stats), "8. Аугментации и проверка датасета",
             "<b>Пресеты аугментаций</b> - «искажения» для обучения (повороты, яркость, шум…) с живым превью на вашем фото. "
             "<b>Статистика</b> покажет, сколько объектов у каждого класса, и найдёт дубли и слишком мелкие рамки."),
            (lambda: self._tb(self.a_adv), "9. Советник по настройкам",
             "Анализирует датасет и подсказывает: хватает ли данных, нет ли перекоса классов, какую модель, "
             "сколько эпох и какой размер изображения выбрать. В окне экспорта есть кнопка «Подобрать настройки»."),
            (lambda: self._tb(self.a_export), "10. Экспорт датасета",
             "Сохраняет разметку в нужном формате: YOLO (v5-v12, 26: детекция, сегментация, OBB), Darknet, COCO, VOC. "
             "Сам делит выборку на train/val/test и создаёт готовый <code>train.py</code>."),
            (lambda: self._tb(self.a_train, self.a_results), "11. Обучение, результаты, экспорт модели",
             "<b>Обучение</b> запускает train.py из программы и показывает графики с объяснением простым языком. "
             "<b>Результаты</b> (Ctrl+Shift+R) - открыть любое прошлое обучение: метрики, матрица ошибок, примеры и "
             "проверка модели на вашем фото. Готовую модель можно выгрузить в ONNX и другие форматы: <i>Инструменты → Экспорт модели</i> "
             "(есть подготовка для ESP32-S3)."),
            (lambda: self._tb(self.a_ref), "12. Справочник всегда под рукой",
             "<b>F1</b> - руководства («с чего начать», «как читать графики», «что делать, если качество низкое») и "
             "словарь терминов: mAP, epoch, batch, переобучение… Горячие клавиши - Shift+F1."),
            (None, "Готово!",
             "Чтобы попробовать без своих данных, откройте готовый набор: <i>Файл → Примеры</i> - он скачается и "
             "откроется с разметкой. Затем пройдите путь: <i>Советник → Экспорт → Обучение → Результаты</i>."
             "<br><br>Повторить тур можно в любой момент: <i>Справка → Тур по приложению</i>."),
        ]
        return S

    def start_tour(self):
        if getattr(self, "_tour", None) is not None:
            try:
                self._tour.finish()
            except Exception:
                pass

        def done():
            self._tour = None
            settings.get().setValue("tour_done", True)

        self._tour = TourOverlay(self, self.tour_steps(), on_finish=done)

    def results(self):
        ResultsDialog(self.project, self).exec()

    def export_model(self):
        ModelExportDialog(self.project, self).exec()

    def view_model(self):
        ModelViewerDialog(self).exec()

    # ================================================================ импорт, классы, видео, снимки, недавние
    def _after_import(self, rep):
        cur = self.entry.path if self.entry else None
        self.idx = next((i for i, e in enumerate(self.project.images) if e.path == cur), -1)
        self.refresh_all()
        if self.idx < 0 and self.project.images:
            self.load_image(0)
        QMessageBox.information(self, "Импорт разметки", rep.summary())

    def import_coco(self):
        p, _ = QFileDialog.getOpenFileName(self, "Разметка COCO (JSON)", "", "COCO JSON (*.json)")
        if not p:
            return
        d = QFileDialog.getExistingDirectory(self, "Папка с изображениями (Отмена - искать рядом с json)")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            rep = importers.import_coco(self.project, p, d or None)
        except Exception as ex:  # noqa
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Импорт COCO", f"{type(ex).__name__}: {ex}")
            return
        QApplication.restoreOverrideCursor()
        self._after_import(rep)

    def import_voc(self):
        d = QFileDialog.getExistingDirectory(self, "Папка Pascal VOC (с Annotations/ и JPEGImages/ или xml рядом с фото)")
        if not d:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            rep = importers.import_voc(self.project, d)
        except Exception as ex:  # noqa
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Импорт VOC", f"{type(ex).__name__}: {ex}")
            return
        QApplication.restoreOverrideCursor()
        self._after_import(rep)

    def extract_video(self):
        dlg = VideoDialog(self)
        if dlg.exec() == dlg.DialogCode.Accepted and dlg.result_paths and dlg.add_to_project:
            self._add_paths(dlg.result_paths)

    def find_duplicates(self):
        if len(self.project.images) < 2:
            QMessageBox.information(self, "Дубли", "Нужно хотя бы два изображения.")
            return
        dlg = DedupeDialog(self.project, self)
        dlg.goto.connect(lambda i: (self.load_image(i), self.refresh_images()))
        dlg.exec()
        if dlg.changed:
            self.idx = -1
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.refresh_all()
            if self.project.images:
                self.load_image(0)

    def review_with_model(self):
        if not self.project.images:
            QMessageBox.information(self, "Проверка разметки", "Сначала добавьте изображения.")
            return
        self._review = ReviewDialog(self.project, self)
        self._review.goto.connect(lambda i: (self.load_image(i), self.refresh_images()))
        self._review.show()

    def compare_runs(self):
        CompareDialog(self.project, self).exec()

    def live_test(self):
        LiveTestDialog(self.project, self).exec()

    def check_updates(self):
        UpdateDialog(self).exec()

    def gpu_help(self):
        GpuHelpDialog(self).exec()

    def code_examples(self):
        names = self.project.class_names() or ["class0"]
        CodeSnippetsDialog("best.onnx", names, 640, self).exec()

    # ---- классы: объединение, порядок
    def _class_menu(self, pos):
        r = self.cls_list.currentRow()
        if r < 0:
            return
        from PySide6.QtWidgets import QMenu
        m = QMenu(self)
        m.addAction("Переименовать…", self.rename_class)
        m.addAction("Цвет…", self.recolor_class)
        sub = m.addMenu("Объединить с классом…")
        for i, c in enumerate(self.project.classes):
            if i != r:
                sub.addAction(c["name"], lambda i=i: self.merge_class(r, i))
        m.addSeparator()
        up = m.addAction("Переместить выше", lambda: self.move_class_to(r, r - 1))
        up.setEnabled(r > 0)
        dn = m.addAction("Переместить ниже", lambda: self.move_class_to(r, r + 1))
        dn.setEnabled(r < len(self.project.classes) - 1)
        m.addSeparator()
        m.addAction("Удалить класс…", self.remove_class)
        m.exec(self.cls_list.mapToGlobal(pos))

    def _classes_changed(self):
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.canvas.select(None)
        self.refresh_all()
        self.canvas.update()

    def merge_class(self, src, dst):
        a, b = self.project.classes[src]["name"], self.project.classes[dst]["name"]
        n = classtools.class_usage(self.project)[src]
        if QMessageBox.question(self, "Объединение классов", f"Все объекты класса «{a}» ({n}) станут классом «{b}», "
                                f"а класс «{a}» исчезнет. Продолжить?") != QMessageBox.Yes:
            return
        classtools.merge_classes(self.project, src, dst)
        self._classes_changed()

    def move_class_to(self, idx, pos):
        classtools.move_class(self.project, idx, pos)
        self._classes_changed()
        self.cls_list.setCurrentRow(pos)

    # ---- снимки разметки
    def save_snapshot(self):
        if not self.project.path:
            if not self.save_project_as():
                return
        elif self.project.dirty:
            self.save_project()
        import datetime
        d = self.project.path.parent / f"{self.project.name}_snapshots"
        d.mkdir(exist_ok=True)
        target = d / f"{self.project.name}_{datetime.datetime.now():%Y%m%d_%H%M%S}.mlide.json"
        try:
            self.project.write_to(target)
        except Exception as ex:  # noqa
            QMessageBox.warning(self, "Снимок", f"{type(ex).__name__}: {ex}")
            return
        self.statusBar().showMessage(f"Снимок сохранён: {target}", 8000)

    def open_snapshot(self):
        start = str(self.project.path.parent / f"{self.project.name}_snapshots") if self.project.path else ""
        p, _ = QFileDialog.getOpenFileName(self, "Открыть снимок разметки", start, "Проект ML IDE (*.mlide.json)")
        if p and self._maybe_save():
            self._open_path(p)

    # ---- недавние проекты
    def _recent(self) -> list:
        v = settings.get().value("recent", [])
        v = [v] if isinstance(v, str) else list(v or [])
        return [p for p in v if Path(p).exists()][:8]

    def _remember(self, path):
        s = settings.get()
        path = str(Path(path).resolve())
        lst = [p for p in self._recent() if os.path.normcase(p) != os.path.normcase(path)]
        s.setValue("recent", [path] + lst[:7])
        self._fill_recent()

    def _fill_recent(self):
        if not hasattr(self, "recent_menu"):
            return
        self.recent_menu.clear()
        items = self._recent()
        for p in items:
            self.recent_menu.addAction(Path(p).name, lambda p=p: self._open_recent(p))
        self.recent_menu.setEnabled(bool(items))
        self.welcome.set_recent(items)

    def _open_recent(self, p):
        if self._maybe_save():
            self._open_path(p)

    def _restart_app(self):
        if self._maybe_save():
            import subprocess
            args = [sys.executable] if getattr(sys, "frozen", False) else [sys.executable, "-m", "mlide"]
            if self.project.path:
                args.append(str(self.project.path))
            subprocess.Popen(args)
            QApplication.quit()

    def change_theme(self, name):
        from . import style
        if name == style.current_theme():
            return
        style.save_theme(name)
        if QMessageBox.question(self, "Тема оформления", "Тема изменится после перезапуска. Перезапустить сейчас?") == QMessageBox.Yes:
            self._restart_app()

    def change_language(self, code):
        from .. import i18n
        if code == i18n.lang():
            return
        i18n.save_language(code)
        r = QMessageBox.question(self, "Язык / Language", "Язык интерфейса изменится после перезапуска. Перезапустить сейчас?\n"
                                 "The interface language changes after a restart. Restart now?")
        if r == QMessageBox.Yes:
            self._restart_app()

    def show_reference(self, start=None):
        HelpDialog(SHORTCUTS_HTML, self, start if isinstance(start, str) else None).exec()

    def advisor(self):
        AdvisorDialog(self.project, self, can_apply=False).exec()

    def train(self):
        if getattr(self, "train_dlg", None) is None:
            self.train_dlg = TrainDialog(self.project, self)
        else:                           # one dialog for the whole session: a running training is never orphaned
            self.train_dlg.set_project(self.project)
        self.train_dlg.show()
        self.train_dlg.raise_()

    def show_help(self):
        QMessageBox.information(self, "Shortcuts", SHORTCUTS_HTML)


def install_translations(app):
    """Русские стандартные кнопки Qt (Сохранить, Отмена, Закрыть…)."""
    from PySide6.QtCore import QLibraryInfo, QTranslator
    app._tr = []
    for name in ("qtbase_ru", "qt_ru"):
        t = QTranslator(app)
        if t.load(name, QLibraryInfo.path(QLibraryInfo.TranslationsPath)):
            app.installTranslator(t)
            app._tr.append(t)


def selftest(out_path):
    """Проверка собранного приложения: все зависимости импортируются (запуск: ML IDE.exe --selftest файл.txt)."""
    lines = []
    for name in ("PySide6", "cv2", "numpy", "PIL", "yaml", "onnxruntime"):
        try:
            m = __import__(name)
            lines.append(f"{name}: OK {getattr(m, '__version__', '')}")
        except Exception as e:  # noqa
            lines.append(f"{name}: FAIL {type(e).__name__}: {e}")
    try:
        import onnxruntime as ort
        lines.append("onnxruntime providers: " + ",".join(ort.get_available_providers()))
        from .. import sam, examples, onnxparse, importers, dedupe, video, modelexport  # noqa
        lines.append("mlide modules: OK; icon: " + str(paths.resource("assets/icon.png").exists()))
        from .. import i18n
        lines.append(f"i18n dictionary entries: {len(i18n.load_table())}")
        if len(i18n.load_table()) < 500:
            lines.append("i18n: FAIL dictionary missing")
    except Exception as e:  # noqa
        lines.append(f"mlide modules: FAIL {type(e).__name__}: {e}")
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    return 0 if all("FAIL" not in ln for ln in lines) else 1


def selftest_sam(out_path):
    """Проверка «Клика (SAM)» в собранном приложении: скачивание моделей, кодировщик, декодер, маска.
    Запуск: ML IDE.exe --selftest-sam файл.txt   (модели кладутся в папку из MLIDE_MODELS, если она задана)."""
    import time

    import cv2
    import numpy as np
    lines = []
    ok = True
    try:
        from .. import sam
        lines.append(f"models dir: {sam.models_dir()}")
        t = time.time()
        sam.download_models(lambda n, d, tot: True)
        lines.append(f"download: OK ({time.time() - t:.0f} s), ready={sam.models_ready()}")
        m = sam.SamModel()
        img = np.zeros((300, 400, 3), np.uint8)
        cv2.circle(img, (200, 150), 70, (255, 255, 255), -1)
        t = time.time()
        m.set_image(img)
        mask, iou = m.predict([(200, 150, 1)])
        area = float((mask > 0).mean())
        expected = np.pi * 70 * 70 / (300 * 400)
        good = abs(area - expected) < 0.04
        ok = good
        poly = sam.mask_to_polygon(mask)
        lines.append(f"predict: {time.time() - t:.1f} s, iou={iou:.2f}, mask area={area:.3f} (expected {expected:.3f}), polygon points={len(poly or [])}"
                     + ("" if good else " FAIL: mask does not match the disk"))
    except Exception as e:  # noqa
        ok = False
        lines.append(f"SAM FAIL {type(e).__name__}: {e}")
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    return 0 if ok else 1


def run(argv):
    if argv and argv[0] == "--selftest-sam":
        return selftest_sam(argv[1] if len(argv) > 1 else "selftest_sam.txt")
    if argv and argv[0] == "--selftest":
        return selftest(argv[1] if len(argv) > 1 else "selftest.txt")
    from .. import i18n
    i18n.set_language(i18n.detect_language())
    app = QApplication(sys.argv[:1])
    app.setApplicationName("ML IDE")
    if i18n.lang() == "ru":
        install_translations(app)
    from .style import apply_theme, detect_theme
    apply_theme(app, detect_theme())
    from PySide6.QtGui import QIcon
    ico = paths.resource("assets/icon.ico")
    if ico.exists():
        app.setWindowIcon(QIcon(str(ico)))
    w = MainWindow()
    w.show()
    if os.environ.get("ML_IDE_NO_TOUR"):
        pass
    elif not settings.get().value("tour_done", False, type=bool):
        QTimer.singleShot(800, w.start_tour)
    restored = w.offer_recovery()
    if not restored and argv and Path(argv[0]).exists() and argv[0].lower().endswith(".mlide.json"):
        w._open_path(argv[0])
    return app.exec()

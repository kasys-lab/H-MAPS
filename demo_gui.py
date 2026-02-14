#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
demo_gui.py
- Modern Dark Glassmorphism UI
- Features: Minimize Toggle (Height Fixed), Trigger Badges, Notifications
- Fix: Aggressive margin reduction on minimize
- Added: Startup Usage Guide (English / Stylish)
"""

import webbrowser
import urllib.parse
from typing import List

from PySide6.QtCore import (
    Qt, QTimer, Signal, QPropertyAnimation, QEasingCurve, QRect, QSize
)
from PySide6.QtGui import QFontMetrics, QColor, QResizeEvent, QFont, QCursor
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel,
    QSizePolicy, QGraphicsDropShadowEffect, QPushButton, QScrollArea,
    QGraphicsOpacityEffect
)

# ================= Utils =================

def wrap_text_to_lines(text: str, font_metrics: QFontMetrics, max_width_px: int, max_lines: int = 3) -> List[str]:
    if max_width_px <= 0: return [text]
    lines = []
    current = ""
    for ch in text:
        if ch == "\n":
            lines.append(current)
            current = ""
            if len(lines) >= max_lines: break
            continue
        if font_metrics.horizontalAdvance(current + ch) <= max_width_px:
            current += ch
        else:
            lines.append(current)
            current = ch
            if len(lines) >= max_lines: break
    if current and len(lines) < max_lines:
        lines.append(current)
    if len(lines) == max_lines:
        last = lines[-1]
        if len(last.strip()) <= 5:
            lines = lines[:-1]
    return lines

# ================= Modern Card =================

class Card(QFrame):
    clicked = Signal()

    def __init__(self, title_text: str, body_text: str, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setFrameShape(QFrame.NoFrame)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.setCursor(Qt.PointingHandCursor)

        self.paper_id = None
        self.url = None
        self.title_raw = title_text
        self._body_original = body_text

        self.inner = QVBoxLayout(self)
        self.inner.setContentsMargins(16, 16, 16, 16)
        self.inner.setSpacing(6)

        self.title = QLabel(title_text, self)
        self.title.setObjectName("cardTitle")
        self.title.setWordWrap(True)
        self.title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.inner.addWidget(self.title)

        parts = body_text.split("\n", 1)
        first_line = parts[0].strip()
        if len(first_line) < 60 and any(c.isdigit() for c in first_line):
            self.chip_container = QWidget(self)
            self.chip_layout = QHBoxLayout(self.chip_container)
            self.chip_layout.setContentsMargins(0, 0, 0, 0)
            self.chip_layout.setSpacing(4)
            chip = QLabel(first_line, self.chip_container)
            chip.setObjectName("metaChip")
            self.chip_layout.addWidget(chip)
            self.chip_layout.addStretch(1)
            self.inner.addWidget(self.chip_container)
            self._body_to_display = parts[1].strip() if len(parts) > 1 else ""
        else:
            self._body_to_display = body_text

        self.body = QLabel(self._body_to_display, self)
        self.body.setObjectName("cardBody")
        self.body.setWordWrap(False)
        self.body.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.body.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.inner.addWidget(self.body)

        QTimer.singleShot(0, self.reflow_body)

    def reflow_body(self):
        m = self.inner.contentsMargins()
        available_width = self.width() - m.left() - m.right() - 10
        if available_width <= 50: return
        fm = QFontMetrics(self.body.font())
        lines = wrap_text_to_lines(self._body_to_display, fm, available_width, max_lines=3)
        new_text = "\n".join(lines)
        if self.body.text() != new_text:
            self.body.setText(new_text)

    def resizeEvent(self, e: QResizeEvent) -> None:
        super().resizeEvent(e)
        if e.oldSize().width() != e.size().width():
            QTimer.singleShot(0, self.reflow_body)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(e)

# ================= QuestionBlock =================

class QuestionBlock(QFrame):
    def __init__(self, question: str, cards: List[dict], trigger_type: str = None, on_card_clicked=None, parent=None):
        super().__init__(parent)
        self.setObjectName("qblock")
        self._expanded = False
        self._on_card_clicked = on_card_clicked

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(16, 16, 16, 16)
        self.main_layout.setSpacing(12)

        if trigger_type:
            badge_container = QWidget(self)
            badge_layout = QHBoxLayout(badge_container)
            badge_layout.setContentsMargins(0, 0, 0, 0)
            badge = QLabel(self)
            badge.setObjectName("badge")
            if trigger_type == "expansion": # Sustained Attention
                badge.setText(" 👁️ Sustained Attention")
                badge.setStyleSheet("""
                    background-color: rgba(59, 130, 246, 0.2); color: #60a5fa; 
                    border: 1px solid #3b82f6; border-radius: 6px; 
                    padding: 4px 8px; font-weight: bold; font-size: 11px;
                """)
            elif trigger_type == "support": # Content Revisit
                badge.setText(" ↩️ Content Revisit")
                badge.setStyleSheet("""
                    background-color: rgba(249, 115, 22, 0.2); color: #fb923c; 
                    border: 1px solid #f97316; border-radius: 6px; 
                    padding: 4px 8px; font-weight: bold; font-size: 11px;
                """)
            badge_layout.addWidget(badge)
            badge_layout.addStretch(1)
            self.main_layout.addWidget(badge_container)

        self.question_label = QLabel(question, self)
        self.question_label.setObjectName("questionLabel")
        self.question_label.setWordWrap(True)
        self.main_layout.addWidget(self.question_label)

        self.cards_container_top1 = QWidget(self)
        self.layout_top1 = QVBoxLayout(self.cards_container_top1)
        self.layout_top1.setContentsMargins(0, 0, 0, 0)
        self.layout_top1.setSpacing(10)
        self.main_layout.addWidget(self.cards_container_top1)

        self.extra_container = QWidget(self)
        self.extra_container.setObjectName("extraContainer")
        self.layout_extra = QVBoxLayout(self.extra_container)
        self.layout_extra.setContentsMargins(0, 0, 0, 0)
        self.layout_extra.setSpacing(10)
        self.extra_container.setAttribute(Qt.WA_LayoutUsesWidgetRect)
        self.extra_container.setMaximumHeight(0)
        self.main_layout.addWidget(self.extra_container)

        self.toggle_btn = QPushButton("Show more results", self)
        self.toggle_btn.setObjectName("toggleBtn")
        self.toggle_btn.setCursor(Qt.PointingHandCursor)
        self.toggle_btn.clicked.connect(self.toggle)
        self.main_layout.addWidget(self.toggle_btn)

        self.set_cards(cards)

        self.anim = QPropertyAnimation(self.extra_container, b"maximumHeight")
        self.anim.setDuration(300)
        self.anim.setEasingCurve(QEasingCurve.OutQuad)

    def set_cards(self, cards: List[dict]):
        for i, rec in enumerate(cards[:5]):
            raw_title = rec['title']
            display_title = raw_title if len(raw_title) < 90 else raw_title[:90] + "..."
            parent_widget = self.cards_container_top1 if i == 0 else self.extra_container
            layout_target = self.layout_top1 if i == 0 else self.layout_extra
            card = Card(display_title, rec["summary"], parent=parent_widget)
            card.paper_id = rec["paper_id"]
            card.url = rec.get("url")
            card.title_raw = raw_title
            if self._on_card_clicked:
                card.clicked.connect(lambda _=None, c=card: self._on_card_clicked(c))
            layout_target.addWidget(card)

        if len(cards) <= 1:
            self.toggle_btn.setVisible(False)
            self.extra_container.setVisible(False)
        else:
            self.toggle_btn.setVisible(True)

    def toggle(self):
        self._expanded = not self._expanded
        if self._expanded:
            start_h = 0
            self.extra_container.adjustSize()
            end_h = self.layout_extra.sizeHint().height()
            self.toggle_btn.setText("Hide extra results")
        else:
            start_h = self.extra_container.height()
            end_h = 0
            self.toggle_btn.setText("Show more results")
        self.anim.stop()
        self.anim.setStartValue(start_h)
        self.anim.setEndValue(end_h)
        self.anim.start()

# ================= Overlay Window =================

class Overlay(QWidget):
    hoverChanged = Signal(bool)
    manualPauseToggled = Signal(bool)

    def __init__(self):
        super().__init__(None)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._init_size()

        self.panel = QWidget(self)
        self.panel.setObjectName("root")
        self._set_stylesheet()

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(self.panel)

        # パネルレイアウト
        self.panel_layout = QVBoxLayout(self.panel)
        self.panel_layout.setContentsMargins(16, 16, 16, 16)
        self.panel_layout.setSpacing(10)

        # Header
        self.header_layout = QHBoxLayout()
        self.title_label = QLabel("H-MAPS Assistant")
        self.title_label.setObjectName("appTitle")
        self.header_layout.addWidget(self.title_label)
        
        # New Results Notification
        self.notify_label = QLabel("✨ New Results!", self.panel)
        self.notify_label.setObjectName("notifyLabel")
        self.notify_label.setVisible(False) 
        self.header_layout.addWidget(self.notify_label)
        
        self.header_layout.addStretch(1)
        
        # Minimize Btn
        self.min_btn = QPushButton("_", self.panel)
        self.min_btn.setObjectName("toolBtn")
        self.min_btn.setCursor(Qt.PointingHandCursor)
        self.min_btn.setFixedSize(30, 26)
        self.min_btn.clicked.connect(self._toggle_minimize)
        self.header_layout.addWidget(self.min_btn)

        # Pause Btn
        self.pause_btn = QPushButton("Active", self.panel)
        self.pause_btn.setObjectName("pauseBtn")
        self.pause_btn.setCheckable(True)
        self.pause_btn.setCursor(Qt.PointingHandCursor)
        self.pause_btn.clicked.connect(self._on_pause_clicked)
        self.header_layout.addWidget(self.pause_btn)
        
        self.panel_layout.addLayout(self.header_layout)

        # Content Area
        self.content_container = QWidget()
        content_layout = QVBoxLayout(self.content_container)
        content_layout.setContentsMargins(0, 0, 0, 0)
        
        self.scroll = QScrollArea(self.panel)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setFrameShape(QFrame.NoFrame)
        
        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("scrollContent")
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_layout.setSpacing(20)
        self.scroll.setWidget(self.scroll_content)
        
        content_layout.addWidget(self.scroll)
        self.panel_layout.addWidget(self.content_container)

        self.blocks: List[QuestionBlock] = []
        self.is_minimized = False
        self.expanded_height = 0 
        self._move_to_top_right()

        # ガイドを表示
        self.show_guide()

    def show_guide(self):
        """初期画面に使い方ガイドを表示する (Stylish English Version)"""
        guide_text = (
            "<div style='margin-top:20px; text-align: left; color: #cbd5e1; font-size: 13px; line-height: 1.6; font-family: \"Segoe UI\", sans-serif;'>"
            
            # Section 1: Triggers
            "<div style='margin-bottom: 12px; padding: 12px; background: rgba(255,255,255,0.05); border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);'>"
            "<strong style='color: #60a5fa; font-size: 14px; letter-spacing: 0.05em;'>OBSERVATION ENGINE</strong><br>"
            "<span style='color: #94a3b8; font-size: 12px;'>Retrieval is triggered by:</span><br>"
            "<span style='color: #93c5fd; font-weight:bold;'>• Sustained Attention</span> (Intensive Reading)<br>"
            "<span style='color: #fdba74; font-weight:bold;'>• Content Revisit</span> (Context Recovery)"
            "</div>"
            
            # Section 2: Pause
            "<div style='margin-bottom: 12px; padding: 12px; background: rgba(255,255,255,0.05); border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);'>"
            "<strong style='color: #60a5fa; font-size: 14px; letter-spacing: 0.05em;'>INTENT PROTECTION</strong><br>"
            "• <b style='color:#e2e8f0;'>Hover</b> over this window to pause updates.<br>"
            "• Click <b style='color:#34d399;'>[Active]</b> to toggle manual suspension."
            "</div>"

            # Section 3: Window
            "<div style='padding: 12px; background: rgba(255,255,255,0.05); border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);'>"
            "<strong style='color: #60a5fa; font-size: 14px; letter-spacing: 0.05em;'>WINDOW CONTROL</strong><br>"
            "• Minimize to header via <b style='color:#e2e8f0;'>[_]</b>.<br>"
            "• Drag the header to reposition."
            "</div>"
            "</div>"
        )
        lbl = QLabel(guide_text)
        lbl.setTextFormat(Qt.RichText)
        lbl.setWordWrap(True)
        lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        lbl.setStyleSheet("background: transparent;")
        
        self.scroll_layout.addWidget(lbl)
        self.scroll_layout.addStretch(1)

    def _init_size(self):
        screen = QApplication.primaryScreen()
        geo = screen.availableGeometry()
        #w = int(geo.width() * 0.22)
        #h = int(geo.height() * 0.85)
        w = int(geo.width() * 0.30)
        h = int(geo.height() * 0.95)
        self.resize(w, h)
        self.expanded_height = h

    def _toggle_minimize(self):
        if self.is_minimized:
            # Restore
            self.content_container.show()
            self.min_btn.setText("_")
            self.panel_layout.setContentsMargins(16, 16, 16, 16)
            self.setMinimumHeight(0)
            self.setMaximumHeight(16777215)
            self.resize(self.width(), self.expanded_height)
            self.is_minimized = False
            self.notify_label.setVisible(False)
        else:
            # Minimize
            self.expanded_height = self.height()
            self.content_container.hide()
            self.min_btn.setText("□")
            self.panel_layout.setContentsMargins(16, 4, 16, 4)
            self.setFixedHeight(58)
            self.is_minimized = True
        
        QTimer.singleShot(10, self._move_to_top_right)

    def _set_stylesheet(self):
        '''
        self.panel.setStyleSheet("""
            QWidget#root {
                background-color: rgba(30, 30, 35, 0.92); 
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 16px;
            }
            QLabel#appTitle {
                color: #e2e8f0; font-weight: 800; font-size: 14px;
            }
            QLabel#notifyLabel {
                color: #fbbf24; font-weight: bold; font-size: 12px;
                border: 1px solid #fbbf24; border-radius: 6px;
                padding: 2px 6px; margin-left: 8px;
            }
            QScrollArea { background: transparent; }
            QWidget#scrollContent { background: transparent; }
            
            QFrame#qblock {
                background-color: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 12px;
            }
            QLabel#questionLabel {
                font-family: "Segoe UI", sans-serif;
                font-size: 14px; font-weight: 600; color: #e2e8f0;
            }
            QPushButton#toggleBtn {
                background: transparent; color: #94a3b8;
                font-size: 12px; font-weight: 600; border: none; text-align: left;
            }
            QPushButton#toggleBtn:hover { color: #60a5fa; }

            QFrame#card {
                background-color: #262626; border-radius: 8px;
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-left: 3px solid #3b82f6;
            }
            QFrame#card:hover {
                background-color: #333333; border-color: rgba(255, 255, 255, 0.3);
            }
            QLabel#cardTitle { font-size: 13px; font-weight: 700; color: #60a5fa; }
            QLabel#cardBody { font-size: 12px; color: #cbd5e1; line-height: 1.3; }
            QLabel#metaChip {
                background-color: rgba(255, 255, 255, 0.1); color: #94a3b8;
                border-radius: 4px; padding: 2px 6px; font-size: 10px;
            }

            QPushButton#toolBtn {
                background-color: rgba(255, 255, 255, 0.1); color: #e2e8f0;
                border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 8px;
                font-weight: 900; padding-bottom: 3px;
            }
            QPushButton#toolBtn:hover { background-color: rgba(255, 255, 255, 0.2); }

            QPushButton#pauseBtn {
                background-color: rgba(16, 185, 129, 0.15); color: #34d399; 
                border: 1px solid #059669; border-radius: 8px;
                padding: 4px 10px; font-weight: bold; font-size: 11px;
            }
            QPushButton#pauseBtn:checked {
                background-color: rgba(239, 68, 68, 0.15); color: #f87171;
                border-color: #b91c1c; text: "Paused";
            }
        """)
        '''
        
        self.panel.setStyleSheet("""
            QWidget#root {
                background-color: rgba(30, 30, 35, 0.92); 
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 16px;
            }
            QLabel#appTitle {
                color: #e2e8f0; font-weight: 800; font-size: 14px;
            }
            QLabel#notifyLabel {
                color: #fbbf24; font-weight: bold; font-size: 12px;
                border: 1px solid #fbbf24; border-radius: 6px;
                padding: 2px 6px; margin-left: 8px;
            }
            QScrollArea { background: transparent; }
            QWidget#scrollContent { background: transparent; }
            
            QFrame#qblock {
                background-color: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 12px;
            }
            QLabel#questionLabel {
                font-family: "Segoe UI", sans-serif;
                font-size: 14px; font-weight: 600; color: #e2e8f0;
            }
            QPushButton#toggleBtn {
                background: transparent; color: #94a3b8;
                font-size: 12px; font-weight: 600; border: none; text-align: left;
            }
            QPushButton#toggleBtn:hover { color: #60a5fa; }

            QFrame#card {
                background-color: #262626; border-radius: 8px;
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-left: 3px solid #3b82f6;
            }
            QFrame#card:hover {
                background-color: #333333; border-color: rgba(255, 255, 255, 0.3);
            }
            QLabel#cardTitle { font-size: 13px; font-weight: 700; color: #60a5fa; }
            QLabel#cardBody { font-size: 12px; color: #cbd5e1; line-height: 1.3; }
            QLabel#metaChip {
                background-color: rgba(255, 255, 255, 0.1); color: #94a3b8;
                border-radius: 4px; padding: 2px 6px; font-size: 10px;
            }

            QPushButton#toolBtn {
                background-color: rgba(255, 255, 255, 0.1); color: #e2e8f0;
                border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 8px;
                font-weight: 900; padding-bottom: 3px;
            }
            QPushButton#toolBtn:hover { background-color: rgba(255, 255, 255, 0.2); }

            QPushButton#pauseBtn {
                background-color: rgba(16, 185, 129, 0.15); color: #34d399; 
                border: 1px solid #059669; border-radius: 8px;
                padding: 4px 10px; font-weight: bold; font-size: 11px;
            }
            QPushButton#pauseBtn:checked {
                background-color: rgba(239, 68, 68, 0.15); color: #f87171;
                border-color: #b91c1c; text: "Paused";
            }
        """)


    def _on_pause_clicked(self):
        is_paused = self.pause_btn.isChecked()
        self.pause_btn.setText("Paused" if is_paused else "Active")
        self.manualPauseToggled.emit(is_paused)

    def enterEvent(self, event) -> None:
        super().enterEvent(event)
        self.hoverChanged.emit(True)

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        self.hoverChanged.emit(False)

    def _move_to_top_right(self):
        screen = QApplication.primaryScreen()
        geo = screen.availableGeometry()
        m = 20 
        self.move(geo.x() + geo.width() - self.width() - m, geo.y() + m)

    def _ensure_topmost(self):
        self.raise_()
        self.activateWindow()

    def get_mask_rect_physical(self):
        rect = self.frameGeometry()
        screen = QApplication.primaryScreen()
        dpr = getattr(screen, "devicePixelRatio", lambda: 1.0)()
        return int(rect.left()*dpr), int(rect.top()*dpr), int(rect.width()*dpr), int(rect.height()*dpr)

    def on_card_clicked(self, card: Card):
        print(f"[DEBUG] Card clicked: paper_id={card.paper_id}", flush=True)
        url = getattr(card, "url", None)
        if url:
            webbrowser.open(url)
            return
        title = getattr(card, "title_raw", "") or str(getattr(card, "paper_id", ""))
        q = urllib.parse.quote(title)
        webbrowser.open(f"https://www.semanticscholar.org/search?q={q}")

    def update_results(self, payload: List[dict]):
        print(f"[DEBUG] UI update: received {len(payload)} blocks", flush=True)
        
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        
        self.blocks.clear()

        for item in payload:
            q = (item.get("question") or "").strip()
            cards = item.get("cards") or []
            trigger_type = item.get("trigger_type") 
            if not q or not cards: continue
            
            block = QuestionBlock(
                q, cards, 
                trigger_type=trigger_type, 
                on_card_clicked=self.on_card_clicked, 
                parent=self.scroll_content
            )
            
            effect = QGraphicsOpacityEffect(block)
            block.setGraphicsEffect(effect)
            anim = QPropertyAnimation(effect, b"opacity")
            anim.setDuration(500)
            anim.setStartValue(0)
            anim.setEndValue(1)
            anim.setEasingCurve(QEasingCurve.OutQuad)
            anim.start()
            block._anim_ref = anim 

            self.scroll_layout.addWidget(block)
            self.blocks.append(block)
        
        self.scroll_layout.addStretch(1)
        
        # 最小化中は通知バッジを表示
        if self.is_minimized:
            self.notify_label.setVisible(True)
        
        self._move_to_top_right()
        self._ensure_topmost()

    def resizeEvent(self, e: QResizeEvent) -> None:
        super().resizeEvent(e)
        self._move_to_top_right()
        self._ensure_topmost()

    def showEvent(self, e) -> None:
        super().showEvent(e)
        self._ensure_topmost()

if __name__ == "__main__":
    app = QApplication([])
    w = Overlay()
    w.show()

    app.exec()

from __future__ import annotations

from typing import List, Tuple

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

# A section is (heading, explanation, matplotlib Figure).
ResultSection = Tuple[str, str, Figure]


class ResultsDialog(QDialog):
    """Scrollable window that shows each real-patient output under a heading
    with a short, paper-style explanation."""

    def __init__(self, patient_id: str, sections: List[ResultSection], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Prediction Results — {patient_id}")
        self.resize(1180, 900)
        self._figures = [figure for _, _, figure in sections]

        self.setStyleSheet(
            """
            QDialog, QScrollArea, QWidget#resultsContainer { background-color: #0d1117; }
            QLabel { color: #e6edf3; font-family: 'Segoe UI', 'Noto Sans', sans-serif; }
            QLabel#resultTitle {
                font-size: 18px;
                font-weight: 700;
                color: #7dc8ff;
                margin-top: 6px;
            }
            QLabel#resultExplain {
                font-size: 13px;
                color: #b9c6d3;
                margin-bottom: 4px;
            }
            QFrame#resultCard {
                background: #121922;
                border: 1px solid #253344;
                border-radius: 10px;
                padding: 10px;
                margin-bottom: 8px;
            }
            QFrame#resultDivider { background: #1c2836; max-height: 1px; }
            """
        )

        outer = QVBoxLayout(self)

        header = QLabel(
            f"These outputs are generated from the real predicted data for patient "
            f"{patient_id}. Every panel comes from this scan and the loaded model checkpoint."
        )
        header.setWordWrap(True)
        header.setObjectName("resultExplain")
        outer.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll, 1)

        container = QWidget()
        container.setObjectName("resultsContainer")
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(4, 4, 4, 4)
        container_layout.setSpacing(10)

        for heading, explanation, figure in sections:
            container_layout.addWidget(self._build_card(heading, explanation, figure))

        container_layout.addStretch(1)
        scroll.setWidget(container)

    def _build_card(self, heading: str, explanation: str, figure: Figure) -> QWidget:
        card = QFrame()
        card.setObjectName("resultCard")
        layout = QVBoxLayout(card)

        title = QLabel(heading)
        title.setObjectName("resultTitle")
        layout.addWidget(title)

        explain = QLabel(explanation)
        explain.setObjectName("resultExplain")
        explain.setWordWrap(True)
        layout.addWidget(explain)

        canvas = FigureCanvasQTAgg(figure)
        canvas.setMinimumHeight(430)
        layout.addWidget(canvas)

        return card

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        import matplotlib.pyplot as plt

        for figure in self._figures:
            plt.close(figure)
        super().closeEvent(event)

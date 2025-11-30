# -*- coding: utf-8 -*-
import sys
import os
import time
from collections import deque

import cv2
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QPushButton, QLabel, QFileDialog, QMessageBox, QGroupBox
from PyQt5.QtWidgets import QProgressBar, QButtonGroup, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QBrush, QRadialGradient
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QPointF

import mediapipe as mp

# ===============================================================
# [설정] 각도 보정 및 기본값
# ===============================================================
YAW_FIX = 1.0
PITCH_FIX = 1.0
ROLL_FIX = 1.0
SCALE_FACTOR = 0.6

FAST_MODE = False
POSTPROC_MODES = ["stat", "real", "gfpgan"]   # 통계처리기반 / Real-ESRGAN / GFPGAN
CURRENT_POSTPROC = None                      # 처음엔 선택 없음
ACCEPTABLE_RISK_VALUE = 0                   # 0: 미선택, 1: 안전, 2: 보통, 3: 위험

STRENGTH_TABLE_PATH = "optimal_strengths.csv"

APP_STYLESHEET = """
QMainWindow { background: #FAFAFA; }
QGroupBox { font-weight: bold; border: 1px solid #E5E5E5; border-radius: 10px; margin-top: 12px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; background: transparent; }
QLabel[cls="badge"] { padding: 5px 10px; border-radius: 12px; background: #F5F5F5; border: 1px solid #E0E0E0; font-weight: bold; }
QLabel[cls="card"]  { font-size: 16px; font-weight: 800; padding: 10px; border-radius: 12px; color: #111; }
QProgressBar { height: 16px; border: 1px solid #DDD; border-radius: 8px; background: #F3F3F3; text-align: center; color: transparent; }
QProgressBar::chunk { border-radius: 8px; }
QPushButton[cls="primary"] { background: #1677FF; color: white; font-weight: 700; padding: 10px 16px; border-radius: 10px; }
QPushButton[cls="primary"]:disabled { background: #A0C5FF; }
QPushButton[cls="ghost"] { background: white; border: 1px solid #DDD; border-radius: 10px; padding: 8px 12px; }
QPushButton.segment { background: #FFF; border: 1px solid #DDD; padding: 8px 12px; border-radius: 10px; }
QPushButton.segment:checked { background: #1677FF; color: #FFF; border-color: #1677FF; }
"""

# -------------------------------------------------------
# [UI] 각도 시각화 위젯
# -------------------------------------------------------
class AngleVisualizer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.setMinimumSize(150, 150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def update_angles(self, yaw, pitch, roll):
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        cx = w // 2
        cy = h // 2
        radius = min(w, h) // 2 - 10

        painter.setPen(QPen(QColor("#E0E0E0"), 2))
        painter.setBrush(QBrush(QColor("#F9F9F9")))
        painter.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)

        painter.setPen(QPen(QColor("#D0D0D0"), 1, Qt.DashLine))
        painter.drawLine(cx, cy - radius, cx, cy + radius)
        painter.drawLine(cx - radius, cy, cx + radius, cy)

        scale = radius / 90.0
        target_x = cx + int(self.yaw * scale)
        target_y = cy - int(self.pitch * scale)

        painter.save()
        painter.translate(cx, cy)
        painter.rotate(-self.roll)
        painter.setPen(QPen(QColor("#4DA3FF"), 3))
        painter.drawLine(-radius + 20, 0, radius - 20, 0)
        painter.restore()

        painter.setPen(QPen(QColor("#FF4D4F"), 3))
        painter.drawLine(cx, cy, target_x, target_y)
        painter.setBrush(QBrush(QColor("#FF4D4F")))
        painter.drawEllipse(target_x - 5, target_y - 5, 10, 10)

        painter.setPen(QColor("#333"))
        painter.setFont(QFont("Arial", 9, QFont.Bold))
        painter.drawText(5, 15, f"Y: {self.yaw:.1f}°")
        painter.drawText(5, 30, f"P: {self.pitch:.1f}°")
        painter.drawText(w - 60, 15, f"R: {self.roll:.1f}°")

# -------------------------------------------------------
# [데이터 로드] 각도/밝기/강도 테이블 (roll 포함)
# -------------------------------------------------------
def load_optimal_strengths(file_path):
    if not os.path.exists(file_path):
        print("[ERROR] strength table not found:", file_path)
        return None

    try:
        if file_path.lower().endswith((".xls", ".xlsx")):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
    except Exception as e:
        print("[ERROR] strength table load failed:", e)
        return None

    for col in ["method", "attacker", "risk_label"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    if "attacker" in df.columns:
        df["attacker"] = df["attacker"].replace({"": "none"}).fillna("none")

    data = {}
    for _, row in df.iterrows():
        try:
            method = str(row["method"]).strip().lower()
            attacker = str(row.get("attacker", "none")).strip().lower() or "none"
            risk_lb = str(row.get("risk_label", "normal")).strip().lower() or "normal"

            yaw = float(row["yaw"])
            pitch = float(row["pitch"])
            roll = float(row["roll"])
            br = float(row["bright"])
            st = float(row["strength"])
        except Exception:
            continue

        if any(np.isnan(v) for v in (yaw, pitch, roll, br, st)):
            continue

        key = (method, attacker, risk_lb)
        bucket = data.setdefault(
            key,
            {"yaw": [], "pitch": [], "roll": [], "brightness": [], "strength": []},
        )
        bucket["yaw"].append(yaw)
        bucket["pitch"].append(pitch)
        bucket["roll"].append(roll)
        bucket["brightness"].append(br)
        bucket["strength"].append(st)

    print(f"[INFO] strength table loaded ({len(data)} key combinations)")
    return data

# -------------------------------------------------------
# [강도 선택] yaw/pitch/roll/밝기 4D 거리로 최적 강도 찾기
# -------------------------------------------------------
def pick_strength_from_table(method, yaw, pitch, roll, brightness,
                             strengths_data, postproc_mode, target_level):
    DEFAULT_MOSAIC = 0.076
    DEFAULT_BLUR = 15.6

    if not strengths_data:
        return DEFAULT_MOSAIC if method == "mosaic" else DEFAULT_BLUR

    attacker_map = {
        "stat": "none",
        "real": "realesrgan",
        "gfpgan": "gfpgan",
    }
    attacker = attacker_map.get(postproc_mode, "none")

    risk_label_map = {
        1: "safe",
        2: "normal",
        3: "risky",
    }
    target_label = risk_label_map.get(target_level, "normal")

    keys_to_try = [
        (method, attacker, target_label),
        (method, attacker, "normal"),
        (method, attacker, "safe"),
        (method, attacker, "risky"),
        (method, "none", target_label),
        (method, "none", "normal"),
    ]

    best_strength = None
    best_d2 = 1e18

    for key in keys_to_try:
        bucket = strengths_data.get(key)
        if not bucket:
            continue

        ys = bucket["yaw"]
        ps = bucket["pitch"]
        rs = bucket["roll"]
        brs = bucket["brightness"]
        sts = bucket["strength"]

        if not ys:
            continue

        for i in range(len(sts)):
            dy = (ys[i] - yaw) / 90.0
            dp = (ps[i] - pitch) / 90.0
            dr = (rs[i] - roll) / 90.0
            db = (brs[i] - brightness) / 255.0

            d2 = dy * dy + dp * dp + dr * dr + db * db
            if d2 < best_d2:
                best_d2 = d2
                best_strength = float(sts[i])

        if best_strength is not None:
            break

    if best_strength is not None:
        return best_strength

    for key, bucket in strengths_data.items():
        if key[0] == method and bucket["strength"]:
            return float(bucket["strength"][0])

    return DEFAULT_MOSAIC if method == "mosaic" else DEFAULT_BLUR

# -------------------------------------------------------
# [MediaPipe] 각도 계산
# -------------------------------------------------------
def _find_pose_method2(points_xy):
    LE, RE, N, LM, RM = points_xy

    dPx_eyes = max((RE[0] - LE[0]), 1.0)
    dPy_eyes = RE[1] - LE[1]
    angle = np.arctan2(dPy_eyes, dPx_eyes)
    roll = np.degrees(angle)

    alpha = np.cos(-angle)
    beta = np.sin(-angle)

    def rotate(pt, center):
        x = pt[0] - center[0]
        y = pt[1] - center[1]
        nx = x * alpha - y * beta
        ny = x * beta + y * alpha
        return [nx + center[0], ny + center[1]]

    center = N
    LE_r = rotate(LE, center)
    RE_r = rotate(RE, center)
    N_r = N
    LM_r = rotate(LM, center)
    RM_r = rotate(RM, center)

    eye_center_x = (LE_r[0] + RE_r[0]) / 2.0
    face_width = abs(RE_r[0] - LE_r[0])
    yaw = ((N_r[0] - eye_center_x) / (face_width / 2.0)) * 90.0

    eye_center_y = (LE_r[1] + RE_r[1]) / 2.0
    mouth_center_y = (LM_r[1] + RM_r[1]) / 2.0
    face_height = abs(mouth_center_y - eye_center_y)
    nose_pos = (N_r[1] - eye_center_y) / (face_height + 1e-6)
    pitch = (0.4 - nose_pos) * 150.0

    yaw = float(np.clip(yaw * YAW_FIX * SCALE_FACTOR, -90, 90))
    pitch = float(np.clip(pitch * PITCH_FIX * SCALE_FACTOR, -90, 90))
    roll = float(roll * ROLL_FIX)
    return roll, yaw, pitch

# -------------------------------------------------------
# [Core] 영상 처리 및 익명화 적용
# -------------------------------------------------------
def apply_anonymization(frame, method, strengths_dict, face_mesh=None, draw_debug=True):
    global CURRENT_POSTPROC, ACCEPTABLE_RISK_VALUE

    h, w = frame.shape[:2]
    out = frame.copy()
    rgb_for_mp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    logs = []
    bsum = 0.0
    cnt = 0
    max_s = 0.0
    max_r = 0
    cur_y = cur_p = cur_r = max_ang = 0.0

    target_level = ACCEPTABLE_RISK_VALUE if ACCEPTABLE_RISK_VALUE > 0 else 2

    if face_mesh:
        results = face_mesh.process(rgb_for_mp)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                x_list = [lm.x for lm in face_landmarks.landmark]
                y_list = [lm.y for lm in face_landmarks.landmark]
                x1 = int(min(x_list) * w)
                y1 = int(min(y_list) * h)
                x2 = int(max(x_list) * w)
                y2 = int(max(y_list) * h)

                mw = (x2 - x1) * 0.1
                mh = (y2 - y1) * 0.1
                x1 = max(0, int(x1 - mw))
                y1 = max(0, int(y1 - mh))
                x2 = min(w, int(x2 + mw))
                y2 = min(h, int(y2 + mh))

                roi = out[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                b = float(np.mean(gray_roi))

                def get_pt(idx):
                    lm = face_landmarks.landmark[idx]
                    return [lm.x * w, lm.y * h]

                pts_ordered = np.array(
                    [get_pt(33), get_pt(263), get_pt(1), get_pt(61), get_pt(291)],
                    dtype=np.float32,
                )

                roll, yaw, pitch = _find_pose_method2(pts_ordered)
                cur_y, cur_p, cur_r = yaw, pitch, roll
                ang_val = float(max(abs(yaw), abs(pitch)))
                nose_pt = (int(pts_ordered[2][0]), int(pts_ordered[2][1]))

                s_val = pick_strength_from_table(
                    method,
                    yaw,
                    pitch,
                    roll,
                    b,
                    strengths_dict,
                    CURRENT_POSTPROC,
                    target_level,
                )

                risk = target_level

                bsum += b
                cnt += 1
                max_s = max(max_s, s_val)
                max_r = max(max_r, risk)
                max_ang = max(max_ang, ang_val)

                if method == "blur":
                    sigma = float(s_val)
                    if sigma > 0:
                        proc = cv2.GaussianBlur(roi, (0, 0), sigmaX=sigma)
                    else:
                        proc = roi
                else:
                    f = float(s_val)
                    mind = min(roi.shape[:2])
                    bs = max(1, int(np.ceil(f * mind)))
                    if bs > 1:
                        sh = max(1, roi.shape[0] // bs)
                        sw = max(1, roi.shape[1] // bs)
                        small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_LINEAR)
                        proc = cv2.resize(
                            small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST
                        )
                    else:
                        proc = roi

                out[y1:y2, x1:x2] = proc

                if draw_debug:
                    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    if nose_pt:
                        info = f"Y:{int(yaw)} P:{int(pitch)} R:{int(roll)} Lv:{target_level}"
                        cv2.putText(
                            out,
                            info,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                        )
                        length = 100
                        end_x = int(nose_pt[0] + length * np.sin(np.deg2rad(yaw)))
                        end_y = int(nose_pt[1] - length * np.sin(np.deg2rad(pitch)))
                        cv2.arrowedLine(out, nose_pt, (end_x, end_y), (0, 0, 255), 2)

                logs.append(
                    f"Y:{yaw:.1f} P:{pitch:.1f} R:{roll:.1f} | Bri:{b:.1f} | Str:{s_val:.4f}"
                )

    avg_b = (bsum / cnt) if cnt > 0 else 0.0
    return out, logs, avg_b, max_s, cur_y, cur_p, cur_r, max_ang, (max_r if cnt > 0 else 1)

# -------------------------------------------------------
# [Thread] 비디오 처리 스레드
# -------------------------------------------------------
class VideoThread(QThread):
    change = pyqtSignal(np.ndarray, np.ndarray, list, float, float, float, float, float, float, int)
    finished = pyqtSignal()

    def __init__(self, path, method, strengths, frame):
        super().__init__()
        self.path = path
        self.method = method
        self.strengths = strengths
        self.run_flag = True
        self.pause_flag = False
        self.mp_face_mesh = mp.solutions.face_mesh

    def run(self):
        cap = cv2.VideoCapture(self.path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        is_img = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) <= 1
        start = time.time()
        i = 0
        cache = None

        with self.mp_face_mesh.FaceMesh(
            max_num_faces=5, refine_landmarks=True, min_detection_confidence=0.5
        ) as face_mesh:
            while self.run_flag and cap.isOpened():
                if self.pause_flag:
                    self.msleep(100)
                    continue
                ret, frame = cap.read()
                if not ret:
                    break
                if i % 3 == 0 or cache is None:
                    cache = apply_anonymization(
                        frame.copy(), self.method, self.strengths, face_mesh, draw_debug=True
                    )

                self.change.emit(frame, *cache)

                if is_img:
                    self.run_flag = False
                else:
                    tgt = start + (i / fps)
                    now = time.time()
                    if now < tgt:
                        self.msleep(int((tgt - now) * 1000))
                i += 1
        cap.release()
        self.finished.emit()

    def stop(self):
        self.run_flag = False
        self.wait()

    def toggle(self):
        self.pause_flag = not self.pause_flag
        return self.pause_flag

# -------------------------------------------------------
# [GUI] 메인 윈도우
# -------------------------------------------------------
class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Anonymization Dashboard (COVA)")
        self.setGeometry(100, 100, 1500, 980)
        self.setStyleSheet(APP_STYLESHEET)

        self.optimal = load_optimal_strengths(STRENGTH_TABLE_PATH)
        if not self.optimal:
            QMessageBox.critical(self, "Error", f"강도 테이블 로드 실패: {STRENGTH_TABLE_PATH}")
            sys.exit()

        self.hist_s = deque(maxlen=300)
        self.context_data = deque(maxlen=300)  # 지금은 사용 안 하지만 그대로 둠
        self.path = None

        self.init_ui()

    def init_ui(self):
        w = QWidget()
        self.setCentralWidget(w)
        main = QVBoxLayout(w)

        vid = QHBoxLayout()
        self.lbl_org = QLabel("Original")
        self.lbl_org.setStyleSheet("background:#000;color:#fff;")
        self.lbl_org.setAlignment(Qt.AlignCenter)
        self.lbl_org.setMinimumSize(400, 300)

        self.lbl_ano = QLabel("Anonymized")
        self.lbl_ano.setStyleSheet("background:#000;color:#fff;")
        self.lbl_ano.setAlignment(Qt.AlignCenter)
        self.lbl_ano.setMinimumSize(400, 300)

        vid.addWidget(self.lbl_org)
        vid.addWidget(self.lbl_ano)
        main.addLayout(vid, 3)

        dash = QHBoxLayout()

        # 설정 그룹
        g_ctrl = QGroupBox("설정 (옵션)")
        ctrl_l = QVBoxLayout(g_ctrl)

        self.bg_pp = QButtonGroup(self)
        self.bg_pp.setExclusive(False)
        self.btn_pp_stat = QPushButton("통계처리기반")
        self.btn_pp_real = QPushButton("Real-ESRGAN")
        self.btn_pp_gfp = QPushButton("GFPGAN")
        for btn in [self.btn_pp_stat, self.btn_pp_real, self.btn_pp_gfp]:
            btn.setCheckable(True)
            btn.setProperty("class", "segment")
            btn.setChecked(False)
            ctrl_l.addWidget(btn)
            self.bg_pp.addButton(btn)

        self.btn_pp_stat.clicked.connect(lambda: self.on_post_clicked("stat", self.btn_pp_stat))
        self.btn_pp_real.clicked.connect(lambda: self.on_post_clicked("real", self.btn_pp_real))
        self.btn_pp_gfp.clicked.connect(lambda: self.on_post_clicked("gfpgan", self.btn_pp_gfp))

        ctrl_l.addSpacing(10)
        ctrl_l.addWidget(QLabel("수용 위험 레벨"))

        self.bg_rk = QButtonGroup(self)
        self.bg_rk.setExclusive(False)
        self.btn_r1 = QPushButton("안전 (Lv.1)")
        self.btn_r2 = QPushButton("보통 (Lv.2)")
        self.btn_r3 = QPushButton("위험 (Lv.3)")
        for i, btn in enumerate([self.btn_r1, self.btn_r2, self.btn_r3], start=1):
            btn.setCheckable(True)
            btn.setProperty("cls", "ghost")
            btn.setChecked(False)
            ctrl_l.addWidget(btn)
            self.bg_rk.addButton(btn, i)

        self.btn_r1.clicked.connect(lambda: self.on_accept_clicked(1))
        self.btn_r2.clicked.connect(lambda: self.on_accept_clicked(2))
        self.btn_r3.clicked.connect(lambda: self.on_accept_clicked(3))

        dash.addWidget(g_ctrl)

        # 환경 분석 그룹
        g_env = QGroupBox("환경 분석")
        env_l = QVBoxLayout(g_env)

        self.lbl_bri = QLabel("밝기: -")
        self.pb_bri = QProgressBar()
        self.pb_bri.setRange(0, 255)
        self.set_bar_color(self.pb_bri, "#4DA3FF")

        env_l.addWidget(self.lbl_bri)
        env_l.addWidget(self.pb_bri)
        env_l.addWidget(QLabel("실시간 얼굴 각도"))
        self.angle_viz = AngleVisualizer()
        env_l.addWidget(self.angle_viz)

        dash.addWidget(g_env)

        # 분석 결과 그룹
        g_res = QGroupBox("분석 결과")
        res_l = QVBoxLayout(g_res)

        self.lbl_str = QLabel("강도: -")
        self.pb_str = QProgressBar()
        self.pb_str.setRange(0, 100)
        self.set_bar_color(self.pb_str, "#00B050")

        self.cv_str = QLabel()
        self.cv_str.setMinimumHeight(80)
        self.cv_context = QLabel()
        self.cv_context.setMinimumHeight(80)

        res_l.addWidget(self.lbl_str)
        res_l.addWidget(self.pb_str)
        res_l.addWidget(QLabel("강도 변화 (최근 프레임)"))
        res_l.addWidget(self.cv_str)
        res_l.addWidget(QLabel("Risk Radar (Context Map)"))
        res_l.addWidget(self.cv_context)

        dash.addWidget(g_res)
        main.addLayout(dash, 2)

        # 하단 컨트롤
        ctrl = QHBoxLayout()
        b_open = QPushButton("열기")
        b_open.clicked.connect(self.load_file)

        self.btn_save = QPushButton("저장")
        self.btn_save.clicked.connect(self.save_file)
        self.btn_save.setEnabled(False)

        self.btn_reset = QPushButton("초기화")
        self.btn_reset.clicked.connect(self.reset_app)
        self.btn_reset.setProperty("cls", "ghost")

        self.btn_run = QPushButton("시작")
        self.btn_run.clicked.connect(self.toggle_run)

        self.bg_mode = QButtonGroup(self)
        b_m = QPushButton("Mosaic")
        b_m.setCheckable(True)
        b_m.setChecked(True)
        b_b = QPushButton("Blur")
        b_b.setCheckable(True)
        self.bg_mode.addButton(b_m, 1)
        self.bg_mode.addButton(b_b, 2)

        ctrl.addWidget(b_open)
        ctrl.addWidget(b_m)
        ctrl.addWidget(b_b)
        ctrl.addStretch()
        ctrl.addWidget(self.btn_run)
        ctrl.addWidget(self.btn_save)
        ctrl.addWidget(self.btn_reset)
        main.addLayout(ctrl)

        self.status = QLabel("Ready")
        main.addWidget(self.status)

        self.draw_graph(self.cv_str, [], 0, 1, "#1677FF", mode="line")
        self.draw_graph(self.cv_context, [], 0, 1, "#FF4D4F", mode="radar")

    # ----------------- UI Helpers -----------------
    def set_bar_color(self, bar, color):
        bar.setStyleSheet(f"QProgressBar::chunk{{background:{color};border-radius:7px;}}")

    def on_post_clicked(self, mode, btn):
        global CURRENT_POSTPROC
        # 토글 동작: 이미 선택되어 있으면 해제
        if CURRENT_POSTPROC == mode:
            CURRENT_POSTPROC = None
            btn.setChecked(False)
        else:
            CURRENT_POSTPROC = mode
            for b in [self.btn_pp_stat, self.btn_pp_real, self.btn_pp_gfp]:
                if b is btn:
                    b.setChecked(True)
                else:
                    b.setChecked(False)

    def on_accept_clicked(self, lv):
        global ACCEPTABLE_RISK_VALUE
        if ACCEPTABLE_RISK_VALUE == lv:
            ACCEPTABLE_RISK_VALUE = 0
            for b in [self.btn_r1, self.btn_r2, self.btn_r3]:
                b.setChecked(False)
                b.setStyleSheet("background:#F3F3F3;color:black;")
            return

        ACCEPTABLE_RISK_VALUE = lv
        colors = ["#8BC34A", "#FFC107", "#F44336"]
        for i, b in enumerate([self.btn_r1, self.btn_r2, self.btn_r3], start=1):
            if i == lv:
                b.setChecked(True)
                b.setStyleSheet(f"background-color:{colors[i-1]};color:white;")
            else:
                b.setChecked(False)
                b.setStyleSheet("background:#F3F3F3;color:black;")

    def load_file(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "", "", "Media (*.mp4 *.avi *.mov *.mkv *.jpg *.png)"
        )
        if p:
            self.path = p
            cap = cv2.VideoCapture(p)
            ret, f = cap.read()
            cap.release()
            if ret:
                self.set_img(self.lbl_org, f)
            self.status.setText(f"Loaded: {os.path.basename(p)}")
            self.btn_save.setEnabled(False)

    def reset_app(self):
        global CURRENT_POSTPROC, ACCEPTABLE_RISK_VALUE
        if hasattr(self, "th") and self.th.isRunning():
            self.th.stop()
        self.path = None
        self.lbl_org.clear()
        self.lbl_org.setText("Original")
        self.lbl_ano.clear()
        self.lbl_ano.setText("Anonymized")
        self.pb_bri.setValue(0)
        self.lbl_bri.setText("밝기: -")
        self.angle_viz.update_angles(0, 0, 0)
        self.pb_str.setValue(0)
        self.lbl_str.setText("강도: -")
        self.hist_s.clear()
        self.context_data.clear()
        self.draw_graph(self.cv_str, [], 0, 1, "#1677FF", mode="line")
        self.draw_graph(self.cv_context, [], 0, 1, "#FF4D4F", mode="radar")
        self.status.setText("Ready")
        self.btn_save.setEnabled(False)
        self.btn_run.setText("시작")

        CURRENT_POSTPROC = None
        for b in [self.btn_pp_stat, self.btn_pp_real, self.btn_pp_gfp]:
            b.setChecked(False)

        ACCEPTABLE_RISK_VALUE = 0
        for b in [self.btn_r1, self.btn_r2, self.btn_r3]:
            b.setChecked(False)
            b.setStyleSheet("background:#F3F3F3;color:black;")

    def toggle_run(self):
        if not self.path:
            return
        if CURRENT_POSTPROC is None:
            QMessageBox.warning(self, "Alert", "설정(옵션)을 먼저 선택하세요.")
            return
        if ACCEPTABLE_RISK_VALUE == 0:
            QMessageBox.warning(self, "Alert", "수용 위험 레벨을 선택하세요.")
            return

        if hasattr(self, "th") and self.th.isRunning():
            paused = self.th.toggle()
            self.btn_run.setText("Resume" if paused else "Pause")
            self.btn_save.setEnabled(paused)
        else:
            method = "mosaic" if self.bg_mode.button(1).isChecked() else "blur"
            self.th = VideoThread(self.path, method, self.optimal, None)
            self.th.change.connect(self.update_ui)
            self.th.finished.connect(
                lambda: [self.status.setText("Done"), self.btn_save.setEnabled(True)]
            )
            self.th.start()
            self.btn_run.setText("Pause")
            self.hist_s.clear()
            self.context_data.clear()

    def update_ui(self, org, ano, logs, bri, strength, yaw, pitch, roll, max_ang, risk):
        self.set_img(self.lbl_org, org)
        self.set_img(self.lbl_ano, ano)

        self.pb_bri.setValue(int(bri))
        self.lbl_bri.setText(f"Bright: {bri:.1f}")
        self.angle_viz.update_angles(yaw, pitch, roll)

        method = "blur" if self.bg_mode.button(2).isChecked() else "mosaic"
        if method == "blur":
            bv = int((strength / 16.0) * 100)
            ymax = 16.0
        else:
            bv = int((strength / 0.10) * 100)
            ymax = 0.10
        self.pb_str.setValue(min(100, bv))
        self.lbl_str.setText(f"Str: {strength:.4f}")

        # 강도 변화 히스토리 (위쪽 선 그래프)
        self.hist_s.append(strength)

        # 레이더용 데이터 (현재 프레임 1개만 사용)
        radar_data = [(yaw, pitch, strength, ymax)]

        self.draw_graph(self.cv_str, list(self.hist_s), 0, ymax, "#1677FF", mode="line")
        self.draw_graph(self.cv_context, radar_data, 0, ymax, "#FF4D4F", mode="radar")

    def draw_graph(self, lbl, data, ymin, ymax, color, mode="line"):
        pix = QPixmap(lbl.width(), lbl.height())
        pix.fill(Qt.white)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing)
        w = pix.width()
        h = pix.height()

        # 공통 테두리
        p.setPen(QColor("#EEE"))
        p.drawRect(0, 0, w - 1, h - 1)

        if mode == "line":
            # 강도 변화 (최근 프레임) 그래프
            if len(data) > 1:
                p.setPen(QPen(QColor(color), 2))
                pts = []
                for i, v in enumerate(data):
                    x = 5 + i * (w - 10) / max(1, len(data) - 1)
                    y = (h - 5) - ((v - ymin) / (ymax - ymin + 1e-6) * (h - 15))
                    pts.append((x, y))
                for i in range(len(pts) - 1):
                    p.drawLine(
                        int(pts[i][0]),
                        int(pts[i][1]),
                        int(pts[i + 1][0]),
                        int(pts[i + 1][1]),
                    )

        elif mode == "radar":
            # Risk Radar (Context Map)
            cx = w / 2
            cy = h / 2 + 8
            radius = min(w, h) / 2 - 18

            # 배경 그라디언트 (중심 위험, 외곽 안전)
            grad = QRadialGradient(cx, cy, radius)
            grad.setColorAt(0.0, QColor(255, 230, 230))
            grad.setColorAt(0.4, QColor(255, 255, 255))
            grad.setColorAt(1.0, QColor(230, 255, 230))
            p.setBrush(QBrush(grad))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(cx, cy), radius, radius)

            # 동심원 + 십자선
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor("#DDD"), 1, Qt.DashLine))
            p.drawEllipse(QPointF(cx, cy), radius * 0.33, radius * 0.33)
            p.drawEllipse(QPointF(cx, cy), radius * 0.66, radius * 0.66)
            p.setPen(QPen(QColor("#BBB"), 2))
            p.drawEllipse(QPointF(cx, cy), radius, radius)

            p.setPen(QPen(QColor("#DDD"), 1))
            p.drawLine(int(cx - radius), int(cy), int(cx + radius), int(cy))
            p.drawLine(int(cx), int(cy - radius), int(cx), int(cy + radius))

            # 중앙 제목
            #p.setPen(QColor("#333"))
            #p.setFont(QFont("Arial", 9, QFont.Bold))
            #p.drawText(8, 16, "Risk Radar (Context Map)")

            # 현재 프레임 점 찍기
            if data:
                # data = [(yaw, pitch, strength, max_ref)]
                curr_yaw, curr_pitch, curr_str, max_ref = data[0]

                tx = cx + (curr_yaw / 90.0) * radius
                ty = cy - (curr_pitch / 90.0) * radius

                tx = max(cx - radius, min(cx + radius, tx))
                ty = max(cy - radius, min(cy + radius, ty))

                dist = np.sqrt((tx - cx) ** 2 + (ty - cy) ** 2) / max(radius, 1e-6)
                is_danger = dist < 0.3
                dot_color = QColor("#FF4D4F") if is_danger else QColor("#52C41A")

                norm_str = max(0.2, min(1.0, curr_str / (max_ref + 1e-6)))
                dot_size = 10 + (norm_str * 20)

                p.setPen(QPen(dot_color, 2))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(QPointF(tx, ty), dot_size, dot_size)

                p.setBrush(QBrush(dot_color))
                p.drawEllipse(QPointF(tx, ty), 6, 6)

                status_text = "HIGH RISK" if is_danger else "SAFE ZONE"
                p.setPen(dot_color)
                p.setFont(QFont("Arial", 9, QFont.Bold))
                p.drawText(int(cx - 40), int(cy - radius - 5), status_text)

        p.end()
        lbl.setPixmap(pix)

    def set_img(self, lbl, img):
        if img is None:
            return
        h, w, c = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        q = QImage(rgb.data, w, h, c * w, QImage.Format_RGB888)
        lbl.setPixmap(QPixmap.fromImage(q).scaled(lbl.size(), Qt.KeepAspectRatio))

    def save_file(self):
        if not self.path:
            return
        cap = cv2.VideoCapture(self.path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        ret, f0 = cap.read()
        if not ret:
            return

        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            max_num_faces=5, refine_landmarks=True, min_detection_confidence=0.5
        ) as face_mesh:
            if n <= 1:
                p, _ = QFileDialog.getSaveFileName(self, "", "out.jpg", "Img(*.jpg)")
                if p:
                    m = "mosaic" if self.bg_mode.button(1).isChecked() else "blur"
                    o, _, _, _, _, _, _, _, _ = apply_anonymization(
                        f0, m, self.optimal, face_mesh, draw_debug=False
                    )
                    cv2.imencode(os.path.splitext(p)[1], o)[1].tofile(p)
            else:
                p, _ = QFileDialog.getSaveFileName(self, "", "out.avi", "Avi(*.avi)")
                if p:
                    writer = cv2.VideoWriter(
                        p,
                        cv2.VideoWriter_fourcc(*"MJPG"),
                        fps,
                        (f0.shape[1], f0.shape[0]),
                    )
                    m = "mosaic" if self.bg_mode.button(1).isChecked() else "blur"
                    o, _, _, _, _, _, _, _, _ = apply_anonymization(
                        f0, m, self.optimal, face_mesh, draw_debug=False
                    )
                    writer.write(o)
                    cnt = 1
                    while True:
                        ret, f = cap.read()
                        if not ret:
                            break
                        o, _, _, _, _, _, _, _, _ = apply_anonymization(
                            f, m, self.optimal, face_mesh, draw_debug=False
                        )
                        writer.write(o)
                        cnt += 1
                        if cnt % 10 == 0:
                            self.status.setText(f"Saving {cnt}/{n}")
                            QApplication.processEvents()
                    writer.release()
        cap.release()
        self.status.setText("Saved")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AppWindow()
    window.show()
    sys.exit(app.exec_())

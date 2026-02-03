import pygame
import cv2
import mediapipe as mp
import sys
import time
import json
import socket
import threading
import os
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# ===================== UDP AGENT (НЕ змінює протокол контролера) =====================
LISTEN_IP = "0.0.0.0"
LISTEN_PORT = 5005
AGENT_ID = socket.gethostname()

# reward (можна отримувати/ставити)
REWARD = 1337

# поточний стан агента
_state = {"name": "RUNNING"}
_state_lock = threading.Lock()

# сокет
_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_sock.bind((LISTEN_IP, LISTEN_PORT))


def _send(to_ip: str, to_port: int, msg: dict):
    """Відправка JSON-повідомлення через UDP"""
    data = (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")
    _sock.sendto(data, (to_ip, to_port))


def _handle_message(msg: dict, addr):
    global REWARD

    mtype = msg.get("type")
    req_id = msg.get("req_id")
    reply_port = int(msg.get("reply_port", 5006))  # куди слати відповіді
    controller_ip = addr[0]

    if mtype == "ping":
        with _state_lock:
            cur = _state["name"]
        _send(controller_ip, reply_port, {
            "type": "ack",
            "agent_id": AGENT_ID,
            "req_id": req_id,
            "ok": True,
            "data": {"pong": True, "state": cur},
        })
        return

    if mtype == "set_state":
        new_state = msg.get("state")
        if new_state not in {"IDLE", "RUNNING", "PAUSED", "STOPPED"}:
            _send(controller_ip, reply_port, {
                "type": "ack",
                "agent_id": AGENT_ID,
                "req_id": req_id,
                "ok": False,
                "error": "invalid_state",
            })
            return

        with _state_lock:
            old = _state["name"]
            _state["name"] = new_state

        _send(controller_ip, reply_port, {
            "type": "ack",
            "agent_id": AGENT_ID,
            "req_id": req_id,
            "ok": True,
            "data": {"prev": old, "state": new_state},
        })
        return

    if mtype == "get_state":
        with _state_lock:
            cur = _state["name"]
        _send(controller_ip, reply_port, {
            "type": "ack",
            "agent_id": AGENT_ID,
            "req_id": req_id,
            "ok": True,
            "data": {"state": cur},
        })
        return

    # ===================== reward API =====================
    if mtype == "get_reward":
        with _state_lock:
            cur = REWARD
        _send(controller_ip, reply_port, {
            "type": "ack",
            "agent_id": AGENT_ID,
            "req_id": req_id,
            "ok": True,
            "data": {"reward": cur},
        })
        return

    if mtype == "set_reward":
        new_reward = msg.get("reward")
        with _state_lock:
            old = REWARD
            REWARD = new_reward

        _send(controller_ip, reply_port, {
            "type": "ack",
            "agent_id": AGENT_ID,
            "req_id": req_id,
            "ok": True,
            "data": {"prev": old, "state": new_reward},
        })
        return

    # невідомий тип повідомлення
    _send(controller_ip, reply_port, {
        "type": "ack",
        "agent_id": AGENT_ID,
        "req_id": req_id,
        "ok": False,
        "error": "unknown_type",
    })


def _udp_loop():
    """Фоновий потік для обробки UDP"""
    while True:
        try:
            data, addr = _sock.recvfrom(64 * 1024)
            msg = json.loads(data.decode("utf-8").strip())
            _handle_message(msg, addr)
        except Exception:
            # ігноруємо помилки некоректного JSON або recv
            continue


# ===================== запуск агента у фоні =====================
def start_agent():
    t = threading.Thread(target=_udp_loop, daemon=True)
    t.start()

pygame.init()
pygame.font.init()
start_agent()

# ===================== SCREEN & SCALE =====================
info = pygame.display.Info()
WIDTH, HEIGHT = info.current_w, info.current_h
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))  # Псевдо-фулскрин
pygame.display.set_caption("Розміщення фігур")

BASE_W, BASE_H = 640, 480
SCALE_X = WIDTH / BASE_W
SCALE_Y = HEIGHT / BASE_H
FPS = 30

FONT_PATH = "images/Jura-Regular.ttf"
FONT = pygame.font.Font(FONT_PATH, int(32 * SCALE_Y))

# ===================== STATES =====================
STATE_MENU = 0
STATE_INSTRUCTIONS = 1
STATE_GAME = 2
STATE_LEVEL_COMPLETE = 3
STATE_GAME_OVER = 4
STATE_EXIT = 5
state = STATE_MENU

def reset_hold_state():
    global active_button
    active_button = None
    hand_cursor["hovering"] = False
    hand_cursor["hover_time"] = 0
    hand_cursor["click"] = False
    hand_cursor["click_button"] = None

# ===================== DEBUG FLAGS =====================
DEBUG_DRAW_LANDMARKS = True      # лендмарки руки (на pygame-екрані)
DEBUG_DRAW_HITBOXES = True       # хітбокси предметів/цілей (pygame rect)
DEBUG_DRAW_GRAB_AREA = True      # зона "хвату" (circle) для пальця
DEBUG_DRAW_GRAB_STATE = True     # текст "GRAB/OPEN"

# ===================== GRAB CIRCLE SETTINGS =====================
GRAB_CIRCLE_RADIUS = 100  # коло взаємодії (в пікселях екрана)

# ===================== MOUSE =====================
mouse = {"click": False, "pos": (0, 0)}

# ===================== HAND CURSOR =====================
HOLD_TIME = 1.5

hand_cursor = {
    "pos": (0, 0),
    "hovering": False,
    "hover_time": 0,
    "click": False,
    "click_button": None   # 👈 новое
}

active_button = None

def handle_events():
    mouse["click"] = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse["click"] = True
            mouse["pos"] = event.pos
        if event.type == pygame.MOUSEMOTION:
            mouse["pos"] = event.pos

# ===================== CAMERA =====================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# ===================== CAMERA PREVIEW =====================
CAM_PREVIEW_W = int(100 * SCALE_X)
CAM_PREVIEW_H = int(93 * SCALE_Y)
CAM_PREVIEW_MARGIN = int(6 * SCALE_X)


def draw_camera_preview(screen):
    ok, frame = cap.read()
    if not ok:
        return

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.flip(rgb, 1)

    frame_small = cv2.resize(rgb, (CAM_PREVIEW_W, CAM_PREVIEW_H))
    frame_surf = pygame.surfarray.make_surface(frame_small)  # <-- безопаснее для Pygame
    frame_surf = pygame.transform.rotate(frame_surf, -90)    # если ориентация неправильная
    frame_surf = pygame.transform.flip(frame_surf, True, False)

    preview_rect = pygame.Rect(
        WIDTH - CAM_PREVIEW_W - CAM_PREVIEW_MARGIN,
        CAM_PREVIEW_MARGIN,
        CAM_PREVIEW_W,
        CAM_PREVIEW_H
    )

    screen.blit(frame_surf, (preview_rect.left, preview_rect.top))

    # Рисуем рамку **после blit**
    pygame.draw.rect(screen, (47, 207, 247), preview_rect, max(3, int(0.5 * SCALE_X)))

    # Рисуем лендмарки руки на маленьком окне
    res = hands.process(rgb)
    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0].landmark
        pts = []
        for i in range(21):
            x = int(lm[i].x * CAM_PREVIEW_W)
            y = int(lm[i].y * CAM_PREVIEW_H)
            pts.append((preview_rect.left + x, preview_rect.top + y))
            pygame.draw.circle(screen, (47, 207, 247), (preview_rect.left + x, preview_rect.top + y), 3)

        # Ребра между лендмарками
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # index
            (0, 9), (9, 10), (10, 11), (11, 12),   # middle
            (0, 13), (13, 14), (14, 15), (15, 16), # ring
            (0, 17), (17, 18), (18, 19), (19, 20), # pinky
            (5, 9), (9, 13), (13, 17)              # palm
        ]
        for a, b in edges:
            pygame.draw.line(screen, (47, 207, 247), pts[a], pts[b], 2)

# ===================== KRONOS WORD GAME CONSTANTS =====================
pygame.display.set_caption("KRONOS GATE")

# MediaPipe для гри "слово KRONOS"
mp_hands_game = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_game = mp_hands_game.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

LETTER_HOLD_TIME = 2.0
FINAL_HOLD_TIME = 3.0
OVERALL_TIME_LIMIT = 300.0
TARGET_WORD = "KRONOS"

# Цвета (BGR)
COLOR_MAIN = (255, 150, 50)
COLOR_ACCENT = (100, 255, 100)
COLOR_WHITE = (255, 255, 255)
COLOR_FINAL = (0, 200, 255)


def draw_text_pil(img, text, pos, size, color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(FONT_PATH, size)
    except Exception:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def overlay_png(bg, overlay, x, y):
    h, w = overlay.shape[:2]
    if x < 0 or y < 0 or x + w > bg.shape[1] or y + h > bg.shape[0]:
        return bg
    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3] / 255.0
        for c in range(3):
            bg[y:y + h, x:x + w, c] = alpha * overlay[:, :, c] + (1 - alpha) * bg[y:y + h, x:x + w, c]
    else:
        bg[y:y + h, x:x + w] = overlay[:, :, :3]
    return bg


def finger_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def get_landmark_points(landmarks):
    return [(lm.x, lm.y) for lm in landmarks.landmark]


def hand_size(landmarks):
    lm = get_landmark_points(landmarks)
    return np.linalg.norm(np.array(lm[0]) - np.array(lm[9]))


def is_finger_bent(lm, tip_idx, dip_idx, wrist_idx=0):
    tip_dist = finger_distance(lm[tip_idx], lm[wrist_idx])
    dip_dist = finger_distance(lm[dip_idx], lm[wrist_idx])
    return tip_dist < dip_dist * 0.9


GESTURE_MAP = {
    "K": lambda lm, s: (
        lm[16][1] > lm[15][1]
        and lm[20][1] > lm[19][1]
        and lm[8][1] < lm[7][1]
        and lm[12][1] < lm[11][1]
        and lm[4][0] > lm[5][0]
        and lm[4][0] < lm[9][0]
    ),
    "R": lambda lm, s: (
        lm[16][1] > lm[15][1]
        and lm[20][1] > lm[19][1]
        and finger_distance(lm[8], lm[11]) / s < 0.2
        and lm[4][1] < lm[15][1]
        and lm[4][0] > lm[15][0]
    ),
    "O": lambda lm, s: (finger_distance(lm[4], lm[8]) / s < 0.15 and lm[12][1] > lm[11][1]),
    "N": lambda lm, s: (lm[8][1] > lm[7][1] and lm[12][1] > lm[11][1] and lm[4][1] < lm[8][1]),
    "S": lambda lm, s: (
        is_finger_bent(lm, 8, 6)
        and is_finger_bent(lm, 12, 10)
        and is_finger_bent(lm, 16, 14)
        and is_finger_bent(lm, 20, 18)
        and lm[4][1] < min(lm[6][1], lm[10][1], lm[14][1]) + 0.05
        and lm[4][1] > min(lm[7][1], lm[11][1], lm[15][1]) - 0.05
        and abs(lm[4][0] - lm[9][0]) / s < 0.35
    ),
}


def detect_gesture(landmarks):
    lm = get_landmark_points(landmarks)
    s = hand_size(landmarks)
    if s == 0:
        return None
    for letter, check in GESTURE_MAP.items():
        if check(lm, s):
            return letter
    return None


def detect_eye_triangle(hands_results, face_results, frame_shape):
    if not hands_results.multi_hand_landmarks or not face_results.detections:
        return False

    detection = face_results.detections[0]
    bbox = detection.location_data.relative_bounding_box
    eye_y = bbox.y_min + 0.25  # приблизительно уровень глаз

    for hand_landmarks in hands_results.multi_hand_landmarks:
        lm = get_landmark_points(hand_landmarks)
        s = hand_size(hand_landmarks)
        if s < 0.15:
            continue

        thumb_ext = not is_finger_bent(lm, 4, 2)
        index_ext = not is_finger_bent(lm, 8, 6)
        middle_ext = not is_finger_bent(lm, 12, 10)
        ring_bent = is_finger_bent(lm, 16, 14)
        pinky_bent = is_finger_bent(lm, 20, 18)

        tri_dist = (finger_distance(lm[4], lm[8]) + finger_distance(lm[8], lm[12]) + finger_distance(lm[12], lm[4])) / s
        if thumb_ext and index_ext and middle_ext and ring_bent and pinky_bent and tri_dist < 1.2:
            center_y = (lm[4][1] + lm[8][1] + lm[12][1]) / 3
            if abs(center_y - eye_y) < 0.15 and s > 0.2:
                return True
    return False

# ===================== UTILS =====================
def blit_fullscreen(surface, img):
    iw, ih = img.get_size()
    scale = max(WIDTH / iw, HEIGHT / ih)  # заполняем весь экран
    new_w, new_h = int(iw * scale), int(ih * scale)
    img_scaled = pygame.transform.smoothscale(img, (new_w, new_h))
    x = (WIDTH - new_w) // 2
    y = (HEIGHT - new_h) // 2
    surface.blit(img_scaled, (x, y))

def circle_rect_collision(cx, cy, radius, rect):
    """
    Перетин кола (cx, cy, radius) з pygame.Rect (геометрично коректно)
    """
    closest_x = max(rect.left, min(cx, rect.right))
    closest_y = max(rect.top,  min(cy, rect.bottom))
    dx = cx - closest_x
    dy = cy - closest_y
    return (dx * dx + dy * dy) <= (radius * radius)

# ===================== DEBUG DRAW HELPERS =====================
def _draw_hand_landmarks(screen, lm, color=(47, 207, 247)):
    """Малює 21 лендмарк і ребра (спрощено) на pygame screen."""
    pts = []
    for i in range(21):
        x = int(lm[i].x * WIDTH)
        y = int(lm[i].y * HEIGHT)
        pts.append((x, y))
        pygame.draw.circle(screen, color, (x, y), max(2, int(4 * SCALE_Y)))

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # index
        (0, 9), (9, 10), (10, 11), (11, 12),   # middle
        (0, 13), (13, 14), (14, 15), (15, 16), # ring
        (0, 17), (17, 18), (18, 19), (19, 20), # pinky
        (5, 9), (9, 13), (13, 17)              # palm
    ]
    lw = max(1, int(2 * SCALE_Y))
    for a, b in edges:
        pygame.draw.line(screen, color, pts[a], pts[b], lw)

def _draw_hitboxes(screen, shapes, targets):
    """Малює rect-и для предметів (shapes) і цілей (targets)."""
    lw = max(1, int(2 * SCALE_Y))
    for s in shapes:
        pygame.draw.rect(screen, (255, 255, 0), s["rect"], lw)   # shapes: жовті
    for t in targets:
        pygame.draw.rect(screen, (0, 200, 255), t["rect"], lw)   # targets: блакитні

def _draw_grab_debug(screen, px, py, grabbing):
    """Коло взаємодії + індикатор стану."""
    if px == 0 and py == 0:
        return

def draw_hand_cursor(screen):
    """Малює курсор руки для UI екранів (меню, інструкція, level complete, game over)"""
    px, py = hand_cursor["pos"]

    if px == 0 and py == 0:
        return

    color = (47, 207, 247)
    radius = 8

    # Маленький кружечок
    pygame.draw.circle(screen, color, (px, py), radius)

# ===================== HAND CURSOR =====================
def update_hand_cursor(frame, ok):
    global hand_cursor

    if not ok:
        return

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.flip(rgb, 1)

    res = hands.process(rgb)

    hand_cursor["click"] = False

    if not res.multi_hand_landmarks:
        hand_cursor["hovering"] = False
        hand_cursor["grabbing"] = False
        return

    lm = res.multi_hand_landmarks[0].landmark

    x = max(0, min(WIDTH, int(lm[9].x * WIDTH)))
    y = max(0, min(HEIGHT, int(lm[9].y * HEIGHT)))
    hand_cursor["pos"] = (x, y)

    finger_tips = [4, 8, 12, 16, 20]
    finger_pips = [3, 6, 10, 14, 18]
    bent = sum(lm[tip].y > lm[pip].y for tip, pip in zip(finger_tips, finger_pips))
    grabbing = bent >= 4

    if grabbing:
        if not hand_cursor["hovering"]:
            hand_cursor["hovering"] = True
            hand_cursor["hover_time"] = time.time()
        elif time.time() - hand_cursor["hover_time"] >= HOLD_TIME:
            if hand_cursor.get("click_button") == active_button:
                hand_cursor["click"] = True
    else:
        hand_cursor["hovering"] = False

    hand_cursor["grabbing"] = grabbing  # ✅ теперь доступно для экранов

def draw_hold_progress(screen, rect, grabbing):
    global active_button

    if not grabbing:
        # Сброс если нет захвата
        active_button = None
        hand_cursor["click_button"] = None
        return

    if active_button != rect:
        active_button = rect
        hand_cursor["hover_time"] = time.time()
        hand_cursor["click_button"] = rect

    elapsed = time.time() - hand_cursor["hover_time"]
    progress = max(0, min(elapsed / HOLD_TIME, 1))

    bar_w = rect.width
    bar_h = max(6, int(8 * SCALE_Y))
    x = rect.left
    y = rect.bottom + int(6 * SCALE_Y)

    pygame.draw.rect(screen, (30, 30, 30), (x, y, bar_w, bar_h), border_radius=6)
    fill_w = int(bar_w * progress)
    pygame.draw.rect(screen, (47, 207, 247), (x, y, fill_w, bar_h), border_radius=6)
    pygame.draw.rect(screen, (200, 200, 200), (x, y, bar_w, bar_h), 2, border_radius=6)

    if progress >= 1:
        hand_cursor["click"] = True

# ===================== OVERLAY (оновлена версія) =====================
def draw_overlay_fullscreen(state_name=None):
    if state_name is None:
        return  # нічого не малюємо
    overlay_images = {
        "IDLE": "images/idle.png",
        "PAUSED": "images/paused.png",
        "STOPPED": "images/stopped.png"
    }
    path = overlay_images.get(state_name)
    if path:
        img = pygame.image.load(path).convert()
        blit_fullscreen(SCREEN, img)


# ===================== ARROW =====================
ARROW_WIDTH = int(360 * SCALE_Y)
ARROW_HEIGHT = int(30 * SCALE_Y)
arrow_image = pygame.image.load("images/arrows.png").convert_alpha()
arrow_image = pygame.transform.smoothscale(arrow_image, (ARROW_WIDTH, ARROW_HEIGHT))

def get_agent_state():
    with _state_lock:
        return _state["name"]

# ===================== MENU =====================
def menu_screen():
    reset_hold_state()
    bg = pygame.image.load("images/menubg.png").convert()
    start_img = pygame.image.load("images/start_button.png").convert_alpha()
    exit_img = pygame.image.load("images/exit_button.png").convert_alpha()

    start_w, start_h = int(250 * SCALE_Y), int(80 * SCALE_Y)
    start = pygame.transform.smoothscale(start_img, (start_w, start_h))
    exitb = pygame.transform.smoothscale(exit_img, (start_w, start_h))

    BUTTON_OFFSET = 60
    start_rect = start.get_rect(center=(WIDTH // 2, int(HEIGHT // 2 - BUTTON_OFFSET * SCALE_Y)))
    exit_rect = exitb.get_rect(center=(WIDTH // 2, int(HEIGHT // 2 + BUTTON_OFFSET * SCALE_Y)))

    while True:
        handle_events()
        ok, frame = cap.read()
        update_hand_cursor(frame, ok)
        px, py = hand_cursor["pos"]
        grabbing = hand_cursor["grabbing"]
        mouse["pos"] = (px, py)
        mouse["click"] = hand_cursor["click"]
        blit_fullscreen(SCREEN, bg)
        SCREEN.blit(start, start_rect)
        SCREEN.blit(exitb, exit_rect)

        if start_rect.collidepoint(mouse["pos"]):
            arrow_x = start_rect.centerx - ARROW_WIDTH // 2
            arrow_y = start_rect.top + (start_rect.height - ARROW_HEIGHT) // 2
            SCREEN.blit(arrow_image, (arrow_x, arrow_y))
            draw_hold_progress(SCREEN, start_rect, grabbing)

        if exit_rect.collidepoint(mouse["pos"]):
            arrow_x = exit_rect.centerx - ARROW_WIDTH // 2
            arrow_y = exit_rect.top + (exit_rect.height - ARROW_HEIGHT) // 2
            SCREEN.blit(arrow_image, (arrow_x, arrow_y))
            draw_hold_progress(SCREEN, exit_rect, grabbing)

        if not (start_rect.collidepoint(mouse["pos"]) or exit_rect.collidepoint(mouse["pos"])):
            active_button = None
            hand_cursor["click_button"] = None

        draw_hand_cursor(SCREEN)
        draw_camera_preview(SCREEN)
        draw_overlay_fullscreen(get_agent_state())
        pygame.display.flip()

        if mouse["click"]:
            if start_rect.collidepoint(mouse["pos"]):
                return STATE_INSTRUCTIONS
            if exit_rect.collidepoint(mouse["pos"]):
                return STATE_EXIT


# ===================== INSTRUCTIONS =====================
def instructions_screen():
    reset_hold_state()
    bg = pygame.image.load("images/instrbg.png").convert()
    framee = pygame.image.load("images/frame.png").convert_alpha()
    cont_img = pygame.image.load("images/continue_button.png").convert_alpha()
    return_img = pygame.image.load("images/return_button.png").convert_alpha()

    cont_w, cont_h = int(200 * SCALE_Y), int(65 * SCALE_Y)
    cont_btn = pygame.transform.smoothscale(cont_img, (cont_w, cont_h))
    return_btn = pygame.transform.smoothscale(return_img, (cont_w, cont_h))

    spacing = int(100 * SCALE_X)
    total_width = cont_w + spacing + cont_w
    start_x = WIDTH // 2 - total_width // 2
    y_pos = HEIGHT - int(120 * SCALE_Y)

    cont_rect = cont_btn.get_rect(topleft=(start_x, y_pos))
    return_rect = return_btn.get_rect(topleft=(start_x + cont_w + spacing, y_pos))

    font_s = pygame.font.Font(FONT_PATH, int(22 * SCALE_Y))
    font_h = pygame.font.Font(FONT_PATH, int(36 * SCALE_Y))
    color = (47, 207, 247)

    lines = [
        "1. Збирай слово KRONOS жестами руки (по одній літері).",
        "2. Утримуй правильний жест 2 секунди, щоб зафіксувати літеру.",
        "3. Після слова зроби фінальний знак: 'Око в трикутнику'.",
        "4. Утримуй фінальний знак 3 секунди.",
        "5. Ліміт часу: 5 хвилин."
    ]

    arrow_scale_factor = 0.8
    arrow_small = pygame.transform.smoothscale(
        arrow_image,
        (int(ARROW_WIDTH * arrow_scale_factor), int(ARROW_HEIGHT * arrow_scale_factor))
    )

    # визначимо рамку для тексту
    frame_margin_x = int(40 * SCALE_X)
    frame_margin_y = int(30 * SCALE_Y)
    frame_w = WIDTH - 2 * frame_margin_x
    frame_h = int(240 * SCALE_Y)  # висота рамки під текст
    frame_rect = pygame.Rect(frame_margin_x, int(100 * SCALE_Y), frame_w, frame_h)
    frame_surf = pygame.transform.smoothscale(framee, (frame_w, frame_h))

    while True:
        handle_events()
        ok, frame = cap.read()
        update_hand_cursor(frame, ok)  # Використання кадру тільки для рука-контролю

        px, py = hand_cursor["pos"]
        grabbing = hand_cursor["grabbing"]
        mouse["pos"] = (px, py)
        mouse["click"] = hand_cursor["click"]

        blit_fullscreen(SCREEN, bg)

        # слово "ІНСТРУКЦІЯ ДО ГРИ" окремо
        title = font_h.render("ІНСТРУКЦІЯ ДО ГРИ", True, color)
        SCREEN.blit(title, (WIDTH // 2 - title.get_width() // 2, int(30 * SCALE_Y)))

        # малюємо рамку
        SCREEN.blit(frame_surf, frame_rect)

        # малюємо текст по середині рамки
        y_text = frame_rect.top + int(20 * SCALE_Y)
        for l in lines:
            text_surf = font_s.render(l, True, color)
            x_text = frame_rect.left + (frame_rect.width - text_surf.get_width()) // 2
            SCREEN.blit(text_surf, (x_text, y_text))
            y_text += int(40 * SCALE_Y)

        SCREEN.blit(cont_btn, cont_rect)
        SCREEN.blit(return_btn, return_rect)

        if cont_rect.collidepoint(mouse["pos"]):
            arrow_x = cont_rect.centerx - arrow_small.get_width() // 2
            arrow_y = cont_rect.top + (cont_rect.height - arrow_small.get_height()) // 2
            SCREEN.blit(arrow_small, (arrow_x, arrow_y))
            draw_hold_progress(SCREEN, cont_rect, grabbing)

        if return_rect.collidepoint(mouse["pos"]):
            arrow_x = return_rect.centerx - arrow_small.get_width() // 2
            arrow_y = return_rect.top + (return_rect.height - arrow_small.get_height()) // 2
            SCREEN.blit(arrow_small, (arrow_x, arrow_y))
            draw_hold_progress(SCREEN, return_rect, grabbing)

        if not (cont_rect.collidepoint(mouse["pos"]) or return_rect.collidepoint(mouse["pos"])):
            active_button = None
            hand_cursor["click_button"] = None

        draw_hand_cursor(SCREEN)
        draw_camera_preview(SCREEN)
        draw_overlay_fullscreen(get_agent_state())
        pygame.display.flip()

        if mouse["click"]:
            if cont_rect.collidepoint(mouse["pos"]):
                return STATE_GAME
            if return_rect.collidepoint(mouse["pos"]):
                return STATE_MENU
# ===================== LEVEL COMPLETE =====================
def level_complete_screen(level_index):
    # Для KRONOS-режима этот экран не используется (сценарий один, без уровней).
    return "return"


# ===================== KRONOS GAME =====================
def run_game():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    frame_path = os.path.join(script_dir, "images", "frame.png")
    frame_png = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)

    overall_start = time.time()
    current_letter_idx = 0
    hold_start = None
    final_hold_start = None
    word_completed = False
    final_completed = False

    margin = 50
    clock = pygame.time.Clock()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        if w != 1280 or h != 720:
            frame = cv2.resize(frame, (1280, 720))
            h, w = 720, 1280

        # Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return STATE_MENU

        time_elapsed = time.time() - overall_start
        time_left = max(0.0, OVERALL_TIME_LIMIT - time_elapsed)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_results = hands_game.process(rgb)
        face_results = face_detection.process(rgb)

        detected_letter = None
        if hands_results.multi_hand_landmarks:
            for hl in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hl, mp_hands_game.HAND_CONNECTIONS)

        if hands_results.multi_hand_landmarks and not word_completed:
            detected_letter = detect_gesture(hands_results.multi_hand_landmarks[0])

            if current_letter_idx < len(TARGET_WORD):
                expected = TARGET_WORD[current_letter_idx]
                if detected_letter == expected:
                    if hold_start is None:
                        hold_start = time.time()
                    if time.time() - hold_start >= LETTER_HOLD_TIME:
                        current_letter_idx += 1
                        hold_start = None
                        if current_letter_idx == len(TARGET_WORD):
                            word_completed = True
                else:
                    hold_start = None

        # Финальный жест
        if word_completed and not final_completed:
            if detect_eye_triangle(hands_results, face_results, frame.shape):
                if final_hold_start is None:
                    final_hold_start = time.time()
                if time.time() - final_hold_start >= FINAL_HOLD_TIME:
                    final_completed = True
            else:
                final_hold_start = None

        # --- UI / overlay ---
        if frame_png is not None:
            resized = cv2.resize(frame_png, (w - 2 * margin, h - 2 * margin))
            frame = overlay_png(frame, resized, margin, margin)

        mins, secs = divmod(int(time_left), 60)
        frame = draw_text_pil(frame, f"TIME: {mins:02d}:{secs:02d}", (margin + 80, margin + 40), 30, COLOR_WHITE)

        if not word_completed:
            hint_text = f"Next gesture: {TARGET_WORD[current_letter_idx]}"
            frame = draw_text_pil(frame, hint_text, (margin + 80, margin + 80), 25, COLOR_ACCENT)

            display_word = ""
            for i, char in enumerate(TARGET_WORD):
                display_word += (char + " ") if i < current_letter_idx else "_ "
            frame = draw_text_pil(frame, display_word, (w - margin - 360, margin + 40), 40, COLOR_WHITE)

            bx, by = margin + 80, h - margin - 60
            total_progress = current_letter_idx / len(TARGET_WORD)
            cv2.rectangle(frame, (bx, by), (bx + 300, by + 15), (30, 30, 50), -1)
            cv2.rectangle(frame, (bx, by), (bx + int(300 * total_progress), by + 15), COLOR_ACCENT, -1)
            frame = draw_text_pil(frame, "Word Progress", (bx, by - 30), 20, COLOR_WHITE)

            if hold_start:
                hold_ratio = min(1.0, (time.time() - hold_start) / LETTER_HOLD_TIME)
                cv2.rectangle(frame, (bx, by + 25), (bx + int(300 * hold_ratio), by + 30), (0, 255, 255), -1)

        elif not final_completed:
            frame = draw_text_pil(frame, "ACCESS GRANTED", (w // 2 - 200, h // 2 - 100), 50, COLOR_ACCENT)
            frame = draw_text_pil(frame, "Final emblem: Eye in Triangle", (w // 2 - 280, h // 2 - 20), 35, COLOR_FINAL)

            if final_hold_start:
                hold_ratio = min(1.0, (time.time() - final_hold_start) / FINAL_HOLD_TIME)
                cv2.rectangle(frame, (w // 2 - 150, h // 2 + 60), (w // 2 + 150, h // 2 + 80), (30, 30, 50), -1)
                cv2.rectangle(frame, (w // 2 - 150, h // 2 + 60), (w // 2 - 150 + int(300 * hold_ratio), h // 2 + 80), COLOR_FINAL, -1)

        else:
            frame = draw_text_pil(frame, "TRUE ACCESS GRANTED", (w // 2 - 320, h // 2 - 50), 60, (0, 255, 100))
            frame = draw_text_pil(frame, "Welcome, Illuminated One", (w // 2 - 260, h // 2 + 20), 40, COLOR_FINAL)

        # --- Convert to pygame surface ---
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = np.fliplr(frame_rgb)
        frame_rgb = np.rot90(frame_rgb)

        frame_surf = pygame.surfarray.make_surface(frame_rgb)
        frame_surf = pygame.transform.smoothscale(frame_surf, (WIDTH, HEIGHT))
        SCREEN.blit(frame_surf, (0, 0))

        draw_overlay_fullscreen(get_agent_state())
        pygame.display.flip()
        clock.tick(30)

        if time_left <= 0:
            return STATE_GAME_OVER

        if final_completed:
            pygame.time.wait(2500)
            return STATE_MENU

    return STATE_GAME_OVER

# (удалено) В файле больше не должно быть второго независимого OpenCV-`main()`.
# ===================== GAME OVER =====================
def game_over():
    reset_hold_state()
    bg = pygame.image.load("images/gameover.png").convert()
    return_img = pygame.image.load("images/return_button.png").convert_alpha()
    return_btn = pygame.transform.smoothscale(return_img, (int(250 * SCALE_Y), int(80 * SCALE_Y)))
    return_rect = return_btn.get_rect(center=(WIDTH // 2, HEIGHT // 2 + int(60 * SCALE_Y)))

    while True:
        handle_events()
        ok, frame = cap.read()
        update_hand_cursor(frame, ok)
        px, py = hand_cursor["pos"]
        grabbing = hand_cursor["grabbing"]
        mouse["pos"] = (px, py)
        mouse["click"] = hand_cursor["click"]

        blit_fullscreen(SCREEN, bg)
        SCREEN.blit(return_btn, return_rect)

        if return_rect.collidepoint(mouse["pos"]):
            arrow_x = return_rect.centerx - ARROW_WIDTH // 2
            arrow_y = return_rect.top + (return_rect.height - ARROW_HEIGHT) // 2
            SCREEN.blit(arrow_image, (arrow_x, arrow_y))
            draw_hold_progress(SCREEN, return_rect, grabbing)

        if not return_rect.collidepoint(mouse["pos"]):
            active_button = None
            hand_cursor["click_button"] = None

        draw_hand_cursor(SCREEN)
        draw_camera_preview(SCREEN)
        draw_overlay_fullscreen(get_agent_state())
        pygame.display.flip()

        if mouse["click"] and return_rect.collidepoint(mouse["pos"]):
            return STATE_MENU

# ===================== MAIN LOOP =====================
while state != STATE_EXIT:
    if state == STATE_MENU:
        state = menu_screen()
    elif state == STATE_INSTRUCTIONS:
        state = instructions_screen()
    elif state == STATE_GAME:
        result = run_game()
        if result == "VICTORY":
            state = STATE_MENU
        else:
            state = result
    elif state == STATE_GAME_OVER:
        state = game_over()

cap.release()
pygame.quit()

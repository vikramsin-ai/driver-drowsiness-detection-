"""
Driver Drowsiness Detection System
===================================
Uses webcam + MediaPipe Face Mesh to detect:
  - Eye Aspect Ratio (EAR) → drowsy if eyes closed too long
  - Mouth Aspect Ratio (MAR) → yawning detection
  - Head pose tilt → nodding off detection

Author : Your Name (College Minor Project)
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import pygame
import os
import sys

# ──────────────────────────────────────────────
# CONFIGURATION  (tweak these to your liking)
# ──────────────────────────────────────────────
EAR_THRESHOLD      = 0.25   # below this → eye considered closed
EAR_CONSEC_FRAMES  = 20     # frames eye must be closed to trigger alert
MAR_THRESHOLD      = 0.6    # above this → yawn detected
HEAD_TILT_THRESHOLD= 25     # degrees nose-tip tilt triggers alert
ALARM_SOUND        = "alarm.wav"   # place alarm.wav next to this script

# MediaPipe landmark indices for left & right eye
# (from the 468-point face mesh)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158,  133, 153, 144]

# Mouth landmarks (upper-lower pairs + corners)
MOUTH_TOP    = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT   = 78
MOUTH_RIGHT  = 308

# ──────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────

def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    """Compute EAR for one eye given 6 landmark indices."""
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((lm.x * img_w, lm.y * img_h))

    # Vertical distances
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    # Horizontal distance
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))

    ear = (A + B) / (2.0 * C) if C > 0 else 0
    return ear


def mouth_aspect_ratio(landmarks, img_w, img_h):
    """Compute MAR (mouth openness ratio)."""
    top    = landmarks[MOUTH_TOP]
    bottom = landmarks[MOUTH_BOTTOM]
    left   = landmarks[MOUTH_LEFT]
    right  = landmarks[MOUTH_RIGHT]

    vert  = np.linalg.norm(
        np.array([top.x * img_w,    top.y * img_h]) -
        np.array([bottom.x * img_w, bottom.y * img_h])
    )
    horiz = np.linalg.norm(
        np.array([left.x * img_w,  left.y * img_h]) -
        np.array([right.x * img_w, right.y * img_h])
    )
    mar = vert / horiz if horiz > 0 else 0
    return mar


def get_head_tilt(landmarks, img_w, img_h):
    """Return approximate head roll angle (degrees)."""
    left_eye_corner  = landmarks[33]
    right_eye_corner = landmarks[263]

    dx = (right_eye_corner.x - left_eye_corner.x) * img_w
    dy = (right_eye_corner.y - left_eye_corner.y) * img_h

    angle = np.degrees(np.arctan2(dy, dx))
    return angle


def play_alarm():
    """Play alarm sound (non-blocking)."""
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        if os.path.exists(ALARM_SOUND):
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.load(ALARM_SOUND)
                pygame.mixer.music.play(-1)   # loop
        else:
            # Fallback: system beep
            print("\a", end="", flush=True)
    except Exception:
        print("\a", end="", flush=True)


def stop_alarm():
    """Stop alarm sound."""
    try:
        if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
    except Exception:
        pass


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    # Init pygame mixer for alarm
    try:
        pygame.mixer.init()
    except Exception:
        pass

    # MediaPipe face mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check camera index.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # State variables
    closed_frame_count = 0
    yawn_count         = 0
    alarm_on           = False
    alert_text         = ""
    alert_color        = (0, 255, 0)
    fps_time           = time.time()
    frame_count        = 0
    fps                = 0

    print("=" * 50)
    print("  Driver Drowsiness Detection System RUNNING")
    print("  Press  Q  to quit")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame capture failed.")
            break

        frame = cv2.flip(frame, 1)   # mirror view
        h, w = frame.shape[:2]

        # FPS counter
        frame_count += 1
        elapsed = time.time() - fps_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            fps_time   = time.time()
            frame_count = 0

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        ear_avg  = 0.0
        mar_val  = 0.0
        tilt_val = 0.0
        face_detected = False

        if results.multi_face_landmarks:
            face_detected = True
            lms = results.multi_face_landmarks[0].landmark

            # ── EAR ──────────────────────────────
            ear_left  = eye_aspect_ratio(lms, LEFT_EYE,  w, h)
            ear_right = eye_aspect_ratio(lms, RIGHT_EYE, w, h)
            ear_avg   = (ear_left + ear_right) / 2.0

            # ── MAR ──────────────────────────────
            mar_val = mouth_aspect_ratio(lms, w, h)

            # ── HEAD TILT ─────────────────────────
            tilt_val = get_head_tilt(lms, w, h)

            # ── DRAW eye contours ─────────────────
            for idx in LEFT_EYE + RIGHT_EYE:
                lm = lms[idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 2, (0, 255, 255), -1)

            # ── DROWSINESS LOGIC ──────────────────
            drowsy   = False
            yawning  = False
            tilted   = False

            if ear_avg < EAR_THRESHOLD:
                closed_frame_count += 1
                if closed_frame_count >= EAR_CONSEC_FRAMES:
                    drowsy = True
            else:
                closed_frame_count = 0

            if mar_val > MAR_THRESHOLD:
                yawning = True
                yawn_count += 1  # will over-count per frame; fine for demo

            if abs(tilt_val) > HEAD_TILT_THRESHOLD:
                tilted = True

            # Determine alert
            if drowsy:
                alert_text  = "⚠  DROWSY! WAKE UP!"
                alert_color = (0, 0, 255)
                if not alarm_on:
                    play_alarm()
                    alarm_on = True
            elif yawning:
                alert_text  = "😮  YAWNING DETECTED"
                alert_color = (0, 165, 255)
                stop_alarm(); alarm_on = False
            elif tilted:
                alert_text  = "↙  HEAD TILTING!"
                alert_color = (0, 200, 255)
                stop_alarm(); alarm_on = False
            else:
                alert_text  = "✔  ALERT & FOCUSED"
                alert_color = (0, 220, 0)
                stop_alarm(); alarm_on = False

        else:
            alert_text  = "No Face Detected"
            alert_color = (150, 150, 150)
            stop_alarm(); alarm_on = False
            closed_frame_count = 0

        # ── OVERLAY UI ────────────────────────────────────────────────

        # Dark header bar
        cv2.rectangle(frame, (0, 0), (w, 50), (20, 20, 20), -1)
        cv2.putText(frame, "DRIVER DROWSINESS DETECTION",
                    (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 215, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (w - 100, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 200), 1)

        # Alert banner (bottom)
        cv2.rectangle(frame, (0, h - 55), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, alert_text,
                    (15, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    alert_color, 2)

        # Stats panel (right side)
        panel_x = w - 200
        cv2.rectangle(frame, (panel_x - 10, 55), (w, 200), (20, 20, 20, 180), -1)

        stats = [
            (f"EAR  : {ear_avg:.2f}", (0, 255, 255)),
            (f"MAR  : {mar_val:.2f}", (255, 200, 0)),
            (f"TILT : {tilt_val:.1f}°", (200, 200, 255)),
            (f"CLOSED: {closed_frame_count}", (255, 100, 100)),
        ]
        for i, (txt, col) in enumerate(stats):
            cv2.putText(frame, txt,
                        (panel_x, 80 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)

        # Threshold indicators
        cv2.putText(frame, f"[EAR thr={EAR_THRESHOLD}]",
                    (panel_x, 80 + 4 * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # Progress bar for EAR
        bar_max = 180
        bar_val = int(min(ear_avg / 0.4, 1.0) * bar_max)
        bar_col = (0, 255, 0) if ear_avg > EAR_THRESHOLD else (0, 0, 255)
        cv2.rectangle(frame, (panel_x, 215), (panel_x + bar_max, 230), (60, 60, 60), -1)
        cv2.rectangle(frame, (panel_x, 215), (panel_x + bar_val, 230), bar_col, -1)
        cv2.putText(frame, "EYE OPEN", (panel_x, 245),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # Flashing red border when alarm
        if alarm_on and (int(time.time() * 4) % 2 == 0):
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)

        cv2.imshow("Driver Drowsiness Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    # Cleanup
    stop_alarm()
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("[INFO] Detection stopped. Goodbye!")


if __name__ == "__main__":
    main()

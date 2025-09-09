"""
Ejemplo: Contador de repeticiones y descansos con MediaPipe BlazePose + LOG a CSV

Características:
- Detección de pose en tiempo real con webcam (plano completo sugerido).
- Menú en pantalla y por teclado para seleccionar ejercicio (Sentadillas, Flexiones, Curl de bíceps).
- Conteo de repeticiones por umbrales de ángulos articulares.
- Detección de "fase" (bajada/subida) para validar repeticiones completas.
- Objetivo de repeticiones por serie: al alcanzarlo, inicia temporizador de descanso.
- HUD con FPS, ejercicio activo, repeticiones, objetivo, tiempo de descanso restante.
- **Nuevo**: Registro automático de **cada serie** en `sessions.csv` (fecha, ejercicio, set, reps, objetivo, duración, descanso).

Dependencias:
    pip install mediapipe opencv-python numpy

Usar:
    python webcam_blazepose_ejercicios.py

Atajos de teclado durante la sesión:
    1 -> Sentadillas
    2 -> Flexiones
    3 -> Curl bíceps
    +/- -> Subir / bajar objetivo de repeticiones de la serie
    r -> Reset del contador de repeticiones (misma serie, sin loguear)
    n -> Nueva serie (si hay reps, se registra la serie actual y se inicia otra)
    f -> Toggle pantalla completa
    q/ESC -> Salir (se registra la serie en curso si tiene reps)

Notas:
- Para flexiones se asume cámara lateral (perfil) si es posible; si no, puede fallar más.
- Ajuste fino de umbrales disponible en el diccionario EXERCISES_THRESHOLDS.
"""
from __future__ import annotations
import cv2
import numpy as np
import time
import csv
import os
from dataclasses import dataclass

try:
    import mediapipe as mp
except ImportError:
    raise SystemExit("Falta mediapipe. Instalá con: pip install mediapipe")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ---------------------------------------------
# Utilidades geométricas
# ---------------------------------------------

def angle_3pts(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Ángulo ABC en grados (b es el vértice)."""
    ba = a - b
    bc = c - b
    # Evitar división por cero
    if np.linalg.norm(ba) < 1e-6 or np.linalg.norm(bc) < 1e-6:
        return 0.0
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

@dataclass
class ExerciseConfig:
    name: str
    # puntos de referencia (landmarks) indices según MediaPipe
    # https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
    # Usamos el modelo BlazePose Full Body con 33 landmarks.

    # función que dada la lista de puntos devuelve el ángulo a controlar
    angle_fn: callable
    # Umbrales de ángulo para fase inferior/superior (deg)
    down_threshold: float
    up_threshold: float
    # Texto de ayuda
    hint: str

class RepCounter:
    def __init__(self, target_reps: int = 10):
        self.count = 0
        self.direction = 0  # 0 esperando bajar, 1 esperando subir
        self.target = max(1, int(target_reps))
        self.resting = False
        self.rest_end_ts = 0.0
        self.rest_seconds = 60

    def set_target(self, t: int):
        self.target = max(1, int(t))

    def reset(self):
        self.count = 0
        self.direction = 0

    def new_set(self):
        self.reset()
        self.resting = False
        self.rest_end_ts = 0.0

    def start_rest(self, seconds: int):
        self.rest_seconds = max(5, int(seconds))
        self.rest_end_ts = time.time() + self.rest_seconds
        self.resting = True

    def rest_remaining(self) -> int:
        if not self.resting:
            return 0
        return max(0, int(self.rest_end_ts - time.time()))

    def maybe_finish_rest(self):
        if self.resting and time.time() >= self.rest_end_ts:
            self.resting = False

# ---------------------------------------------
# Logger de sesiones a CSV
# ---------------------------------------------

class SessionLogger:
    def __init__(self, csv_path: str = "sessions.csv"):
        self.csv_path = csv_path
        self._ensure_header()

    def _ensure_header(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "date_iso", "exercise", "set_index", "reps", "target_reps",
                    "set_duration_sec", "rest_planned_sec"
                ])

    def log_set(self, exercise: str, set_index: int, reps: int, target: int,
                started_ts: float, ended_ts: float, rest_planned: int):
        duration = max(0.0, ended_ts - started_ts)
        date_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                date_iso, exercise, set_index, reps, target,
                round(duration, 2), rest_planned
            ])

# ---------------------------------------------
# Definiciones de ejercicios (ángulos y thresholds)
# ---------------------------------------------

LM = mp_pose.PoseLandmark


def squat_angle(landmarks: np.ndarray) -> float:
    """Ángulo de la rodilla (derecha) usando cadera-rodilla-tobillo."""
    hip = landmarks[LM.RIGHT_HIP.value]
    knee = landmarks[LM.RIGHT_KNEE.value]
    ankle = landmarks[LM.RIGHT_ANKLE.value]
    return angle_3pts(hip, knee, ankle)


def pushup_angle(landmarks: np.ndarray) -> float:
    """Ángulo del codo (derecho) usando hombro-codo-muñeca."""
    shoulder = landmarks[LM.RIGHT_SHOULDER.value]
    elbow = landmarks[LM.RIGHT_ELBOW.value]
    wrist = landmarks[LM.RIGHT_WRIST.value]
    return angle_3pts(shoulder, elbow, wrist)


def curl_angle(landmarks: np.ndarray) -> float:
    """Ángulo del codo (derecho) usando hombro-codo-muñeca para curl bíceps."""
    shoulder = landmarks[LM.RIGHT_SHOULDER.value]
    elbow = landmarks[LM.RIGHT_ELBOW.value]
    wrist = landmarks[LM.RIGHT_WRIST.value]
    return angle_3pts(shoulder, elbow, wrist)


EXERCISES_THRESHOLDS = {
    "Sentadillas": ExerciseConfig(
        name="Sentadillas",
        angle_fn=squat_angle,
        # Bajando: rodilla se cierra (< 90-100°). Subiendo: se estira (> 160°)
        down_threshold=100.0,
        up_threshold=160.0,
        hint="Bajá < 100°, subí > 160°",
    ),
    "Flexiones": ExerciseConfig(
        name="Flexiones",
        angle_fn=pushup_angle,
        # Abajo: codo < 70-80°, Arriba: codo > 160°
        down_threshold=80.0,
        up_threshold=160.0,
        hint="Bajá < 80°, subí > 160°",
    ),
    "Curl bíceps": ExerciseConfig(
        name="Curl bíceps",
        angle_fn=curl_angle,
        # Arriba: codo < 50-60°, Abajo: codo > 150°
        down_threshold=60.0,
        up_threshold=150.0,
        hint="Flexioná < 60°, extendé > 150°",
    ),
}

EXERCISE_KEYS = ["Sentadillas", "Flexiones", "Curl bíceps"]

# ---------------------------------------------
# Dibujo y HUD
# ---------------------------------------------

def draw_hud(frame, exercise: ExerciseConfig, reps: RepCounter, angle: float | None, fps: float):
    h, w = frame.shape[:2]
    pad = 12
    # Caja superior
    cv2.rectangle(frame, (pad, pad), (w - pad, 130), (0, 0, 0), -1)
    cv2.putText(frame, f"Ejercicio: {exercise.name}", (pad + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"Reps: {reps.count}/{reps.target}", (pad + 10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"Hint: {exercise.hint}", (pad + 10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # FPS y ángulo
    txt = f"FPS: {fps:.1f}"
    if angle is not None:
        txt += f" | Ángulo: {int(angle)}°"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (w - tw - pad - 10, pad), (w - pad, pad + th + 14), (0, 0, 0), -1)
    cv2.putText(frame, txt, (w - tw - pad - 5, pad + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Descanso
    if reps.resting:
        remaining = reps.rest_remaining()
        bar_w = int((w - 2 * pad) * (1 - remaining / max(1, reps.rest_seconds)))
        cv2.rectangle(frame, (pad, h - 50), (pad + bar_w, h - 20), (0, 165, 255), -1)
        cv2.rectangle(frame, (pad, h - 50), (w - pad, h - 20), (0, 0, 0), 2)
        cv2.putText(frame, f"Descanso: {remaining}s", (pad + 10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Ayuda de atajos
    shortcuts = "[1] Sentadillas  [2] Flexiones  [3] Curl  [+/-] Objetivo  [r] Reset  [n] Nueva serie  [f] Full  [q] Salir"
    (tw, th), _ = cv2.getTextSize(shortcuts, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (pad, h - 80), (min(w - pad, pad + tw + 20), h - 55), (0, 0, 0), -1)
    cv2.putText(frame, shortcuts, (pad + 10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def draw_landmarks(frame, results):
    mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
    )

# ---------------------------------------------
# Lógica de conteo de repeticiones
# ---------------------------------------------

def update_reps(exercise: ExerciseConfig, reps: RepCounter, angle: float) -> bool:
    """Actualiza el contador y devuelve True si se completó una serie (alcanzó objetivo)."""
    if reps.resting:
        return False

    completed = False
    # Dirección 0 -> esperando ir hacia abajo (cerrar/superar umbral de contracción)
    if reps.direction == 0:
        if angle < exercise.down_threshold:
            reps.direction = 1  # ahora esperamos volver a subir
    else:
        # Dirección 1 -> esperando volver arriba (extensión)
        if angle > exercise.up_threshold:
            reps.count += 1
            reps.direction = 0
            if reps.count >= reps.target:
                completed = True
                reps.start_rest(reps.rest_seconds)
    return completed

# ---------------------------------------------
# Main
# ---------------------------------------------

def main():
    # Config inicial
    current_exercise = EXERCISE_KEYS[0]
    config = EXERCISES_THRESHOLDS[current_exercise]
    reps = RepCounter(target_reps=10)
    reps.rest_seconds = 45  # descanso por defecto

    logger = SessionLogger(csv_path="sessions.csv")
    set_index = 1
    set_start_ts = time.time()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la cámara (índice 0). Probá con otro índice.")

    # Intentar resolución más alta para plano completo
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fullscreen = False
    prev_ts = time.time()
    fps = 0.0
    prev_resting = reps.resting

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # espejo
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = pose.process(rgb)
            angle_val = None

            if results.pose_landmarks:
                # Convertir landmarks a np.array en coordenadas de imagen
                lm = results.pose_landmarks.landmark
                h, w = frame.shape[:2]
                pts = np.array([[p.x * w, p.y * h, p.z] for p in lm], dtype=np.float32)

                # Calcular ángulo para el ejercicio activo
                try:
                    angle_val = config.angle_fn(pts)
                except Exception:
                    angle_val = None

                # Conteo
                if angle_val is not None:
                    completed = update_reps(config, reps, angle_val)
                    if completed:
                        # Log de la serie completada
                        logger.log_set(
                            exercise=config.name,
                            set_index=set_index,
                            reps=reps.count,
                            target=reps.target,
                            started_ts=set_start_ts,
                            ended_ts=time.time(),
                            rest_planned=reps.rest_seconds,
                        )
                        set_index += 1

                # Dibujo landmarks
                draw_landmarks(frame, results)

            # FPS
            now = time.time()
            dt = now - prev_ts
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            prev_ts = now

            # Actualizar descanso y detectar fin de descanso para timestamp de nueva serie
            reps.maybe_finish_rest()
            if prev_resting and not reps.resting:
                set_start_ts = time.time()  # arranca nueva serie
            prev_resting = reps.resting

            # HUD
            draw_hud(frame, config, reps, angle_val, fps)

            cv2.imshow("BlazePose - Ejercicios", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):  # ESC o q
                # Al salir, si hay reps parciales, registrar la serie en curso
                if reps.count > 0:
                    logger.log_set(
                        exercise=config.name,
                        set_index=set_index,
                        reps=reps.count,
                        target=reps.target,
                        started_ts=set_start_ts,
                        ended_ts=time.time(),
                        rest_planned=reps.rest_seconds,
                    )
                break
            elif key == ord('1'):
                current_exercise = EXERCISE_KEYS[0]
            elif key == ord('2'):
                current_exercise = EXERCISE_KEYS[1]
            elif key == ord('3'):
                current_exercise = EXERCISE_KEYS[2]
            elif key == ord('+') or key == ord('='):
                reps.set_target(reps.target + 1)
            elif key == ord('-') or key == ord('_'):
                reps.set_target(max(1, reps.target - 1))
            elif key == ord('r'):
                reps.reset()
                set_start_ts = time.time()
            elif key == ord('n'):
                # Si hay reps hechas, registrar la serie actual y comenzar otra
                if reps.count > 0:
                    logger.log_set(
                        exercise=config.name,
                        set_index=set_index,
                        reps=reps.count,
                        target=reps.target,
                        started_ts=set_start_ts,
                        ended_ts=time.time(),
                        rest_planned=reps.rest_seconds,
                    )
                    set_index += 1
                reps.new_set()
                set_start_ts = time.time()
            elif key == ord('f'):
                fullscreen = not fullscreen
                cv2.namedWindow("BlazePose - Ejercicios", cv2.WINDOW_NORMAL)
                if fullscreen:
                    cv2.setWindowProperty("BlazePose - Ejercicios", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty("BlazePose - Ejercicios", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

            # Actualizar config si cambió ejercicio
            config = EXERCISES_THRESHOLDS[current_exercise]

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

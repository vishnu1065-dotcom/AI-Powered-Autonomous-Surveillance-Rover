import cv2
import face_recognition
import numpy as np
import os
import time
import threading
import urllib.request
import urllib.parse
import json

from datetime import datetime

from flask import (
    Flask,
    Response,
    render_template_string,
    send_from_directory,
    jsonify
)

# ============================================================
# SETTINGS
# ============================================================

CAMERA_INDEX = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

KNOWN_DIR = "known_faces"
UNKNOWN_DIR = "unknown_faces"

YOLO_CFG = "models/yolov4-tiny.cfg"
YOLO_WEIGHTS = "models/yolov4-tiny.weights"
YOLO_NAMES = "models/coco.names"

# ------------------------------------------------------------
# ESP32 SETTINGS
# ------------------------------------------------------------

# CHANGE THIS TO YOUR ESP32 IP ADDRESS
ESP32_IP = "192.168.18.252"

ESP32_PORT = 80

# Command timeout (rover may be finishing a servo/turn step)
ESP32_TIMEOUT = 3.0

# Health-check timeout
ESP32_CHECK_TIMEOUT = 2.0

# Consecutive failed checks before showing OFFLINE
ESP32_FAIL_LIMIT = 2

# ------------------------------------------------------------
# FACE / YOLO
# ------------------------------------------------------------

FACE_TOLERANCE = 0.60

YOLO_CONFIDENCE = 0.40
YOLO_NMS = 0.40

# Run detection every N frames
DETECT_EVERY = 3

# Same unknown person will not be saved again for 180 seconds
UNKNOWN_COOLDOWN = 180


# ============================================================
# FOLDERS
# ============================================================

os.makedirs(KNOWN_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# GLOBAL VARIABLES
# ============================================================

known_encodings = []
known_names = []

unknown_history = []

unknown_count = 0

frame_number = 0

# Person tracking
tracked_people = {}
next_person_id = 1

# Current rover mode
rover_mode = "AUTO"

# Last command
last_command = "STOP"

# ESP32 status
esp32_online = False

# Current camera frame
latest_frame = None

# JPEG frame for web
latest_jpeg = None

# Locks
frame_lock = threading.Lock()
tracking_lock = threading.Lock()
esp32_lock = threading.Lock()

# Stop camera thread
camera_running = True


# ============================================================
# ESP32 COMMUNICATION - ROBUST VERSION
# ============================================================

import socket

# A single lock prevents two Flask requests from talking to the
# ESP32 at exactly the same time.
esp32_command_lock = threading.Lock()


def send_esp32_command(command):
    """
    Send an HTTP command to the ESP32.

    Expected ESP32 endpoint:
        GET /command?cmd=FORWARD
        GET /command?cmd=BACKWARD
        GET /command?cmd=LEFT
        GET /command?cmd=RIGHT
        GET /command?cmd=STOP
        GET /command?cmd=MANUAL
        GET /command?cmd=AUTO

    Returns:
        (True, response_text) on success
        (False, error_text) on failure
    """
    global esp32_online
    global esp32_fail_count
    global last_command

    command = str(command).strip().upper()

    allowed = {
        "FORWARD",
        "BACKWARD",
        "LEFT",
        "RIGHT",
        "STOP",
        "AUTO",
        "MANUAL"
    }

    if command not in allowed:
        return False, "Invalid ESP32 command"

    try:
        params = urllib.parse.urlencode({"cmd": command})

        url = (
            f"http://{ESP32_IP}:{ESP32_PORT}"
            f"/command?{params}"
        )

        with esp32_command_lock:
            with urllib.request.urlopen(
                url,
                timeout=ESP32_TIMEOUT
            ) as response:

                status_code = getattr(response, "status", 200)
                result = response.read().decode(
                    "utf-8",
                    errors="ignore"
                ).strip()

        if status_code < 200 or status_code >= 300:
            raise RuntimeError(
                f"ESP32 HTTP status {status_code}"
            )

        esp32_online = True
        esp32_fail_count = 0
        last_command = command

        print(
            f"[ESP32] COMMAND={command} RESPONSE={result}"
        )

        return True, result or "OK"

    except Exception as e:
        esp32_online = False

        print(
            f"[ESP32] COMMAND ERROR ({command}): {e}"
        )

        return False, str(e)


esp32_fail_count = 0


def check_esp32():
    """
    Check whether the ESP32 is online by requesting  GET /
    The ESP32 firmware replies with the text  ESP32_OK.

    The dashboard only goes OFFLINE after ESP32_FAIL_LIMIT
    consecutive failures, so one slow reply does not flicker
    the status.
    """
    global esp32_online
    global esp32_fail_count

    try:
        url = f"http://{ESP32_IP}:{ESP32_PORT}/"

        with urllib.request.urlopen(
            url,
            timeout=ESP32_CHECK_TIMEOUT
        ) as response:

            body = response.read().decode(
                "utf-8",
                errors="ignore"
            ).strip()

        if "ESP32_OK" not in body:
            raise RuntimeError(f"Unexpected reply: {body}")

        esp32_fail_count = 0
        esp32_online = True
        return True

    except Exception as e:
        esp32_fail_count += 1

        if esp32_fail_count >= ESP32_FAIL_LIMIT:
            esp32_online = False

        print(f"[ESP32] CHECK FAILED ({esp32_fail_count}): {e}")
        return False


def esp32_monitor():
    """
    Background ESP32 connection monitor.

    When the ESP32 comes back online (for example after a reboot,
    when it always starts in AUTO), the Raspberry Pi re-sends the
    mode it believes the rover is in, so both stay in sync.
    """
    was_online = False

    while camera_running:

        check_esp32()

        if esp32_online and not was_online:
            try:
                if rover_mode == "MANUAL":
                    send_esp32_command("STOP")
                    send_esp32_command("MANUAL")
                else:
                    send_esp32_command("AUTO")
                print(f"[ESP32] MODE RE-SYNCED: {rover_mode}")
            except Exception as e:
                print(f"[ESP32] RE-SYNC ERROR: {e}")

        was_online = esp32_online

        time.sleep(3)


# ============================================================
# LOAD YOLO
# ============================================================
# LOAD YOLO
# ============================================================

print()
print("======================================")
print("LOADING YOLO")
print("======================================")

if not os.path.exists(YOLO_CFG):

    print("ERROR:", YOLO_CFG)

    raise SystemExit


if not os.path.exists(YOLO_WEIGHTS):

    print("ERROR:", YOLO_WEIGHTS)

    raise SystemExit


if not os.path.exists(YOLO_NAMES):

    print("ERROR:", YOLO_NAMES)

    raise SystemExit


net = cv2.dnn.readNetFromDarknet(
    YOLO_CFG,
    YOLO_WEIGHTS
)

net.setPreferableBackend(
    cv2.dnn.DNN_BACKEND_OPENCV
)

net.setPreferableTarget(
    cv2.dnn.DNN_TARGET_CPU
)


with open(
    YOLO_NAMES,
    "r"
) as f:

    class_names = [
        line.strip()
        for line in f.readlines()
    ]


layer_names = net.getLayerNames()

output_layers = [
    layer_names[i - 1]
    for i in net.getUnconnectedOutLayers()
]


print("YOLO loaded.")


# ============================================================
# LOAD VISHNU FACE DATABASE
# ============================================================

print()
print("======================================")
print("LOADING VISHNU FACE DATABASE")
print("======================================")


vishnu_folder = os.path.join(
    KNOWN_DIR,
    "Vishnu"
)


if not os.path.isdir(vishnu_folder):

    print(
        "ERROR: known_faces/Vishnu does not exist"
    )

    raise SystemExit


files = sorted(
    os.listdir(vishnu_folder)
)


for filename in files:

    if not filename.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):
        continue


    path = os.path.join(
        vishnu_folder,
        filename
    )


    try:

        image = face_recognition.load_image_file(
            path
        )


        locations = face_recognition.face_locations(
            image,
            model="hog"
        )


        if len(locations) == 0:

            print(
                "NO FACE:",
                filename
            )

            continue


        # Largest face
        location = max(
            locations,
            key=lambda b:
            (b[2] - b[0]) *
            (b[1] - b[3])
        )


        encodings = face_recognition.face_encodings(
            image,
            known_face_locations=[location]
        )


        if len(encodings) == 1:

            known_encodings.append(
                encodings[0]
            )

            known_names.append(
                "Vishnu"
            )

            print(
                "Loaded:",
                filename
            )


    except Exception as e:

        print(
            "ERROR:",
            filename,
            e
        )


print()

print(
    "TOTAL VISHNU ENCODINGS:",
    len(known_encodings)
)


if len(known_encodings) == 0:

    print(
        "ERROR: No valid Vishnu faces."
    )

    raise SystemExit


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_V4L2
)


camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    FRAME_WIDTH
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    FRAME_HEIGHT
)

camera.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


if not camera.isOpened():

    print(
        "ERROR: Camera cannot be opened."
    )

    raise SystemExit


print()
print("CAMERA READY")
print("======================================")


# ============================================================
# YOLO PERSON DETECTION
# ============================================================

def detect_persons(frame):

    height, width = frame.shape[:2]


    blob = cv2.dnn.blobFromImage(
        frame,
        1 / 255.0,
        (320, 320),
        swapRB=True,
        crop=False
    )


    net.setInput(blob)


    outputs = net.forward(
        output_layers
    )


    boxes = []
    confidences = []


    for output in outputs:

        for detection in output:

            scores = detection[5:]


            class_id = np.argmax(
                scores
            )


            confidence = float(
                scores[class_id]
            )


            # COCO class 0 = person
            if (
                class_id != 0
                or
                confidence < YOLO_CONFIDENCE
            ):

                continue


            center_x = int(
                detection[0] * width
            )

            center_y = int(
                detection[1] * height
            )

            w = int(
                detection[2] * width
            )

            h = int(
                detection[3] * height
            )


            x = int(
                center_x - w / 2
            )

            y = int(
                center_y - h / 2
            )


            x = max(0, x)
            y = max(0, y)


            w = min(
                w,
                width - x
            )

            h = min(
                h,
                height - y
            )


            if w <= 20 or h <= 40:

                continue


            boxes.append(
                [x, y, w, h]
            )

            confidences.append(
                confidence
            )


    if len(boxes) == 0:

        return []


    indexes = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        YOLO_CONFIDENCE,
        YOLO_NMS
    )


    persons = []


    if len(indexes) > 0:

        for index in indexes:

            index = int(index)


            persons.append(
                {
                    "box": boxes[index],

                    "confidence":
                        confidences[index]
                }
            )


    return persons


# ============================================================
# FACE RECOGNITION
# ============================================================

def recognize_faces(frame):

    # Half resolution for Raspberry Pi speed

    small = cv2.resize(
        frame,
        (0, 0),
        fx=0.5,
        fy=0.5
    )


    rgb = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2RGB
    )


    locations = face_recognition.face_locations(
        rgb,
        model="hog"
    )


    if not locations:

        return []


    encodings = face_recognition.face_encodings(
        rgb,
        locations
    )


    results = []


    for location, encoding in zip(
        locations,
        encodings
    ):


        top, right, bottom, left = location


        # Back to original coordinates

        top *= 2
        right *= 2
        bottom *= 2
        left *= 2


        distances = face_recognition.face_distance(
            known_encodings,
            encoding
        )


        best_index = int(
            np.argmin(distances)
        )


        best_distance = float(
            distances[best_index]
        )


        print(
            "FACE DISTANCE:",
            round(best_distance, 3)
        )


        if best_distance <= FACE_TOLERANCE:

            name = "Vishnu"

            known = True

        else:

            name = "UNKNOWN"

            known = False


        results.append(
            {
                "location": (
                    left,
                    top,
                    right,
                    bottom
                ),

                "encoding": encoding,

                "name": name,

                "known": known,

                "distance": best_distance
            }
        )


    return results


# ============================================================
# CHECK FACE INSIDE PERSON
# ============================================================

def face_belongs_to_person(
    face,
    person_box
):

    fx1, fy1, fx2, fy2 = (
        face["location"]
    )


    px, py, pw, ph = person_box


    px2 = px + pw
    py2 = py + ph


    face_cx = (
        fx1 + fx2
    ) / 2


    face_cy = (
        fy1 + fy2
    ) / 2


    return (
        px <= face_cx <= px2
        and
        py <= face_cy <= py2
    )


# ============================================================
# UNKNOWN CAPTURE
# ============================================================

def can_save_unknown(
    encoding
):

    global unknown_history


    now = time.time()


    unknown_history = [
        item
        for item in unknown_history
        if now - item["time"]
        < UNKNOWN_COOLDOWN
    ]


    for item in unknown_history:

        distance = face_recognition.face_distance(
            [item["encoding"]],
            encoding
        )[0]


        if distance < 0.55:

            return False


    return True


def save_unknown(
    frame,
    person_box,
    encoding
):

    global unknown_count
    global unknown_history


    if encoding is None:

        return


    if not can_save_unknown(
        encoding
    ):

        return


    x, y, w, h = person_box


    padding = 10


    x1 = max(
        0,
        x - padding
    )


    y1 = max(
        0,
        y - padding
    )


    x2 = min(
        frame.shape[1],
        x + w + padding
    )


    y2 = min(
        frame.shape[0],
        y + h + padding
    )


    image = frame[
        y1:y2,
        x1:x2
    ]


    if image.size == 0:

        return


    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


    filename = (
        "unknown_"
        +
        timestamp
        +
        ".jpg"
    )


    filepath = os.path.join(
        UNKNOWN_DIR,
        filename
    )


    cv2.imwrite(
        filepath,
        image
    )


    unknown_history.append(
        {
            "encoding": encoding,

            "time": time.time()
        }
    )


    unknown_count += 1


    print(
        "UNKNOWN SAVED:",
        filename
    )


# ============================================================
# CENTER DISTANCE
# ============================================================

def center_of(box):

    x, y, w, h = box


    return (
        x + w / 2,
        y + h / 2
    )


def center_distance(
    box1,
    box2
):

    x1, y1 = center_of(box1)

    x2, y2 = center_of(box2)


    return np.sqrt(
        (x1 - x2) ** 2
        +
        (y1 - y2) ** 2
    )


# ============================================================
# ASSIGN IDENTITIES
# ============================================================

def assign_people(
    persons,
    faces,
    frame
):

    global tracked_people
    global next_person_id


    new_people = {}

    used_ids = set()


    for person in persons:

        box = person["box"]


        matching_face = None


        for face in faces:

            if face_belongs_to_person(
                face,
                box
            ):

                matching_face = face

                break


        # ----------------------------------------------------
        # Find previous person ID
        # ----------------------------------------------------

        best_id = None

        best_distance = 999999


        for pid, old in tracked_people.items():

            if pid in used_ids:

                continue


            distance = center_distance(
                box,
                old["box"]
            )


            if (
                distance < 150
                and
                distance < best_distance
            ):

                best_distance = distance

                best_id = pid


        if best_id is None:

            best_id = next_person_id

            next_person_id += 1


        used_ids.add(
            best_id
        )


        # ----------------------------------------------------
        # Identity update
        # ----------------------------------------------------

        if matching_face is not None:

            name = matching_face["name"]

            known = matching_face["known"]

            encoding = matching_face["encoding"]


            if not known:

                save_unknown(
                    frame,
                    box,
                    encoding
                )


        else:

            if best_id in tracked_people:

                name = tracked_people[
                    best_id
                ]["name"]

                known = tracked_people[
                    best_id
                ]["known"]

            else:

                name = "UNKNOWN"

                known = False


        new_people[best_id] = {

            "box": box,

            "name": name,

            "known": known,

            "last_seen":
                time.time()
        }


    tracked_people = new_people


# ============================================================
# DRAW FRAME
# ============================================================

def process_frame(frame):

    global frame_number


    frame_number += 1


    # --------------------------------------------------------
    # Detection / recognition
    # --------------------------------------------------------

    if frame_number % DETECT_EVERY == 0:

        persons = detect_persons(
            frame
        )


        faces = recognize_faces(
            frame
        )


        with tracking_lock:

            assign_people(
                persons,
                faces,
                frame
            )


    # --------------------------------------------------------
    # Draw tracked people
    # --------------------------------------------------------

    with tracking_lock:

        people_copy = dict(
            tracked_people
        )


    for pid, person in people_copy.items():

        box = person["box"]


        x, y, w, h = box


        name = person["name"]

        known = person["known"]


        if known:

            color = (
                0,
                255,
                0
            )

            label = "VISHNU - KNOWN"

        else:

            color = (
                0,
                0,
                255
            )

            label = "UNKNOWN"


        # Full body rectangle

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            color,
            4
        )


        text_size = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2
        )[0]


        label_y = max(
            0,
            y - text_size[1] - 12
        )


        cv2.rectangle(
            frame,
            (x, label_y),
            (
                x +
                text_size[0] +
                14,
                y
            ),
            color,
            -1
        )


        cv2.putText(
            frame,
            label,
            (
                x + 7,
                y - 7
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


    # --------------------------------------------------------
    # Dashboard information on camera
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "SURVEILLANCE ROVER",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # Mode

    mode_text = (
        "MODE: "
        +
        rover_mode
    )


    cv2.putText(
        frame,
        mode_text,
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ESP32 status

    if esp32_online:

        esp_text = "ESP32: ONLINE"

    else:

        esp_text = "ESP32: OFFLINE"


    cv2.putText(
        frame,
        esp_text,
        (15, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    return frame


# ============================================================
# CAMERA THREAD
# ============================================================

def camera_loop():

    global latest_frame
    global latest_jpeg


    while camera_running:

        success, frame = camera.read()


        if not success:

            time.sleep(0.01)

            continue


        processed = process_frame(
            frame
        )


        ret, buffer = cv2.imencode(
            ".jpg",
            processed,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                80
            ]
        )


        if not ret:

            continue


        jpeg = buffer.tobytes()


        with frame_lock:

            latest_frame = processed.copy()

            latest_jpeg = jpeg


        # Small sleep prevents CPU overload

        time.sleep(0.001)


# ============================================================
# VIDEO STREAM
# ============================================================

def generate_frames():

    while camera_running:

        with frame_lock:

            jpeg = latest_jpeg


        if jpeg is None:

            time.sleep(0.01)

            continue


        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            +
            jpeg
            +
            b"\r\n"
        )


        time.sleep(0.03)


# ============================================================
# WEB PAGE
# ============================================================

HTML = """

<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta name="viewport"
content="width=device-width, initial-scale=1">

<title>Surveillance Rover</title>


<style>

* {
    box-sizing: border-box;
}


body {

    margin: 0;

    font-family:
        Arial,
        sans-serif;

    background: #f4f6f8;

    color: #111827;
}


.header {

    background: #111827;

    color: white;

    padding: 16px 20px;

    display: flex;

    justify-content:
        space-between;

    align-items: center;

    gap: 10px;

}


.header h2 {

    margin: 0;

    font-size: 21px;

}


.connection {

    font-size: 13px;

}


.container {

    max-width: 1250px;

    margin: 20px auto;

    padding:
        0 15px;

}


.dashboard {

    display: grid;

    grid-template-columns:
        1.5fr 1fr;

    gap: 18px;

}


.card {

    background: white;

    padding: 18px;

    border-radius: 14px;

    box-shadow:
        0 4px 15px
        rgba(0,0,0,0.08);

}


.video {

    width: 100%;

    display: block;

    border-radius: 10px;

    background: black;

}


.stats {

    display: grid;

    grid-template-columns:
        1fr 1fr;

    gap: 10px;

    margin-bottom: 20px;

}


.stat {

    padding: 14px;

    text-align: center;

    border-radius: 10px;

}


.known {

    background: #dcfce7;

    color: #166534;

}


.unknown {

    background: #fee2e2;

    color: #991b1b;

}


.number {

    font-size: 28px;

    font-weight: bold;

}


.mode-title {

    margin-top: 10px;

}


.mode-buttons {

    display: grid;

    grid-template-columns:
        1fr 1fr;

    gap: 10px;

}


.mode-btn {

    border: none;

    padding: 13px;

    border-radius: 9px;

    font-weight: bold;

    cursor: pointer;

}


.manual-btn {

    background: #2563eb;

    color: white;

}


.auto-btn {

    background: #16a34a;

    color: white;

}


.mode-status {

    margin-top: 10px;

    text-align: center;

    padding: 10px;

    background: #f3f4f6;

    border-radius: 8px;

    font-weight: bold;

}


.controls {

    margin-top: 15px;

}


.control-row {

    display: flex;

    justify-content: center;

    gap: 10px;

    margin: 8px 0;

}


.control-btn {

    width: 65px;

    height: 55px;

    border: none;

    border-radius: 12px;

    font-size: 24px;

    font-weight: bold;

    cursor: pointer;

    background: #e5e7eb;

}


.control-btn {
    touch-action: none;
    user-select: none;
    -webkit-user-select: none;
}

.control-btn:active {

    transform: scale(0.95);

}


.stop-btn {

    background: #dc2626;

    color: white;

    font-size: 14px;

}


.manual-only {

    display: none;

}


.esp-status {

    margin-top: 15px;

    padding: 10px;

    border-radius: 8px;

    text-align: center;

    font-weight: bold;

}


.esp-online {

    background: #dcfce7;

    color: #166534;

}


.esp-offline {

    background: #fee2e2;

    color: #991b1b;

}


.gallery {

    display: grid;

    grid-template-columns:
        repeat(2, 1fr);

    gap: 10px;

    max-height: 420px;

    overflow-y: auto;

}


.gallery img {

    width: 100%;

    aspect-ratio: 1;

    object-fit: cover;

    border-radius: 10px;

    border:
        3px solid #ef4444;

}


button {

    -webkit-tap-highlight-color:
        transparent;

    user-select: none;

    touch-action: manipulation;

}


@media(max-width: 800px) {

    .dashboard {

        grid-template-columns:
            1fr;

    }


    .header {

        padding:
            13px 15px;

    }


    .header h2 {

        font-size: 18px;

    }


    .container {

        margin-top: 12px;

    }


    .card {

        padding: 14px;

    }


    .control-btn {

        width: 72px;

        height: 60px;

    }

}



/* ---------- PREDICTIVE MAINTENANCE ---------- */
.pm-card { grid-column: 1 / -1; }
.pm-head { display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; }
.pm-tag { font-size:11px; background:#fef3c7; color:#92400e; padding:3px 8px; border-radius:6px; font-weight:bold; }
.pm-badge { padding:6px 14px; border-radius:20px; color:white; font-weight:bold; font-size:14px; }
.pm-GOOD { background:#16a34a; } .pm-WARNING { background:#f59e0b; } .pm-CRITICAL { background:#dc2626; }
.pm-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-top:12px; }
.pm-box { background:#f4f6f8; border-radius:10px; padding:12px; }
.pm-label { font-size:12px; color:#6b7280; }
.pm-value { font-size:24px; font-weight:bold; margin-top:2px; }
.pm-sub { font-size:11px; color:#6b7280; margin-top:2px; }
.pm-bar { height:8px; background:#e5e7eb; border-radius:6px; margin-top:8px; overflow:hidden; }
.pm-fill { height:100%; width:0; background:#2563eb; transition:width .8s; }
.pm-advice { margin-top:12px; padding:10px 12px; border-radius:8px; background:#eff6ff; font-size:14px; }
.pm-actions { margin-top:10px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; }
.pm-btn { padding:8px 14px; border:0; border-radius:8px; background:#111827; color:white; cursor:pointer; }
.pm-note { font-size:11px; color:#6b7280; }
#pmChart { width:100%; height:90px; background:#f4f6f8; border-radius:10px; margin-top:12px; display:block; }
@media(max-width: 800px) { .pm-grid { grid-template-columns:repeat(2,1fr); } }

</style>

</head>


<body>


<div class="header">

    <h2>
        Surveillance Rover
    </h2>

    <div
        class="connection"
        id="headerStatus"
    >
        ● CHECKING
    </div>

</div>


<div class="container">


<div class="dashboard">


<!-- =====================================================
     LEFT
====================================================== -->

<div class="card">

    <h3>
        Live Surveillance
    </h3>


    <img
        class="video"
        src="{{ url_for('video_feed') }}"
    >


    <div
        class="esp-status"
        id="espStatus"
    >
        Checking ESP32...
    </div>

</div>


<!-- =====================================================
     RIGHT
====================================================== -->

<div class="card">


    <h3>
        Rover Control
    </h3>


    <div
        class="mode-buttons"
    >

        <button
            class="mode-btn manual-btn"
            onclick="setMode('MANUAL')"
        >
            MANUAL
        </button>


        <button
            class="mode-btn auto-btn"
            onclick="setMode('AUTO')"
        >
            AUTONOMOUS
        </button>

    </div>


    <div
        class="mode-status"
        id="modeStatus"
    >
        MODE: AUTONOMOUS
    </div>


    <!-- Manual controls -->

    <div
        class="controls manual-only"
        id="manualControls"
    >

        <div
            class="control-row"
        >

            <button
                class="control-btn"
                data-cmd="FORWARD"
            >
                ▲
            </button>

        </div>


        <div
            class="control-row"
        >

            <button
                class="control-btn"
                data-cmd="LEFT"
            >
                ◀
            </button>


            <button
                class="control-btn stop-btn"
                onclick="sendCommand('STOP')"
            >
                STOP
            </button>


            <button
                class="control-btn"
                data-cmd="RIGHT"
            >
                ▶
            </button>

        </div>


        <div
            class="control-row"
        >

            <button
                class="control-btn"
                data-cmd="BACKWARD"
            >
                ▼
            </button>

        </div>

    </div>


    <h3>
        Detection Status
    </h3>


    <div class="stats">


        <div class="stat known">

            Known

            <div
                class="number"
                id="knownCount"
            >
                0
            </div>

        </div>


        <div class="stat unknown">

            Unknown

            <div
                class="number"
                id="unknownCount"
            >
                {{ unknown_images|length }}
            </div>

        </div>


    </div>


    <h3>
        Unknown Persons
    </h3>


    <div
        class="gallery"
        id="gallery"
    >

        {% for image in unknown_images %}

        <img
            src="{{ url_for(
                'unknown_file',
                filename=image
            ) }}"
        >

        {% endfor %}

    </div>


</div>


<!-- =====================================================
     PREDICTIVE MAINTENANCE (SIMULATED)
====================================================== -->

<div class="card pm-card">

    <div class="pm-head">
        <h3 style="margin:0">Predictive Maintenance</h3>
        <span class="pm-tag">SIMULATED VALUES</span>
        <span class="pm-badge pm-GOOD" id="pmLevel">GOOD</span>
    </div>

    <div class="pm-grid">

        <div class="pm-box">
            <div class="pm-label">Running Time</div>
            <div class="pm-value" id="pmUptime">0m 0s</div>
            <div class="pm-sub">Moving: <span id="pmMoving">0m 0s</span></div>
        </div>

        <div class="pm-box">
            <div class="pm-label">Distance Covered</div>
            <div class="pm-value" id="pmDistance">0 m</div>
            <div class="pm-sub">Assumed speed 0.25 m/s</div>
        </div>

        <div class="pm-box">
            <div class="pm-label">Battery</div>
            <div class="pm-value" id="pmBattery">100%</div>
            <div class="pm-bar"><div class="pm-fill" id="pmBatteryBar" style="background:#16a34a"></div></div>
            <div class="pm-sub">~<span id="pmBattLeft">0</span> min left</div>
        </div>

        <div class="pm-box">
            <div class="pm-label">Motor Temperature</div>
            <div class="pm-value" id="pmTemp">30 &deg;C</div>
            <div class="pm-bar"><div class="pm-fill" id="pmTempBar" style="background:#f59e0b"></div></div>
            <div class="pm-sub" id="pmMotion">Idle</div>
        </div>

        <div class="pm-box">
            <div class="pm-label">Motor Wear</div>
            <div class="pm-value" id="pmWear">0%</div>
            <div class="pm-bar"><div class="pm-fill" id="pmWearBar"></div></div>
        </div>

        <div class="pm-box">
            <div class="pm-label">Next Service In</div>
            <div class="pm-value" id="pmService">-- min</div>
            <div class="pm-sub">Based on distance / run time</div>
        </div>

        <div class="pm-box">
            <div class="pm-label">Health Score</div>
            <div class="pm-value" id="pmHealth">100</div>
            <div class="pm-bar"><div class="pm-fill" id="pmHealthBar" style="background:#16a34a"></div></div>
        </div>

        <div class="pm-box">
            <div class="pm-label">Prediction</div>
            <div class="pm-value" style="font-size:15px" id="pmAdviceShort">Healthy</div>
            <div class="pm-sub">Updated every 2 s</div>
        </div>

    </div>

    <canvas id="pmChart" width="1000" height="90"></canvas>
    <div class="pm-note">Chart: <span style="color:#16a34a">battery %</span> and <span style="color:#dc2626">temperature &deg;C</span> (last 60 readings)</div>

    <div class="pm-advice" id="pmAdvice">Rover is healthy - no action needed</div>

    <div class="pm-actions">
        <span class="pm-note">Values are assumptions for demonstration, not real sensor data.</span>
        <button class="pm-btn" onclick="pmReset()">Service Done / Recharge</button>
    </div>

</div>


</div>

</div>


<script>


let oldImages = [];

let currentMode = "AUTO";


/* =====================================================
   SEND COMMAND
===================================================== */

function showEspError(message) {

    const status = document.getElementById("espStatus");

    status.innerText = "ESP32 ERROR: " + message;
    status.className = "esp-status esp-offline";

}


function sendCommand(command) {

    return fetch(
        "/control/" + encodeURIComponent(command),
        {
            method: "GET",
            cache: "no-store"
        }
    )
    .then(async response => {

        let data = {};

        try {
            data = await response.json();
        } catch (e) {}

        if (!response.ok || !data.success) {
            throw new Error(
                data.esp32 ||
                data.message ||
                "ESP32 command failed"
            );
        }

        console.log("Command OK:", data);

        if (data.mode) {
            currentMode = data.mode;
        }

        return data;
    })
    .catch(error => {

        console.error("Command error:", error);

        showEspError(error.message);

        return null;

    });

}


/* =====================================================
   MODE  (UI changes ONLY after the ESP32 confirms)
===================================================== */

function applyModeUI(mode) {

    const controls  = document.getElementById("manualControls");
    const modeStatus = document.getElementById("modeStatus");

    if (mode === "MANUAL") {
        controls.style.display = "block";
        modeStatus.innerText = "MODE: MANUAL";
    }
    else {
        controls.style.display = "none";
        modeStatus.innerText = "MODE: AUTONOMOUS";
    }

}


async function setMode(mode) {

    const result = await sendCommand(mode);

    if (result && result.success) {
        currentMode = result.mode;
        applyModeUI(result.mode);
    }

}


/* =====================================================
   HOLD-TO-MOVE BUTTONS
   Press  = move
   Release = STOP
===================================================== */

let activeMove = null;

function startMove(cmd) {

    if (currentMode !== "MANUAL") {
        return;
    }

    if (activeMove === cmd) {
        return;
    }

    activeMove = cmd;

    sendCommand(cmd);

}

function endMove() {

    if (activeMove === null) {
        return;
    }

    activeMove = null;

    sendCommand("STOP");

}

function bindMoveButtons() {

    document.querySelectorAll("[data-cmd]").forEach(btn => {

        const cmd = btn.getAttribute("data-cmd");

        btn.addEventListener("pointerdown", e => {
            e.preventDefault();
            startMove(cmd);
        });

        btn.addEventListener("pointerup",     endMove);
        btn.addEventListener("pointerleave",  endMove);
        btn.addEventListener("pointercancel", endMove);

        btn.addEventListener("contextmenu", e => e.preventDefault());

    });

}

bindMoveButtons();


/* =====================================================
   STATUS
===================================================== */

function updateStatus() {

    fetch(
        "/status"
    )

    .then(
        response =>
        response.json()
    )

    .then(
        data => {


            document.getElementById(
                "knownCount"
            ).innerText =
                data.known_count;


            document.getElementById(
                "unknownCount"
            ).innerText =
                data.images.length;


            /* ESP32 */

            const esp =
                document.getElementById(
                    "espStatus"
                );


            const header =
                document.getElementById(
                    "headerStatus"
                );


            if (
                data.esp32_online
            ) {

                esp.innerText =
                    "ESP32: ONLINE";


                esp.className =
                    "esp-status esp-online";


                header.innerText =
                    "● ESP32 ONLINE";

            }

            else {

                esp.innerText =
                    "ESP32: OFFLINE";


                esp.className =
                    "esp-status esp-offline";


                header.innerText =
                    "● ESP32 OFFLINE";

            }


            /* Mode */

            currentMode =
                data.mode;


            const modeStatus =
                document.getElementById(
                    "modeStatus"
                );


            if (
                data.mode ===
                "MANUAL"
            ) {

                modeStatus.innerText =
                    "MODE: MANUAL";

                document.getElementById(
                    "manualControls"
                ).style.display =
                    "block";

            }

            else {

                modeStatus.innerText =
                    "MODE: AUTONOMOUS";

                document.getElementById(
                    "manualControls"
                ).style.display =
                    "none";

            }


            /* Images */

            if (
                JSON.stringify(
                    oldImages
                )
                ===
                JSON.stringify(
                    data.images
                )
            ) {

                return;

            }


            oldImages =
                data.images;


            const gallery =
                document.getElementById(
                    "gallery"
                );


            gallery.innerHTML = "";


            data.images.forEach(
                function(image) {

                    const img =
                        document.createElement(
                            "img"
                        );


                    img.src =
                        "/unknown/" +
                        encodeURIComponent(
                            image
                        ) +
                        "?t=" +
                        Date.now();


                    gallery.appendChild(
                        img
                    );

                }
            );

        }
    )

    .catch(
        error => {

            console.log(
                "Status error:",
                error
            );

        }
    );

}


/* =====================================================
   UPDATE EVERY 2 SECONDS
===================================================== */

setInterval(
    updateStatus,
    2000
);


updateStatus();



/* =====================================================
   PREDICTIVE MAINTENANCE (SIMULATED)
===================================================== */

const pmBatt = [];
const pmTemp = [];

function pmFmt(sec) {
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = sec % 60;
    return (h ? h + "h " : "") + m + "m " + s + "s";
}

function pmSet(id, text) {
    document.getElementById(id).innerText = text;
}

function pmBar(id, pct) {
    document.getElementById(id).style.width =
        Math.max(0, Math.min(100, pct)) + "%";
}

function pmDraw() {
    const c = document.getElementById("pmChart");
    const g = c.getContext("2d");
    g.clearRect(0, 0, c.width, c.height);

    function line(arr, color, min, max) {
        if (arr.length < 2) return;
        g.beginPath();
        g.strokeStyle = color;
        g.lineWidth = 2;
        arr.forEach((v, i) => {
            const x = i * (c.width / 59);
            const y = c.height - 6 -
                ((v - min) / (max - min)) * (c.height - 12);
            if (i === 0) g.moveTo(x, y); else g.lineTo(x, y);
        });
        g.stroke();
    }

    line(pmBatt, "#16a34a", 0, 100);
    line(pmTemp, "#dc2626", 20, 80);
}

function pmUpdate() {
    fetch("/maintenance")
    .then(r => r.json())
    .then(d => {
        pmSet("pmUptime", pmFmt(d.uptime_s));
        pmSet("pmMoving", pmFmt(d.moving_s));
        pmSet("pmDistance", d.distance_m + " m");
        pmSet("pmBattery", d.battery_pct + "%");
        pmSet("pmBattLeft", d.battery_left_min);
        pmSet("pmTemp", d.temp_c + " \u00B0C");
        pmSet("pmMotion", d.moving ? "Rover moving" : "Idle / cooling");
        pmSet("pmWear", d.wear_pct + "%");
        pmSet("pmService", d.service_in_min + " min");
        pmSet("pmHealth", d.health);
        pmSet("pmAdvice", d.advice);
        pmSet("pmAdviceShort",
            d.level === "GOOD" ? "Healthy" :
            d.level === "WARNING" ? "Service soon" : "Service now");

        pmBar("pmBatteryBar", d.battery_pct);
        pmBar("pmTempBar", (d.temp_c - 20) / 60 * 100);
        pmBar("pmWearBar", d.wear_pct);
        pmBar("pmHealthBar", d.health);

        const badge = document.getElementById("pmLevel");
        badge.innerText = d.level;
        badge.className = "pm-badge pm-" + d.level;

        pmBatt.push(d.battery_pct);
        pmTemp.push(d.temp_c);
        if (pmBatt.length > 60) { pmBatt.shift(); pmTemp.shift(); }
        pmDraw();
    })
    .catch(e => console.log("PM error:", e));
}

function pmReset() {
    fetch("/maintenance/reset", { method: "POST" }).then(pmUpdate);
}

setInterval(pmUpdate, 2000);
pmUpdate();


/* =====================================================
   KEYBOARD CONTROL  (hold arrow key = move, release = stop)
===================================================== */

const KEY_MAP = {
    "ArrowUp":    "FORWARD",
    "ArrowDown":  "BACKWARD",
    "ArrowLeft":  "LEFT",
    "ArrowRight": "RIGHT"
};

document.addEventListener("keydown", function(event) {

    if (currentMode !== "MANUAL" || event.repeat) {
        return;
    }

    if (event.key === " ") {
        event.preventDefault();
        activeMove = null;
        sendCommand("STOP");
        return;
    }

    if (KEY_MAP[event.key]) {
        event.preventDefault();
        startMove(KEY_MAP[event.key]);
    }

});

document.addEventListener("keyup", function(event) {

    if (KEY_MAP[event.key]) {
        endMove();
    }

});


/* =====================================================
   EMERGENCY STOP WHEN PAGE IS CLOSED
===================================================== */

window.addEventListener(
    "beforeunload",
    function() {

        navigator.sendBeacon(
            "/control/STOP"
        );

    }
);


</script>


</body>

</html>
"""



# ============================================================
# PREDICTIVE MAINTENANCE (SIMULATED / ASSUMED VALUES)
# ============================================================
# These are NOT real sensor readings. They are estimated from
# how long the rover has been moving, using simple assumptions,
# only to demonstrate the predictive-maintenance concept.

PM_ASSUMED_SPEED_MPS = 0.25        # rover assumed to move 0.25 m/s
PM_AMBIENT_TEMP = 30.0             # room / outdoor temperature (C)
PM_MAX_MOTOR_TEMP = 65.0           # max temperature the motor can reach
PM_BATTERY_DRAIN_MOVING = 0.06     # battery % lost per second while moving
PM_BATTERY_DRAIN_IDLE = 0.005      # battery % lost per second while idle
PM_SERVICE_DISTANCE_M = 500.0      # assumed service interval (metres)
PM_SERVICE_RUNTIME_S = 3600.0      # assumed service interval (1 hour moving)

pm_lock = threading.Lock()

pm_state = {
    "uptime_s": 0.0,        # total time the program has been running
    "moving_s": 0.0,        # time the rover was moving
    "distance_m": 0.0,      # assumed distance travelled
    "battery_pct": 100.0,   # assumed battery level
    "temp_c": PM_AMBIENT_TEMP,
    "moving": False,
}


def pm_is_moving():
    """Rover is assumed to be moving in AUTO mode (when ESP32 is online)
    or when a movement command is active in MANUAL mode."""
    if not esp32_online:
        return False
    if rover_mode == "AUTO":
        return True
    return last_command in ("FORWARD", "BACKWARD", "LEFT", "RIGHT")


def pm_loop():
    """Update the assumed values once every second."""
    import random

    while camera_running:
        time.sleep(1)

        moving = pm_is_moving()

        with pm_lock:
            pm_state["uptime_s"] += 1
            pm_state["moving"] = moving

            if moving:
                pm_state["moving_s"] += 1
                pm_state["distance_m"] += PM_ASSUMED_SPEED_MPS
                pm_state["battery_pct"] -= PM_BATTERY_DRAIN_MOVING
                # temperature rises slowly towards the maximum
                pm_state["temp_c"] += (
                    (PM_MAX_MOTOR_TEMP - pm_state["temp_c"]) * 0.0015
                )
            else:
                pm_state["battery_pct"] -= PM_BATTERY_DRAIN_IDLE
                # temperature cools down towards ambient
                pm_state["temp_c"] -= (
                    (pm_state["temp_c"] - PM_AMBIENT_TEMP) * 0.02
                )

            # small random variation so it looks like a live reading
            pm_state["temp_c"] += random.uniform(-0.15, 0.15)
            pm_state["battery_pct"] = max(0.0, pm_state["battery_pct"])


def pm_snapshot():
    """Build the values shown on the dashboard."""
    with pm_lock:
        st = dict(pm_state)

    # Motor wear: whichever is closer - distance or moving time
    wear_dist = st["distance_m"] / PM_SERVICE_DISTANCE_M
    wear_time = st["moving_s"] / PM_SERVICE_RUNTIME_S
    wear_pct = min(100.0, max(wear_dist, wear_time) * 100.0)

    # Remaining battery time (at the moving drain rate)
    batt_left_min = st["battery_pct"] / PM_BATTERY_DRAIN_MOVING / 60.0

    # Estimated time left before maintenance is needed
    dist_left = max(0.0, PM_SERVICE_DISTANCE_M - st["distance_m"])
    time_left = max(0.0, PM_SERVICE_RUNTIME_S - st["moving_s"])
    service_in_min = min(
        dist_left / PM_ASSUMED_SPEED_MPS, time_left
    ) / 60.0

    # Overall health score
    health = 100.0
    health -= wear_pct * 0.5
    health -= max(0.0, st["temp_c"] - 45.0) * 1.0
    health -= max(0.0, 30.0 - st["battery_pct"]) * 0.5
    health = max(0.0, min(100.0, health))

    # Status + advice
    if st["temp_c"] >= 60 or wear_pct >= 90 or st["battery_pct"] <= 10:
        level, advice = "CRITICAL", "Stop the rover - service / recharge needed now"
    elif st["temp_c"] >= 50 or wear_pct >= 70 or st["battery_pct"] <= 25:
        level, advice = "WARNING", "Maintenance needed soon - plan a service"
    else:
        level, advice = "GOOD", "Rover is healthy - no action needed"

    return {
        "uptime_s": int(st["uptime_s"]),
        "moving_s": int(st["moving_s"]),
        "distance_m": round(st["distance_m"], 1),
        "battery_pct": round(st["battery_pct"], 1),
        "temp_c": round(st["temp_c"], 1),
        "moving": st["moving"],
        "wear_pct": round(wear_pct, 1),
        "battery_left_min": round(batt_left_min, 1),
        "service_in_min": round(service_in_min, 1),
        "health": round(health),
        "level": level,
        "advice": advice,
        "simulated": True,
    }


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():

    images = []


    if os.path.exists(
        UNKNOWN_DIR
    ):

        images = [

            f

            for f in os.listdir(
                UNKNOWN_DIR
            )

            if f.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png"
                )
            )

        ]


        images.sort(
            reverse=True
        )


    return render_template_string(
        HTML,
        unknown_images=images
    )


# ============================================================
# VIDEO FEED
# ============================================================

@app.route("/video_feed")
def video_feed():

    return Response(

        generate_frames(),

        mimetype=
        "multipart/x-mixed-replace; boundary=frame"

    )


# ============================================================
# UNKNOWN IMAGE
# ============================================================

@app.route(
    "/unknown/<path:filename>"
)
def unknown_file(filename):

    return send_from_directory(
        UNKNOWN_DIR,
        filename
    )


# ============================================================
# ROVER CONTROL
# ============================================================

@app.route(
    "/control/<command>",
    methods=["GET", "POST"]
)
def control(command):
    """
    Web control endpoint.

    Browser -> Raspberry Pi -> ESP32

    The Pi changes its local mode ONLY after the ESP32 accepts
    the requested mode. This prevents the web UI from claiming
    MANUAL/AUTO when the ESP32 is actually unreachable.
    """
    global rover_mode

    command = str(command).strip().upper()

    allowed_commands = {
        "FORWARD",
        "BACKWARD",
        "LEFT",
        "RIGHT",
        "STOP",
        "AUTO",
        "MANUAL"
    }

    if command not in allowed_commands:
        return jsonify({
            "success": False,
            "message": "Invalid command",
            "mode": rover_mode,
            "esp32_online": esp32_online
        }), 400

    # --------------------------------------------------------
    # AUTO MODE
    # --------------------------------------------------------
    if command == "AUTO":
        success, result = send_esp32_command("AUTO")

        if success:
            rover_mode = "AUTO"

        return jsonify({
            "success": success,
            "command": "AUTO",
            "mode": rover_mode,
            "esp32": result,
            "esp32_online": esp32_online
        }), (200 if success else 503)

    # --------------------------------------------------------
    # MANUAL MODE
    # --------------------------------------------------------
    if command == "MANUAL":

        # Stop the rover before changing control mode.
        stop_success, stop_result = send_esp32_command("STOP")

        if not stop_success:
            return jsonify({
                "success": False,
                "command": "MANUAL",
                "mode": rover_mode,
                "esp32": (
                    "Cannot enter MANUAL mode. "
                    f"STOP failed: {stop_result}"
                ),
                "esp32_online": esp32_online
            }), 503

        manual_success, manual_result = send_esp32_command("MANUAL")

        if manual_success:
            rover_mode = "MANUAL"

        return jsonify({
            "success": manual_success,
            "command": "MANUAL",
            "mode": rover_mode,
            "esp32": manual_result,
            "esp32_online": esp32_online
        }), (200 if manual_success else 503)

    # --------------------------------------------------------
    # MOVEMENT COMMANDS
    # --------------------------------------------------------
    if command != "STOP" and rover_mode != "MANUAL":
        return jsonify({
            "success": False,
            "message": "Switch to MANUAL mode first",
            "command": command,
            "mode": rover_mode,
            "esp32_online": esp32_online
        }), 403

    success, result = send_esp32_command(command)

    return jsonify({
        "success": success,
        "command": command,
        "mode": rover_mode,
        "esp32": result,
        "esp32_online": esp32_online
    }), (200 if success else 503)


# ============================================================
# STATUS
# ============================================================

@app.route("/status")
def status():

    # ESP32 state is refreshed by the background monitor thread.

    images = []


    if os.path.exists(
        UNKNOWN_DIR
    ):

        images = [

            f

            for f in os.listdir(
                UNKNOWN_DIR
            )

            if f.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png"
                )
            )

        ]


        images.sort(
            reverse=True
        )


    with tracking_lock:

        known_people = sum(

            1

            for person
            in tracked_people.values()

            if person["known"]

        )


    return jsonify({

        "known_count":
            known_people,

        "unknown_count":
            len(images),

        "images":
            images,

        "mode":
            rover_mode,

        "last_command":
            last_command,

        "esp32_online":
            esp32_online,

        "esp32_ip":
            ESP32_IP

    })



# ============================================================
# PREDICTIVE MAINTENANCE API
# ============================================================

@app.route("/maintenance")
def maintenance():
    return jsonify(pm_snapshot())


@app.route("/maintenance/reset", methods=["POST", "GET"])
def maintenance_reset():
    """Simulate 'service done + battery recharged'."""
    with pm_lock:
        pm_state["moving_s"] = 0.0
        pm_state["distance_m"] = 0.0
        pm_state["battery_pct"] = 100.0
        pm_state["temp_c"] = PM_AMBIENT_TEMP
    return jsonify(pm_snapshot())


# ============================================================
# START
# ============================================================

if __name__ == "__main__":


    print()
    print("======================================")
    print("      SURVEILLANCE ROVER")
    print("======================================")
    print()
    print(
        "Raspberry Pi Web Server:"
    )
    print(
        "http://<PI-IP>:5000"
    )
    print()
    print(
        "ESP32:"
    )
    print(
        f"http://{ESP32_IP}:{ESP32_PORT}"
    )
    print()
    print(
        "MODE:",
        rover_mode
    )
    print()
    print("======================================")


    # --------------------------------------------------------
    # Start camera thread
    # --------------------------------------------------------

    camera_thread = threading.Thread(
        target=camera_loop,
        daemon=True
    )

    camera_thread.start()


    # --------------------------------------------------------
    # ESP32 monitoring thread
    # --------------------------------------------------------

    esp32_thread = threading.Thread(
        target=esp32_monitor,
        daemon=True
    )

    esp32_thread.start()

    # Predictive maintenance (simulated values)
    pm_thread = threading.Thread(
        target=pm_loop,
        daemon=True
    )

    pm_thread.start()

    # Perform one immediate connection test before the web page opens.
    check_esp32()

    print(
        "ESP32 STATUS:",
        "ONLINE" if esp32_online else "OFFLINE"
    )


    # --------------------------------------------------------
    # Start Flask
    # --------------------------------------------------------

    try:

        app.run(

            host="0.0.0.0",

            port=5000,

            threaded=True,

            debug=False

        )

    finally:

        camera_running = False

        camera.release()

        print(
            "Camera released."
        )

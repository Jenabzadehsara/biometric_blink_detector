
import tkinter as tk
import serial
import time
import cv2
import threading
import numpy as np
import datetime
from PIL import Image, ImageTk
import mediapipe as mp

ser = serial.Serial('/dev/cu.usbmodem1101', 9600, timeout=1)
time.sleep(2)

BaseOptions              = mp.tasks.BaseOptions
FaceLandmarker           = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions    = mp.tasks.vision.FaceLandmarkerOptions
GestureRecognizer        = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode        = mp.tasks.vision.RunningMode

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)

hand_options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

FACE_TESSELATION = [
    (127,34),(34,139),(139,127),(11,0),(0,37),(37,11),(232,231),(231,120),
    (120,232),(72,37),(37,39),(39,72),(128,121),(121,47),(47,128),(232,121),
    (121,128),(128,232),(104,69),(69,67),(67,104),(175,171),(171,148),(148,175),
    (157,154),(154,155),(155,157),(118,50),(50,101),(101,118),(73,39),(39,40),
    (40,73),(9,151),(151,108),(108,9),(48,115),(115,131),(131,48),(194,204),
    (204,211),(211,194),(74,40),(40,185),(185,74),(80,42),(42,183),(183,80),
    (40,92),(92,186),(186,40),(230,229),(229,118),(118,230),(202,212),(212,214),
    (214,202),(83,18),(18,17),(17,83),(76,61),(61,146),(146,76),(160,29),
    (29,30),(30,160),(56,157),(157,173),(173,56),(106,204),(204,194),(194,106),
    (135,214),(214,192),(192,135),(203,165),(165,98),(98,203),(21,71),(71,68),
    (68,21),(51,45),(45,4),(4,51),(144,24),(24,23),(23,144),(77,146),(146,91),
    (91,77),(205,50),(50,36),(36,205),(143,57),(57,202),(202,143),(123,116),
    (116,198),(198,123),(213,215),(215,138),(138,213),(59,166),(166,219),
    (219,59),(60,75),(75,60),(236,3),(3,51),(51,236),(207,205),(205,36),
    (36,207),(216,206),(206,205),(205,216),(214,135),(135,192),(192,214),
    (215,58),(58,172),(172,215),(115,48),(48,219),(219,115),(42,80),(80,81),
    (81,42),(195,3),(3,236),(236,195),(144,163),(163,161),(161,144),(178,79),
    (79,214),(214,178),(159,145),(145,144),(144,159),(160,161),(161,246),
    (246,160),(193,3),(3,168),(168,193),(47,100),(100,126),(126,47),(206,165),
    (165,209),(209,206),(126,100),(100,142),(142,126),(123,147),(147,187),
    (187,123),(32,31),(31,228),(228,32),(226,35),(35,111),(111,226),(101,50),
    (50,205),(205,101),(203,98),(98,97),(97,203),(45,51),(51,134),(134,45),
    (234,127),(127,162),(162,234),(21,54),(54,103),(103,21),(44,19),(19,141),
    (141,44),(128,47),(47,121),(121,128),(104,105),(105,69),(69,104),(193,168),
    (168,8),(8,193),(117,228),(228,31),(31,117),(189,193),(193,55),(55,189),
    (98,141),(141,19),(19,98),(197,196),(196,175),(175,197),(184,129),(129,115),
    (115,184),(188,222),(222,245),(245,188),(32,228),(228,229),(229,32),
    (130,114),(114,226),(226,130),(6,197),(197,419),(419,6),(8,1),(1,168),
    # right side
    (356,389),(389,368),(368,356),(10,338),(338,297),(297,10),(454,356),
    (356,365),(365,454),(288,397),(397,361),(361,288),(426,427),(427,360),
    (360,426),(376,352),(352,345),(345,376),(411,376),(376,433),(433,411),
    (453,452),(452,357),(357,453),(333,298),(298,301),(301,333),(251,389),
    (389,385),(385,251),(276,353),(353,383),(383,276),(308,324),(324,325),
    (325,308),(297,338),(338,377),(377,297),(346,347),(347,280),(280,346),
    (394,395),(395,379),(379,394),(399,412),(412,419),(419,399),(410,436),
    (436,416),(416,410),(434,432),(432,430),(430,434),(422,430),(430,394),
    (394,422),(366,401),(401,371),(371,366),(395,378),(378,379),(379,395),
    (412,399),(399,411),(411,412),(363,440),(440,457),(457,363),(371,266),
    (266,425),(425,371),(423,391),(391,327),(327,423),(358,279),(279,420),
    (420,358),(279,331),(331,420),(420,279),(446,342),(342,353),(353,446),
    (424,422),(422,430),(430,424),(391,423),(423,327),(327,391),(365,356),
    (356,454),(454,365),(280,347),(347,330),(330,280),(269,303),(303,270),
    (270,269),(303,271),(271,304),(304,303),(267,269),(269,270),(270,267),
    (272,304),(304,408),(408,272),(394,430),(430,431),(431,394),(395,369),
    (369,378),(378,395),(400,296),(296,334),(334,400),(386,387),(387,388),
    (388,386),(418,424),(424,406),(406,418),(367,416),(416,435),(435,367),
    (364,394),(394,367),(367,364),(435,401),(401,367),(367,435),(391,269),
    (269,322),(322,391),(417,465),(465,464),(464,417),(386,260),(260,380),
    (380,386),(381,382),(382,362),(362,381),(408,272),(272,271),(271,408),
    (395,394),(394,431),(431,395),(397,365),(365,288),(288,397),(384,398),
    (398,362),(362,384),(347,348),(348,330),(330,347),(303,304),(304,270),
    (270,303),(9,336),(336,151),(151,9),(344,440),(440,275),(275,344),
    (263,249),(249,390),(390,263),(466,263),(263,388),(388,466),(387,466),
    (466,260),(260,387),(456,341),(341,463),(463,456),(452,350),(350,357),
    (357,452),(464,453),(453,465),(465,464),(343,277),(277,355),(355,343),
    (452,453),(453,350),(350,452),(383,276),(276,293),(293,383),(282,295),
    (295,283),(283,282),(407,408),(408,304),(304,407),(321,405),(405,320),
    (320,321),(404,406),(406,405),(405,404),(423,426),(426,391),(391,423),
    (429,355),(355,437),(437,429),(391,327),(327,393),(393,391),(438,439),
    (439,344),(344,438),(277,343),(343,437),(437,277),(443,444),(444,445),
    (445,443),(342,446),(446,467),(467,342),(466,464),(464,465),(465,466),
]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

blink_cooldown   = False
latest_frame     = None
blink_count      = 0
recording        = False
out              = None
thumbs_up_active = False
peace_active     = False

def eye_aspect_ratio(landmarks, eye_indices):
    pts = [(landmarks[i].x, landmarks[i].y) for i in eye_indices]
    A = ((pts[1][0]-pts[5][0])**2 + (pts[1][1]-pts[5][1])**2) ** 0.5
    B = ((pts[2][0]-pts[4][0])**2 + (pts[2][1]-pts[4][1])**2) ** 0.5
    C = ((pts[0][0]-pts[3][0])**2 + (pts[0][1]-pts[3][1])**2) ** 0.5
    return (A + B) / (2.0 * C)

def handle_gestures(hand_result):
    global thumbs_up_active, peace_active
    thumbs_up_active = False
    peace_active     = False
    if not hand_result or not hand_result.gestures:
        ser.write(b'0')
        return
    for gesture_list in hand_result.gestures:
        if gesture_list:
            name = gesture_list[0].category_name
            if name == "Thumb_Up":
                thumbs_up_active = True
            elif name == "Victory":
                peace_active = True
    if thumbs_up_active:
        ser.write(b'2')
    else:
        ser.write(b'0')

def draw_hand(frame, hand_landmarks, h, w):
    HAND_LINE = (30, 30, 160)
    HAND_DOT  = (50, 50, 220)
    for (a, b) in HAND_CONNECTIONS:
        x1 = int(hand_landmarks[a].x * w)
        y1 = int(hand_landmarks[a].y * h)
        x2 = int(hand_landmarks[b].x * w)
        y2 = int(hand_landmarks[b].y * h)
        cv2.line(frame, (x1,y1), (x2,y2), HAND_LINE, 2)
    for lm in hand_landmarks:
        px, py = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (px,py), 4, HAND_DOT, -1)
        cv2.circle(frame, (px,py), 4, (80,80,255), 1)

def draw_hud(frame, face_landmarks, eyes_open, ear_val, hand_result):
    h, w       = frame.shape[:2]
    RED_BRIGHT = (50, 50, 255)
    RED_DIM    = (20, 20, 100)
    GREEN      = (80, 200, 80)
    BLINK_RED  = (60, 60, 220)
    DOT        = (60, 60, 200)

    # corner brackets
    size = 55
    for (cx, cy, dx, dy) in [(0,0,1,1),(w,0,-1,1),(0,h,1,-1),(w,h,-1,-1)]:
        cv2.line(frame, (cx,cy), (cx+dx*size, cy), RED_BRIGHT, 2)
        cv2.line(frame, (cx,cy), (cx, cy+dy*size), RED_BRIGHT, 2)

    # top bar
    cv2.rectangle(frame, (0,0), (w,28), (8,5,5), -1)
    cv2.putText(frame, "BIOMETRIC SCAN ACTIVE", (12,19),
                cv2.FONT_HERSHEY_PLAIN, 1.1, RED_BRIGHT, 1)
    cv2.circle(frame, (w-18,14), 6, GREEN if eyes_open else BLINK_RED, -1)
    if recording:
        cv2.circle(frame, (w-50,14), 6, (0,0,255), -1)
        cv2.putText(frame, "REC", (w-42,19),
                    cv2.FONT_HERSHEY_PLAIN, 0.85, (0,0,255), 1)
    if thumbs_up_active:
        cv2.putText(frame, "THUMBS UP", (w//2-50,19),
                    cv2.FONT_HERSHEY_PLAIN, 0.85, (0,200,0), 1)

    # bottom bar
    cv2.rectangle(frame, (0,h-28), (w,h), (8,5,5), -1)
    cv2.putText(frame, "EYES OPEN" if eyes_open else "BLINK DETECTED",
                (w//2-80, h-8), cv2.FONT_HERSHEY_PLAIN, 1.1,
                GREEN if eyes_open else BLINK_RED, 1)

    # hands
    hand_count = 0
    if hand_result and hand_result.hand_landmarks:
        hand_count = len(hand_result.hand_landmarks)
        for hand_lms in hand_result.hand_landmarks:
            draw_hand(frame, hand_lms, h, w)

    if face_landmarks is None:
        cv2.putText(frame, "NO FACE DETECTED", (w//2-100, h//2),
                    cv2.FONT_HERSHEY_PLAIN, 1.4, BLINK_RED, 1)
        return frame

    pts = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks]

    # full tesselation both sides
    for (a, b) in FACE_TESSELATION:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], RED_DIM, 1)

    # all dots
    for (px, py) in pts:
        cv2.circle(frame, (px,py), 1, DOT, -1)

    # face bounding corners
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    fx, fy  = min(xs)-15, min(ys)-15
    fw, fh_ = max(xs)-fx+15, max(ys)-fy+15

    for (px, py, dx, dy) in [(fx,fy,1,1),(fx+fw,fy,-1,1),
                              (fx,fy+fh_,1,-1),(fx+fw,fy+fh_,-1,-1)]:
        cv2.line(frame, (px,py), (px+dx*25, py), RED_BRIGHT, 2)
        cv2.line(frame, (px,py), (px, py+dy*25), RED_BRIGHT, 2)

    # reticle
    cx, cy = fx+fw//2, fy+fh_//2
    cv2.circle(frame, (cx,cy), 5, RED_BRIGHT, 1)
    for (x1,y1,x2,y2) in [(cx-18,cy,cx-7,cy),(cx+7,cy,cx+18,cy),
                           (cx,cy-18,cx,cy-7),(cx,cy+7,cx,cy+18)]:
        cv2.line(frame, (x1,y1), (x2,y2), RED_BRIGHT, 1)

    # left panel
    for i, label in enumerate([f"EAR   {ear_val:.2f}", f"BLINKS {blink_count}",
                                "STATUS  OK", "FACE  LOCK", f"PTS  {len(pts)}"]):
        py = fy + 20 + i*32
        if 0 < py < h:
            cv2.line(frame, (fx,py), (max(fx-35,0),py), RED_DIM, 1)
            cv2.rectangle(frame, (2,py-12), (112,py+10), (8,5,5), -1)
            cv2.rectangle(frame, (2,py-12), (112,py+10), RED_DIM, 1)
            cv2.putText(frame, label, (5,py+5),
                        cv2.FONT_HERSHEY_PLAIN, 0.85, RED_BRIGHT, 1)

    # right panel
    gesture_label = "THUMB UP" if thumbs_up_active else (
                    "PEACE" if peace_active else "NO GESTURE")
    for i, label in enumerate(["MODEL  MP", f"W  {fw}px",
                                f"HANDS  {hand_count}", gesture_label]):
        py = fy + 20 + i*32
        if 0 < py < h:
            cv2.line(frame, (fx+fw,py), (min(fx+fw+35,w),py), RED_DIM, 1)
            cv2.rectangle(frame, (w-115,py-12), (w-2,py+10), (8,5,5), -1)
            cv2.rectangle(frame, (w-115,py-12), (w-2,py+10), RED_DIM, 1)
            cv2.putText(frame, label, (w-112,py+5),
                        cv2.FONT_HERSHEY_PLAIN, 0.85, RED_BRIGHT, 1)

    # YAY popup
        if peace_active:
            text = "Yay!"
            scale = 1.0
            thickness = 2
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness)
            tx, ty = 40, 80
            cv2.rectangle(frame, (tx - 8, ty - th - 8), (tx + tw + 8, ty + 8), (5, 5, 5), -1)
            cv2.rectangle(frame, (tx - 8, ty - th - 8), (tx + tw + 8, ty + 8), (50, 50, 255), 2)
            cv2.putText(frame, text, (tx, ty),
                        cv2.FONT_HERSHEY_DUPLEX, scale, (50, 50, 255), thickness)

    return frame

def trigger_blink():
    global blink_cooldown, blink_count
    if not blink_cooldown:
        blink_cooldown = True
        blink_count += 1
        ser.write(b'1')
        canvas.itemconfig(pupil, fill='#0a0000')
        root.after(150, lambda: canvas.itemconfig(pupil, fill='#ff4444'))
        root.after(500, reset_cooldown)

def reset_cooldown():
    global blink_cooldown
    blink_cooldown = False

def toggle_record():
    global recording, out
    if not recording:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"/Users/sarajenabzadeh/Desktop/blink_recording_{timestamp}.mp4"
        fourcc    = cv2.VideoWriter_fourcc(*'mp4v')
        out       = cv2.VideoWriter(filename, fourcc, 20.0, (1280, 720))
        recording = True
        record_btn.config(text="⏹ STOP", fg='#ff0000')
    else:
        recording = False
        out.release()
        out = None
        record_btn.config(text="⏺ REC", fg='#ff4444')

def camera_loop():
    global latest_frame, out
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    was_blink = False

    with FaceLandmarker.create_from_options(face_options) as landmarker, \
         GestureRecognizer.create_from_options(hand_options) as hand_recognizer:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result      = landmarker.detect(mp_image)
            hand_result = hand_recognizer.recognize(mp_image)

            handle_gestures(hand_result)

            face_landmarks = None
            eyes_open      = True
            ear_val        = 0.0

            if result.face_landmarks:
                face_landmarks = result.face_landmarks[0]
                left_ear  = eye_aspect_ratio(face_landmarks, LEFT_EYE)
                right_ear = eye_aspect_ratio(face_landmarks, RIGHT_EYE)
                ear_val   = (left_ear + right_ear) / 2.0
                eyes_open = ear_val >= 0.25
                if not eyes_open and not was_blink:
                    root.after(0, trigger_blink)
                was_blink = not eyes_open

            frame = draw_hud(frame, face_landmarks, eyes_open, ear_val, hand_result)

            if recording and out is not None:
                out.write(frame)

            latest_frame = frame

    cap.release()

def update_feed():
    global latest_frame
    if latest_frame is not None:
        rgb       = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
        img       = Image.fromarray(rgb)
        sw        = root.winfo_screenwidth()
        display_h = int(sw * 720 / 1280)
        img       = img.resize((sw, display_h))
        imgtk     = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)
    root.after(30, update_feed)

thread = threading.Thread(target=camera_loop, daemon=True)
thread.start()

root = tk.Tk()
root.title("Biometric Scan")
root.attributes('-fullscreen', True)
root.config(bg='#000000')

sw  = root.winfo_screenwidth()
mid = sw // 2

camera_label = tk.Label(root, bg='#000000')
camera_label.pack()

canvas = tk.Canvas(root, width=sw, height=80, bg='#000000', highlightthickness=0)
canvas.place(relx=0.5, rely=0.88, anchor='center')
canvas.create_oval(mid-60, 5, mid+60, 75, outline='#3a0000', width=1)
pupil = canvas.create_oval(mid-25, 20, mid+25, 60, fill='#ff4444')

btn_frame = tk.Frame(root, bg='#000000')
btn_frame.place(relx=0.5, rely=0.95, anchor='center')

btn = tk.Button(btn_frame, text="BLINK", font=('Courier', 16, 'bold'),
                command=trigger_blink,
                bg='#000000', fg='#ff4444',
                activebackground='#ff4444', activeforeground='#000000',
                relief='flat', bd=0, highlightthickness=0,
                cursor='hand2')
btn.pack(side='left', padx=30)

record_btn = tk.Button(btn_frame, text="⏺ REC", font=('Courier', 16, 'bold'),
                command=toggle_record,
                bg='#000000', fg='#ff4444',
                activebackground='#ff4444', activeforeground='#000000',
                relief='flat', bd=0, highlightthickness=0,
                cursor='hand2')
record_btn.pack(side='left', padx=30)

esc_label = tk.Label(root, text="press ESC to exit",
                     fg='#3a0000', bg='#000000', font=('Courier', 9))
esc_label.place(relx=0.5, rely=0.98, anchor='center')

root.bind('<Escape>', lambda e: root.destroy())

root.after(30, update_feed)
root.mainloop()
ser.close()
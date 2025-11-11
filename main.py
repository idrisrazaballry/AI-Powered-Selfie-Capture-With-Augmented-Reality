# ================== AI SELFIE CAMERA (FINAL FIXED - STRICT SMILE ONLY) ==================
import os, cv2, math, time, numpy as np, threading, queue, datetime, tkinter as tk
import mediapipe as mp, warnings, speech_recognition as sr
from PIL import Image, ImageTk

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
#  FIX: Robust Filter Imports (Prevents all filters from failing if one is missing)
# --------------------------------------------------------------------------

def create_passthrough(filter_name):
    """Creates a dummy function that just returns the image."""
    def passthrough_filter(img):
        print(f"Warning: Filter '{filter_name}' failed to load or is missing.")
        return img
    
    # Set the function name for the GUI
    passthrough_filter.__name__ = f"apply_{filter_name}"
    return passthrough_filter

try:
    from glass import apply_glass
except ImportError:
    apply_glass = create_passthrough("glass")

try:
    from mustache import apply_mustache
except ImportError:
    apply_mustache = create_passthrough("mustache")

try:
    from oil_paint import apply_oilpaint
except ImportError:
    apply_oilpaint = create_passthrough("oil_paint")

try:
    from dog_filter import apply_dogfilter
except ImportError:
    apply_dogfilter = create_passthrough("dog_filter")

# NOTE: Keeping fire_eyes import since it was in your latest uploaded main.py
try:
    from fire_eyes import apply_fire_eyes
except ImportError:
    apply_fire_eyes = create_passthrough("fire_eyes")

try:
    from brightener import apply_brighten
except ImportError:
    apply_brighten = create_passthrough("brighten")

try:
    from bw import apply_bw
except ImportError:
    apply_bw = create_passthrough("bw")

try:
    from blur import apply_blur
except ImportError:
    apply_blur = create_passthrough("blur")

# --------------------------------------------------------------------------
# ---------- Mediapipe Setup ----------
# --------------------------------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------- Constants ----------
LEFT_EYE=[362,385,387,263,373,380]
RIGHT_EYE=[33,160,158,133,153,144]
MOUTH_L,MOUTH_R,MOUTH_T,MOUTH_B=61,291,13,14

SMILE_HOLD=0.3
COOLDOWN=3
BLINK_FRAMES=2
BLINK_THRESHOLD=0.25

# New thresholds for the h/w ratio
SMILE_THRESHOLD = 0.05     
SMILE_BAR_MIN = 0.15       
SMILE_BAR_MAX = 0.07       

# ---------- Globals ----------
smile_start=0
blink_counter=0
last_capture_time=0
capture_mode=None
current_filter=lambda i:i
preview_label=None
preview_size=(320,240)
voice_queue,voice_stop_event=queue.Queue(),threading.Event()
recognizer=sr.Recognizer()
voice_active=False

# ---------- Helper Functions ----------
def dist(a,b): return math.dist((a.x,a.y,a.z),(b.x,b.y,b.z))
def curve(lm):
    l,r,t,b=lm[MOUTH_L],lm[MOUTH_R],lm[MOUTH_T],lm[MOUTH_B]
    w,h=dist(l,r),dist(t,b)
    return h/w if w>0 else 0 
def ear(lm,idx):
    p1,p2,p3,p4,p5,p6=[lm[i] for i in idx]
    A,B,C=dist(p2,p6),dist(p3,p5),dist(p1,p4)
    return (A+B)/(2*C) if C>0 else 0

# ---------- Voice Listener ----------
def voice_listener():
    global voice_active
    try:
        with sr.Microphone() as src:
            recognizer.adjust_for_ambient_noise(src,2)
            keys=("hey sefi","take selfie")
            while not voice_stop_event.is_set():
                if not voice_active: time.sleep(0.5); continue
                try:
                    audio=recognizer.listen(src,timeout=8,phrase_time_limit=5)
                    txt=recognizer.recognize_google(audio,language="en-IN").lower()
                    if any(k in txt for k in keys):
                        voice_queue.put("TAKE")
                        print(" Voice Triggered:",txt)
                except: continue
    except: print(" Mic error")

# ---------- Save & Preview ----------
def show_preview(frm):
    if not preview_label: return
    h,w=frm.shape[:2]
    s=min(preview_size[0]/w,preview_size[1]/h)
    rgb=cv2.cvtColor(cv2.resize(frm,(int(w*s),int(h*s))),cv2.COLOR_BGR2RGB)
    imgtk=ImageTk.PhotoImage(Image.fromarray(rgb))
    preview_label.imgtk=imgtk
    preview_label.config(image=imgtk)

def save_img(frm):
    global last_capture_time
    name=f"captured_{datetime.datetime.now():%Y%m%d_%H%M%S}.jpg"
    cv2.imwrite(name,frm)
    last_capture_time=time.time()
    print(" Saved:",name)
    show_preview(frm)

# ---------- Frame Update ----------
def update(root,cap,video,flt_lbl,stat_lbl):
    global smile_start,blink_counter,last_capture_time
    ok,fr=cap.read()
    if not ok:
        root.after(10,lambda:update(root,cap,video,flt_lbl,stat_lbl));return
    fr=cv2.flip(fr,1)
    res=face_mesh.process(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB))
    lm=res.multi_face_landmarks[0].landmark if res.multi_face_landmarks else None
    img=current_filter(fr.copy())
    mode=capture_mode.get()
    status=f"Mode:{mode}"

    now=time.time()
    cooldown_active = (now - last_capture_time) < COOLDOWN

    # Reset detection flags safely
    smile_trigger=False
    blink_trigger=False
    voice_trigger=False

    if lm:
        e=(ear(lm,LEFT_EYE)+ear(lm,RIGHT_EYE))/2
        c=curve(lm)
        if mode=="SMILE":
            # Updated smile bar logic with new h/w min/max
            bar_ratio = min(1, max(0, (c - SMILE_BAR_MIN) / (SMILE_BAR_MAX - SMILE_BAR_MIN)))
            cv2.rectangle(img,(10,50),(210,70),(50,50,50),-1)
            cv2.rectangle(img,(10,50),(10+int(200*bar_ratio),70),(0,255,0),-1)
            cv2.putText(img,"Smile",(220,65),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
            
            # This logic (c >= threshold) now works correctly
            if c >= SMILE_THRESHOLD and not cooldown_active:
                if smile_start==0: smile_start=time.time()
                elif time.time()-smile_start>SMILE_HOLD:
                    smile_trigger=True
                    smile_start=0
            else:
                smile_start=0

        elif mode=="BLINK":
            if e<BLINK_THRESHOLD: blink_counter+=1
            else: blink_counter=0
            if blink_counter>=BLINK_FRAMES and not cooldown_active:
                blink_trigger=True
                blink_counter=0

    if mode=="VOICE" and not voice_queue.empty():
        cmd=voice_queue.get_nowait()
        if cmd=="TAKE" and not cooldown_active:
            voice_trigger=True

    # Capture only if new valid trigger
    if (smile_trigger or blink_trigger or voice_trigger) and not cooldown_active:
        save_img(img)
        status=" Captured!"
    elif cooldown_active:
        remaining=int(COOLDOWN-(now-last_capture_time))
        status=f"Cooldown:{remaining}s | {mode}"

    cv2.putText(img,status,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
    rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    imgtk=ImageTk.PhotoImage(Image.fromarray(rgb))
    video.imgtk=imgtk
    video.config(image=imgtk)
    flt_lbl.config(text=f"Filter:{current_filter.__name__.replace('apply_','').title()}")
    stat_lbl.config(text=status)
    root.after(10,lambda:update(root,cap,video,flt_lbl,stat_lbl))

# ---------- GUI ----------
def run():
    global capture_mode,current_filter,preview_label,voice_active
    root=tk.Tk()
    root.title("AI Selfie Camera (Final Fixed)")
    capture_mode=tk.StringVar(value="NONE")

    threading.Thread(target=voice_listener,daemon=True).start()
    
    # --------------------------------------------------------------------------
    # Robust Camera Initialization 
    # --------------------------------------------------------------------------
    try:
        cap=cv2.VideoCapture(0,cv2.CAP_DSHOW) 
    except AttributeError:
        cap=cv2.VideoCapture(0)

    if not cap.isOpened():
        cap=cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("FATAL ERROR: Could not open the default camera (index 0). Check camera drivers or permissions.")
        return 
    
    cap.set(cv2.CAP_PROP_FPS,30)
    # --------------------------------------------------------------------------
    
    main=tk.Frame(root)
    main.pack(padx=8,pady=8)
    left=tk.Frame(main)
    left.grid(row=0,column=0)
    video=tk.Label(left)
    video.pack(pady=6)
    right=tk.Frame(main)
    right.grid(row=0,column=1,padx=10,sticky="n")

    flt_lbl=tk.Label(right,text="Filter:None",font=("Arial",12))
    flt_lbl.pack(pady=(0,4))
    stat_lbl=tk.Label(right,text="Mode:NONE",font=("Arial",12))
    stat_lbl.pack(pady=(0,10))

    def setf(f):
        global current_filter
        current_filter=f
        flt_lbl.config(text=f"Filter:{f.__name__.replace('apply_','').title()}")

    ff=tk.LabelFrame(right,text="1) Filters",padx=5,pady=5)
    ff.pack(padx=4,pady=6,fill="x")
    for t,f in [("Glasses",apply_glass),("Mustache",apply_mustache),
                ("Oil Paint",apply_oilpaint),("Dog Face",apply_dogfilter),
                ("Fire Eyes",apply_fire_eyes),("Brighten",apply_brighten),
                ("B & W",apply_bw),("Blur",apply_blur),("Clear",lambda i:i)]:
        tk.Button(ff,text=t,command=lambda f=f:setf(f)).pack(side=tk.LEFT,padx=3)

    def set_mode(m):
        global voice_active,blink_counter,smile_start
        capture_mode.set(m)
        blink_counter=0
        smile_start=0
        voice_active=(m=="VOICE")
        print("\nMode:",m) 
        stat_lbl.config(text=f"Mode:{m}")

    mf=tk.LabelFrame(right,text="2) Capture Mode",padx=5,pady=5)
    mf.pack(padx=4,pady=6,fill="x")
    for t,m in [("Smile","SMILE"),("Blink","BLINK"),("Voice","VOICE")]:
        tk.Button(mf,text=t,command=lambda m=m:set_mode(m)).pack(side=tk.LEFT,padx=4)
    tk.Button(mf,text="Manual",command=lambda:save_img(current_filter(cv2.flip(cap.read()[1],1)))).pack(side=tk.LEFT,padx=4)
    tk.Button(mf,text="Disable",command=lambda:set_mode("NONE")).pack(side=tk.LEFT,padx=4)

    pf=tk.LabelFrame(right,text="Last Captured Selfie",padx=5,pady=5)
    pf.pack(padx=4,pady=10,fill="both")
    ph=ImageTk.PhotoImage(Image.new("RGB",preview_size,color=(30,30,30)))
    p=tk.Label(pf,image=ph)
    p.pack()
    p.imgtk=ph
    preview_label=p

    update(root,cap,video,flt_lbl,stat_lbl)

    def close():
        global voice_active
        voice_active=False
        voice_stop_event.set()
        cap.release()
        cv2.destroyAllWindows()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW",close)
    root.mainloop()

if __name__=="__main__":
    run()

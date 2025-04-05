import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
import tempfile

# --- Page Configuration (Set this first) ---
st.set_page_config(
    page_title="AI League Scout Eye",
    layout="wide",
    initial_sidebar_state="collapsed", # No sidebar needed for this design
)

# --- Global Variables & Model Loading ---
# Load models only once using caching for efficiency
@st.cache_resource
def load_models():
    try:
        pose_model = YOLO('yolov8n-pose.pt')
        ball_model = YOLO('yolov8n.pt')
        return pose_model, ball_model
    except Exception as e:
        st.error(f"Error loading YOLO models: {e}")
        st.error("Please ensure 'yolov8n-pose.pt' and 'yolov8n.pt' are in the same directory as app.py.")
        return None, None

pose_model, ball_model = load_models()

# --- CSS Styling ---
# Approximates the look of the original image
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #1a1a4a; /* Dark blue background */
        color: white; /* Default text color */
    }

    /* Style buttons to be more subtle */
    .stButton>button {
        background-color: transparent;
        color: white;
        border: 1px solid transparent; /* Make border initially invisible */
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 18px; /* Adjust as needed */
        margin: 10px 5px;
        cursor: pointer;
        transition: background-color 0.3s ease, border-color 0.3s ease;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: rgba(255, 255, 255, 0.1); /* Slight highlight on hover */
        border: 1px solid rgba(255, 255, 255, 0.3); /* Subtle border on hover */
    }
    .stButton>button:active {
        background-color: rgba(255, 255, 255, 0.2); /* Slightly darker highlight when clicked */
    }

    /* Center content */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        display: flex;
        flex-direction: column;
        align-items: center; /* Center horizontally */
    }

    /* Style for the main title and slogan */
    .title-text {
        font-size: 3em; /* Larger font size */
        font-weight: bold;
        color: white;
        text-align: center;
        margin-bottom: 0.5em;
    }
    .slogan-text {
        font-size: 1.8em; /* Slightly smaller font size */
        font-weight: bold;
        color: white;
        text-align: center;
        margin-bottom: 2em; /* Space below slogan */
        direction: rtl; /* Right-to-left text */
    }

    /* Style for section headers */
     h2 {
        color: #c8a2c8; /* Lilac color for headers */
        text-align: center;
        margin-top: 1.5em;
     }

</style>
""", unsafe_allow_html=True)


# --- Skill 1: Jumping with Ball (Copied from your example) ---
def detect_ball_knee_contacts(video_path, frame_skip=2, distance_thresh=40, angle_thresh=60):
    """
    Detects contacts between the ball and knee in a video using YOLO models.

    Args:
        video_path (str): Path to the video file.
        frame_skip (int): Process every Nth frame.
        distance_thresh (int): Max distance between ball center and knee keypoint.
        angle_thresh (int): Max knee angle (hip-knee-ankle) for a valid touch.

    Returns:
        int: Score (0-5) based on the number of successful touches detected.
    """
    if not pose_model or not ball_model:
        st.error("Models not loaded. Cannot process video.")
        return 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error opening video file: {video_path}")
        return 0

    frame_idx = 0
    touch_count = 0
    # successful_touches = [] # Keep track if detailed info is needed later

    # Placeholder for progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = total_frames // frame_skip if frame_skip > 0 else total_frames

    processed_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # End of video

        # Skip frames
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        try:
            # 1. Detect Ball
            ball_results = ball_model(frame, verbose=False)[0] # verbose=False to silence YOLO output
            ball_pos = None
            for box, cls in zip(ball_results.boxes.xyxy, ball_results.boxes.cls):
                # Class ID 32 typically corresponds to 'sports ball' in COCO dataset used by yolov8n.pt
                if int(cls) == 32:
                    x1, y1, x2, y2 = map(int, box)
                    ball_pos = ((x1 + x2) // 2, (y1 + y2) // 2)
                    # Optional: Draw ball bounding box (for debugging)
                    # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # cv2.circle(frame, ball_pos, 5, (0, 255, 0), -1)
                    break # Assume only one ball relevant

            # 2. Detect Pose
            pose_results = pose_model(frame, verbose=False)[0] # verbose=False

            # 3. Check Conditions if both ball and pose detected
            if ball_pos and pose_results.keypoints is not None and pose_results.keypoints.xy.shape[0] > 0:
                keypoints = pose_results.keypoints.xy.cpu().numpy()[0] # Get keypoints for the first detected person

                # Ensure enough keypoints are detected (YOLOv8-pose has 17 keypoints)
                # Indices: 11=right_hip, 13=right_knee, 15=right_ankle
                # Indices: 12=left_hip, 14=left_knee, 16=left_ankle
                if keypoints.shape[0] >= 17:
                    # Check right leg
                    r_hip = keypoints[11]
                    r_knee = keypoints[13]
                    r_ankle = keypoints[15]
                     # Check left leg
                    l_hip = keypoints[12]
                    l_knee = keypoints[14]
                    l_ankle = keypoints[16]

                    # --- Define helper function for angle calculation ---
                    def calculate_angle(a, b, c):
                        # Ensure points are valid (non-zero coordinates)
                        if np.all(a) and np.all(b) and np.all(c):
                            a, b, c = np.array(a), np.array(b), np.array(c)
                            # Calculate vectors BA and BC
                            ba = a - b
                            bc = c - b
                            # Calculate dot product
                            dot = np.dot(ba, bc)
                            # Calculate magnitudes
                            mag_ba = np.linalg.norm(ba)
                            mag_bc = np.linalg.norm(bc)
                            # Avoid division by zero
                            if mag_ba == 0 or mag_bc == 0:
                                return 180.0 # Or some default invalid angle
                            # Calculate cosine of the angle
                            cos_angle = dot / (mag_ba * mag_bc)
                            # Clip value to [-1, 1] to avoid domain errors with arccos
                            cos_angle = np.clip(cos_angle, -1.0, 1.0)
                            # Calculate angle in degrees
                            angle_rad = np.arccos(cos_angle)
                            angle_deg = np.degrees(angle_rad)
                            return angle_deg
                        else:
                            return 180.0 # Indicate invalid points

                    # Calculate knee angles
                    r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
                    l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)

                    # Optional: Draw keypoints (for debugging)
                    # for i, pt in enumerate([r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle]):
                    #    if np.all(pt): cv2.circle(frame, tuple(map(int, pt)), 5, (0, 0, 255), -1)

                    # Check distance and angle for right knee
                    if np.all(r_knee):
                        dist_r = np.linalg.norm(np.array(ball_pos) - r_knee)
                        if dist_r < distance_thresh and r_knee_angle < angle_thresh:
                            touch_count += 1
                            # successful_touches.append({"frame": frame_idx, "leg": "right", "distance": round(dist_r, 1), "angle": round(r_knee_angle, 1)})
                            # Optional: Draw touch indication (for debugging)
                            # cv2.putText(frame, f"TOUCH R! Dist:{dist_r:.1f} Ang:{r_knee_angle:.1f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                            # To avoid double counting in the same frame if ball is near both knees, break after first touch found
                            processed_frame_count += 1
                            frame_idx += 1
                            if frames_to_process > 0:
                                progress_bar.progress(min(1.0, processed_frame_count / frames_to_process))
                            status_text.text(f"Processing frame {frame_idx}/{total_frames}... Touches: {touch_count}")
                            continue # Skip checking left knee if right knee touched

                    # Check distance and angle for left knee
                    if np.all(l_knee):
                        dist_l = np.linalg.norm(np.array(ball_pos) - l_knee)
                        if dist_l < distance_thresh and l_knee_angle < angle_thresh:
                            touch_count += 1
                            # successful_touches.append({"frame": frame_idx, "leg": "left", "distance": round(dist_l, 1), "angle": round(l_knee_angle, 1)})
                            # Optional: Draw touch indication (for debugging)
                            # cv2.putText(frame, f"TOUCH L! Dist:{dist_l:.1f} Ang:{l_knee_angle:.1f}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        except Exception as e:
            st.warning(f"Error processing frame {frame_idx}: {e}")
            # Decide whether to continue or stop on error
            # continue

        processed_frame_count += 1
        frame_idx += 1

        # Update progress
        if frames_to_process > 0:
             progress_bar.progress(min(1.0, processed_frame_count / frames_to_process))
        status_text.text(f"Processing frame {frame_idx}/{total_frames}... Touches: {touch_count}")

        # Optional: Display processed frames (can slow down significantly)
        # st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    progress_bar.progress(1.0) # Ensure progress bar reaches 100%
    status_text.text(f"Processing complete. Total touches found: {touch_count}")
    score = min(5, touch_count) # Cap score at 5
    return score


# --- Streamlit App Layout ---

# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state.page = 'home' # Default page

# --- Top Row: Logo ---
# Use columns to approximate top-left logo position
col1, col2 = st.columns([1, 5]) # Adjust ratio as needed
with col1:
    # Check if logo exists before trying to display
    if os.path.exists("assets/ai_league_logo.png"):
         st.image("assets/ai_league_logo.png", width=150)
    else:
        st.write("AI LEAGUE") # Fallback text

with col2:
    pass # Empty column for spacing

st.write("") # Add some vertical space

# --- Center Area: Main Logo, Title, Slogan ---
st.container() # Use a container to help with centering if needed
# Display Scout Eye Logo (central)
if os.path.exists("assets/scout_eye_logo.png"):
    st.image("assets/scout_eye_logo.png", width=250) # Adjust width as needed
else:
    st.markdown("<h1 style='text-align: center; color: white;'>Scout Eye</h1>", unsafe_allow_html=True) # Fallback text

# Display Title and Slogan using Markdown with custom classes for styling
st.markdown("<div class='title-text'>عين الكشاف</div>", unsafe_allow_html=True)
st.markdown("<div class='slogan-text'>نكتشف ، نحمي ، ندعم</div>", unsafe_allow_html=True)


# --- Bottom Area: Clickable Options ---
# Use columns for horizontal layout of buttons
col_b1, col_b2, col_b3 = st.columns(3)

with col_b1:
    if st.button("الشخص المناسب في المكان المناسب"): # Placeholder for the third option
        st.session_state.page = 'الشخص_المناسب'

with col_b2:
    if st.button("نجم لا يغيب"):
        st.session_state.page = 'نجم_لا_يغيب'

with col_b3:
    if st.button("إسطورة الغد"):
        st.session_state.page = 'اسطورة_الغد'


# --- Conditional Page Content ---

if st.session_state.page == 'اسطورة_الغد':
    st.markdown("---") # Separator
    st.markdown("## ⚽ إسطورة الغد - تقييم القفز بالكرة ⚽")
    st.markdown("<p style='text-align: center;'>ارفع فيديو قصير يظهر اللاعب وهو يقوم بتنطيط الكرة باستخدام الركبة أثناء القفز.</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 اختر ملف الفيديو:", type=["mp4", "avi", "mov", "mkv"], key="jump_video")

    if uploaded_file is not None:
        if pose_model and ball_model: # Check if models loaded successfully
            # Save uploaded video to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                video_path = tmp_file.name # Get the path to the temporary file

            st.info(f"⏳ جاري تحليل الفيديو: {uploaded_file.name} ... قد يستغرق هذا بعض الوقت.")

            # --- Run the analysis ---
            try:
                # Run the detection function
                score = detect_ball_knee_contacts(video_path)

                # Display the result
                st.success(f"✅ اكتمل التحليل!")
                st.metric(label="**النتيجة (الحد الأقصى 5)**", value=f"{score} / 5")
                if score == 5:
                    st.balloons()
                    st.markdown("<p style='text-align: center; font-weight: bold; color: lightgreen;'>🎉 أداء رائع! يبدو أنك أسطورة الغد فعلاً! 🎉</p>", unsafe_allow_html=True)
                elif score > 2:
                    st.markdown("<p style='text-align: center; font-weight: bold; color: yellow;'>👍 جيد جداً! استمر في التدريب.</p>", unsafe_allow_html=True)
                else:
                     st.markdown("<p style='text-align: center; font-weight: bold; color: orange;'>💪 لا بأس، التدريب يصنع المعجزات! حاول مرة أخرى.</p>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"حدث خطأ أثناء معالجة الفيديو: {e}")
            finally:
                # Clean up the temporary file
                if 'video_path' in locals() and os.path.exists(video_path):
                    os.remove(video_path)
        else:
            st.error("❌ تعذر تحميل نماذج الذكاء الاصطناعي. لا يمكن تحليل الفيديو.")

elif st.session_state.page == 'نجم_لا_يغيب':
    st.markdown("---")
    st.markdown("## ⭐ نجم لا يغيب ⭐")
    st.info("سيتم إضافة وظيفة رفع الفيديو وتحليله لهذه الميزة قريباً.")
    # Placeholder for future upload and Gemini analysis

elif st.session_state.page == 'الشخص_المناسب':
    st.markdown("---")
    st.markdown("## ✔️ الشخص المناسب في المكان المناسب ✔️")
    st.info("سيتم إضافة وظيفة رفع مجموعة البيانات وتحليلها لهذه الميزة قريباً.")
     # Placeholder for future dataset upload and Gemini analysis

# --- Footer or other elements if needed ---
# st.markdown("---")
# st.caption("AI League - Scout Eye v0.1")
import streamlit as st
import cv2
import mediapipe as mp
import time
import threading
import av
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@400;500;600&display=swap');
:root{--good:#00e676;--warn:#ffea00;--bad:#ff1744;--bg:#0d0f14;--card:#161922;--border:#1e2230;--text:#e8eaf0;--muted:#6b7280;--accent:#6c63ff}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text)!important;font-family:'DM Sans',sans-serif}
[data-testid="stSidebar"]{background:#0a0c10!important;border-right:1px solid var(--border)}
h1,h2,h3{font-family:'Space Mono',monospace}
#MainMenu,footer,header{visibility:hidden}[data-testid="stToolbar"]{display:none}
.mc{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:1rem 1.1rem;text-align:center;margin-bottom:.5rem}
.mc .lbl{font-size:.68rem;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:.3rem}
.mc .val{font-family:'Space Mono',monospace;font-size:1.75rem;font-weight:700;line-height:1}
.val.good{color:var(--good)}.val.warn{color:var(--warn)}.val.bad{color:var(--bad)}.val.wht{color:var(--text)}
.pill{display:inline-block;padding:.35rem 1.1rem;border-radius:999px;font-family:'Space Mono',monospace;font-size:.82rem;font-weight:700;margin-bottom:.8rem}
.pg{background:rgba(0,230,118,.15);color:var(--good);border:1px solid rgba(0,230,118,.3)}
.pw{background:rgba(255,234,0,.12);color:var(--warn);border:1px solid rgba(255,234,0,.3)}
.pb{background:rgba(255,23,68,.15);color:var(--bad);border:1px solid rgba(255,23,68,.3)}
.pi{background:rgba(107,114,128,.15);color:var(--muted);border:1px solid rgba(107,114,128,.3)}
.sh{font-family:'Space Mono',monospace;font-size:.68rem;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);border-bottom:1px solid var(--border);padding-bottom:.3rem;margin:1rem 0 .65rem}
.tip{background:linear-gradient(135deg,#161922 60%,#1a1f2e);border-left:3px solid var(--good);border-radius:0 8px 8px 0;padding:.7rem .9rem;font-size:.85rem;color:#b0bcc8;margin-bottom:.4rem}
.red-overlay{position:fixed;inset:0;pointer-events:none;z-index:9999;animation:pr 1s ease-in-out infinite alternate}
@keyframes pr{from{background:rgba(255,23,68,.08)}to{background:rgba(255,23,68,.26)}}
.stButton>button{border-radius:8px!important;font-weight:600!important}
</style>
""", unsafe_allow_html=True)

# ── GLOBAL SHARED STATE (module-level: video thread ↔ main thread) ───────────
if "_ps" not in st.__dict__:
    st._ps = {
        "posture":     "Waiting…",
        "baseline":    None,
        "calib_req":   False,
        "bad_thresh":  0.10,
        "warn_thresh": 0.075,
        "lock":        threading.Lock(),
    }
_S = st._ps

# ── SESSION STATE DEFAULTS ────────────────────────────────────────────────────
for k, v in [("session_active", False), ("session_start", None),
             ("total_good", 0), ("total_warn", 0), ("total_bad", 0),
             ("alert_count", 0), ("last_alert_time", 0),
             ("history", deque(maxlen=180))]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── MEDIAPIPE ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_detector():
    opts = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="pose_landmarker_lite.task"),
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.PoseLandmarker.create_from_options(opts)

detector = load_detector()

# ── VIDEO PROCESSOR ───────────────────────────────────────────────────────────
class PostureProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        try:
            rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_img)
        except Exception:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        posture = "No Pose"
        if result.pose_landmarks:
            lms = result.pose_landmarks[0]
            h, w, _ = img.shape
            l_shldr, r_shldr = lms[11], lms[12]
            l_ear,   r_ear   = lms[7],  lms[8]
            tilt = abs(l_shldr.y - r_shldr.y)
            score = (l_shldr.y + r_shldr.y) / 2 - (l_ear.y + r_ear.y) / 2

            with _S["lock"]:
                if _S["calib_req"]:
                    _S["baseline"]  = score
                    _S["calib_req"] = False
                bad_t, warn_t, baseline = _S["bad_thresh"], _S["warn_thresh"], _S["baseline"]

            if baseline is not None:
                posture = ("Bad Posture"   if score < baseline * 0.82 else
                           "Slightly Bent" if score < baseline * 0.91 else
                           "Good Posture")
            else:
                posture = ("Bad Posture"   if tilt > bad_t  else
                           "Slightly Bent" if tilt > warn_t else
                           "Good Posture")

            color = {"Good Posture":(0,210,100), "Slightly Bent":(0,210,220),
                     "Bad Posture":(0,0,220)}.get(posture, (120,120,120))

            for idx in [7, 8, 11, 12, 23, 24]:
                pt = lms[idx]
                cv2.circle(img, (int(pt.x*w), int(pt.y*h)), 5, color, -1)
            ls = (int(l_shldr.x*w), int(l_shldr.y*h))
            rs = (int(r_shldr.x*w), int(r_shldr.y*h))
            cv2.line(img, ls, rs, color, 2)

            bg = {"Bad Posture":(0,0,180),"Slightly Bent":(0,160,180),
                  "Good Posture":(0,150,60)}.get(posture,(60,60,60))
            cv2.rectangle(img, (10,10), (310,58), bg, -1)
            cv2.putText(img, posture, (18,46), cv2.FONT_HERSHEY_SIMPLEX, 1.05,
                        (255,255,255), 2, cv2.LINE_AA)

        with _S["lock"]:
            _S["posture"] = posture

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ── READ POSTURE ──────────────────────────────────────────────────────────────
with _S["lock"]:
    posture_now = _S["posture"]

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"## 🧘 PostureSense")
    st.markdown(f"👤 **{st.session_state.get('user_name','User')}**")
    st.markdown("<div class='sh'>Detection Settings</div>", unsafe_allow_html=True)

    sens = st.select_slider("Shoulder tilt sensitivity",
                            options=["Low","Medium","High"], value="Medium")
    with _S["lock"]:
        _S["bad_thresh"], _S["warn_thresh"] = \
            {"Low":(0.13,0.10),"Medium":(0.10,0.075),"High":(0.07,0.05)}[sens]

    st.markdown("<div class='sh'>Alert Settings</div>", unsafe_allow_html=True)
    voice_alerts   = st.toggle("🔊 Voice alerts",     value=True)
    red_overlay_on = st.toggle("🔴 Red screen flash", value=True)
    alert_cooldown = st.slider("Alert cooldown (sec)", 5, 60, 15)

    st.markdown("<div class='sh'>Quick Tips</div>", unsafe_allow_html=True)
    for tip in ["Keep shoulders level and relaxed.",
                "Ears should align over shoulders.",
                "Sit with hips pushed back in chair.",
                "Screen at eye level, arm's length away.",
                "Take a break every 30 minutes."]:
        st.markdown(f"<div class='tip'>💡 {tip}</div>", unsafe_allow_html=True)

# ── UPDATE COUNTERS ───────────────────────────────────────────────────────────
if st.session_state.session_active and posture_now not in ("Waiting…","No Pose"):
    now = time.time()
    if posture_now == "Good Posture":    st.session_state.total_good += 1
    elif posture_now == "Slightly Bent": st.session_state.total_warn += 1
    elif posture_now == "Bad Posture":   st.session_state.total_bad  += 1
    st.session_state.history.append(posture_now)
    if posture_now == "Bad Posture":
        if now - st.session_state.last_alert_time > alert_cooldown:
            st.session_state.last_alert_time = now
            st.session_state.alert_count    += 1

# ── VOICE ALERT (JS SpeechSynthesis) ─────────────────────────────────────────
st.markdown("""
<div id="_ps_vtrig" data-key="0"></div>
<script>(function(){
  var last="0";
  function fire(el){var k=el.getAttribute("data-key");if(k&&k!==last){last=k;
    window.speechSynthesis.cancel();
    var u=new SpeechSynthesisUtterance("Bad posture detected! Please sit up straight.");
    u.rate=0.95;u.pitch=1;u.volume=1;window.speechSynthesis.speak(u);}}
  var el=document.getElementById("_ps_vtrig");
  if(el)new MutationObserver(function(){fire(el);}).observe(el,{attributes:true});
  else document.addEventListener("DOMContentLoaded",function(){
    var e2=document.getElementById("_ps_vtrig");
    if(e2)new MutationObserver(function(){fire(e2);}).observe(e2,{attributes:true});});
})();</script>
""", unsafe_allow_html=True)

if voice_alerts and st.session_state.session_active and st.session_state.alert_count > 0:
    ak = str(st.session_state.alert_count)
    st.markdown(f"""<script>(function(){{
      var el=document.getElementById("_ps_vtrig");
      if(el)el.setAttribute("data-key","{ak}");
    }})();</script>""", unsafe_allow_html=True)

# ── MAIN LAYOUT ───────────────────────────────────────────────────────────────
st.markdown(f"# 🧘 Monitor")
st.markdown(f"Welcome back, **{st.session_state.get('user_name','User')}** · real-time posture monitoring")

col_cam, col_stats = st.columns([3, 2], gap="large")

with col_cam:
    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        if st.button("▶ Start Session", use_container_width=True, type="primary"):
            st.session_state.update(
                session_active=True, session_start=time.time(),
                total_good=0, total_warn=0, total_bad=0,
                alert_count=0, last_alert_time=0, history=deque(maxlen=180))
            st.rerun()
    with bc2:
        if st.button("⏹ Stop Session", use_container_width=True):
            st.session_state.session_active = False
            st.rerun()
    with bc3:
        if st.button("🎯 Calibrate", use_container_width=True,
                     help="Sit in good posture first, then click"):
            with _S["lock"]: _S["calib_req"] = True
            st.toast("✅ Calibrating on next frame…", icon="🎯")

    with _S["lock"]: b = _S["baseline"]
    if b is not None:
        st.caption(f"🎯 Calibrated — baseline: `{b:.4f}`")
    else:
        st.caption("⚠️ Not calibrated — click **Calibrate** while sitting straight for best results.")

    webrtc_streamer(
        key="posture-v5",
        video_processor_factory=PostureProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if red_overlay_on and posture_now == "Bad Posture" and st.session_state.session_active:
        st.markdown("<div class='red-overlay'></div>", unsafe_allow_html=True)

with col_stats:
    st.markdown("<div class='sh'>Live Status</div>", unsafe_allow_html=True)
    pill = {"Good Posture":"pg","Slightly Bent":"pw","Bad Posture":"pb"}.get(posture_now,"pi")
    st.markdown(f"<span class='pill {pill}'>{posture_now}</span>", unsafe_allow_html=True)

    elapsed = "--:--"
    if st.session_state.session_start:
        s = int(time.time() - st.session_state.session_start)
        sfx = "" if st.session_state.session_active else " (ended)"
        elapsed = f"{s//60:02d}:{s%60:02d}{sfx}"

    t1, t2 = st.columns(2)
    with t1:
        st.markdown(f"<div class='mc'><div class='lbl'>Session Time</div>"
                    f"<div class='val wht'>{elapsed}</div></div>", unsafe_allow_html=True)
    with t2:
        st.markdown(f"<div class='mc'><div class='lbl'>Alerts Sent</div>"
                    f"<div class='val bad'>{st.session_state.alert_count}</div></div>",
                    unsafe_allow_html=True)

    total    = st.session_state.total_good + st.session_state.total_warn + st.session_state.total_bad
    good_pct = round(st.session_state.total_good / total * 100) if total else 0
    warn_pct = round(st.session_state.total_warn / total * 100) if total else 0
    bad_pct  = round(st.session_state.total_bad  / total * 100) if total else 0

    st.markdown("<div class='sh'>Posture Breakdown</div>", unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    for col, lbl, pct, cls in [(g1,"Good",good_pct,"good"),
                                (g2,"Slight",warn_pct,"warn"),
                                (g3,"Bad",bad_pct,"bad")]:
        with col:
            st.markdown(f"<div class='mc'><div class='lbl'>{lbl}</div>"
                        f"<div class='val {cls}'>{pct}%</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='sh'>Posture Timeline</div>", unsafe_allow_html=True)
    hl = list(st.session_state.history)
    if hl:
        n=len(hl); svw,svh=380,80; bw=max(2,svw//n-1)
        bars="".join(
            f'<rect x="{i*(svw//n)}" y="{svh-int(v*svh)}" width="{bw}" height="{int(v*svh)}" fill="{c}" rx="1"/>'
            for i,p in enumerate(hl)
            for v,c in [((1.00,"#00e676") if p=="Good Posture" else
                         (0.55,"#ffea00") if p=="Slightly Bent" else (0.18,"#ff1744"))])
        st.markdown(f'<svg viewBox="0 0 {svw} {svh}" xmlns="http://www.w3.org/2000/svg" '
                    f'style="width:100%;border-radius:8px;background:#161922;display:block;">'
                    f'{bars}</svg>', unsafe_allow_html=True)
        st.caption("🟢 Good  🟡 Slightly Bent  🔴 Bad")
    else:
        st.info("Start a session to see your posture timeline.")

    if not st.session_state.session_active and st.session_state.session_start and total > 0:
        st.markdown("<div class='sh'>📋 Session Summary</div>", unsafe_allow_html=True)
        dur = int(time.time() - st.session_state.session_start)
        gs  = round(st.session_state.total_good / total * dur)
        bs  = round(st.session_state.total_bad  / total * dur)
        grade = ("🏆 Excellent" if good_pct>=80 else "👍 Good" if good_pct>=60
                 else "⚠️ Needs Work" if good_pct>=40 else "❌ Poor")
        for ln in [f"**Duration:** {dur//60}m {dur%60}s",
                   f"**Posture Score:** {good_pct}/100 — {grade}",
                   f"**Good time:** ~{gs//60}m {gs%60}s ({good_pct}%)",
                   f"**Bad time:** ~{bs//60}m {bs%60}s ({bad_pct}%)",
                   f"**Alerts:** {st.session_state.alert_count}"]:
            st.markdown(ln)
        if good_pct >= 80:   st.success("🎉 Excellent session!")
        elif good_pct >= 60: st.warning("👍 Decent. More mindfulness needed.")
        else:                st.error("⚠️ Posture needs attention. Check Analytics for tips.")

# ── AUTO-REFRESH ──────────────────────────────────────────────────────────────
if st.session_state.session_active:
    time.sleep(1)
    st.rerun()

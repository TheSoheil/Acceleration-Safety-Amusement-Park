# Acceleration-Safety-Amusement-Park
# IMU Safety Checker – ISO 17842 (X/Y/Z Accel + Combined Check)

A Streamlit app to evaluate safety of IMU accelerations based on ISO thresholds:
- Per-axis thresholds with duration
- Spike detection (instantaneous violations)
- Combined safety check: (ax/ax_adm)^2 + (ay/ay_adm)^2 <= 1 etc.
- Interactive plots and tables
- Optional axis inversion toggle (X/Z)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py

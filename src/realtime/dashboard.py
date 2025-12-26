"""
NEO-COGNITION DASHBOARD - Real-time Cognitive Monitoring Dashboard
Simple standalone version - no integration needed
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time

# Page config
st.set_page_config(
    page_title="NEO-COGNITION Dashboard",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for better look
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #FFD700;
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #CCCCCC;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-title">🧠 NEO-COGNITION DASHBOARD</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-time Cognitive State Monitoring System</p>', unsafe_allow_html=True)

# Sidebar with controls
with st.sidebar:
    st.title("⚙️ Controls")
    
    # Demo mode selection
    st.subheader("Demo Mode")
    demo_mode = st.selectbox(
        "Choose scenario:",
        ["Normal Work", "Deep Focus", "Distracted", "Tired", "Stressed"]
    )
    
    # Display settings
    st.subheader("Display Settings")
    update_speed = st.slider("Update speed", 1, 10, 3)
    show_raw_data = st.checkbox("Show raw data", False)
    
    st.divider()
    
    # Information
    st.info("""
    **How to read metrics:**
    - 🧠 **Cognitive Load**: Mental effort (like CPU usage)
    - 👁️ **Attention**: Focus level on screen
    - 😴 **Fatigue**: Tiredness level
    - 😰 **Stress**: Anxiety level
    - 👁️ **Blink Rate**: Natural blink frequency
    """)

# Initialize session state for storing data
if 'data_history' not in st.session_state:
    st.session_state.data_history = {
        'time': [],
        'cognitive': [],
        'attention': [],
        'fatigue': [],
        'stress': [],
        'blink_rate': [],
        'ear': []
    }
    st.session_state.session_start = datetime.now()

# Function to generate demo data based on mode
def generate_demo_data(mode):
    current_time = datetime.now()
    
    # Base patterns for different modes
    if mode == "Normal Work":
        return {
            'cognitive': 50 + np.sin(time.time() * 0.1) * 10,
            'attention': 70 + np.sin(time.time() * 0.15) * 15,
            'fatigue': 30 + (current_time.minute * 0.3),
            'stress': 25 + np.sin(time.time() * 0.2) * 10,
            'blink_rate': 15 + np.sin(time.time() * 0.25) * 5,
            'ear': 0.25 + np.sin(time.time() * 0.3) * 0.03
        }
    
    elif mode == "Deep Focus":
        return {
            'cognitive': 65 + np.sin(time.time() * 0.05) * 5,
            'attention': 85 + np.sin(time.time() * 0.08) * 5,
            'fatigue': 40 + (current_time.minute * 0.4),
            'stress': 20 + np.sin(time.time() * 0.1) * 5,
            'blink_rate': 8 + np.sin(time.time() * 0.2) * 3,
            'ear': 0.28 + np.sin(time.time() * 0.15) * 0.02
        }
    
    elif mode == "Distracted":
        return {
            'cognitive': 35 + np.random.randn() * 8,
            'attention': 45 + np.random.randn() * 12,
            'fatigue': 20 + np.random.randn() * 5,
            'stress': 30 + np.random.randn() * 8,
            'blink_rate': 18 + np.random.randn() * 6,
            'ear': 0.23 + np.random.randn() * 0.04
        }
    
    elif mode == "Tired":
        return {
            'cognitive': 40 + np.sin(time.time() * 0.2) * 8,
            'attention': 55 + np.sin(time.time() * 0.25) * 10,
            'fatigue': 75 + (current_time.minute * 0.6),
            'stress': 35 + np.sin(time.time() * 0.3) * 8,
            'blink_rate': 10 + np.sin(time.time() * 0.35) * 4,
            'ear': 0.22 + np.sin(time.time() * 0.4) * 0.02
        }
    
    else:  # Stressed
        return {
            'cognitive': 70 + np.random.randn() * 10,
            'attention': 60 + np.random.randn() * 15,
            'fatigue': 50 + (current_time.minute * 0.5),
            'stress': 65 + np.random.randn() * 12,
            'blink_rate': 25 + np.random.randn() * 8,
            'ear': 0.24 + np.random.randn() * 0.05
        }

# Generate current data
current_data = generate_demo_data(demo_mode)

# Clip values to reasonable ranges
for key in ['cognitive', 'attention', 'fatigue', 'stress']:
    current_data[key] = max(0, min(100, current_data[key]))
current_data['blink_rate'] = max(0, min(60, current_data['blink_rate']))
current_data['ear'] = max(0.15, min(0.35, current_data['ear']))

# Update history
st.session_state.data_history['time'].append(datetime.now())
for key in ['cognitive', 'attention', 'fatigue', 'stress', 'blink_rate', 'ear']:
    st.session_state.data_history[key].append(current_data[key])

# Keep only last 100 data points
for key in st.session_state.data_history:
    if len(st.session_state.data_history[key]) > 100:
        st.session_state.data_history[key] = st.session_state.data_history[key][-100:]

# Display 4 main metrics in columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🧠 COGNITIVE LOAD</div>
        <div class="metric-value">{current_data['cognitive']:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
    if current_data['cognitive'] < 40:
        st.caption("🟢 Light")
    elif current_data['cognitive'] < 70:
        st.caption("🟡 Moderate")
    else:
        st.caption("🔴 Heavy")

with col2:
    st.markdown(f"""
    <div class="metric-card" style="background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);">
        <div class="metric-label">👁️ ATTENTION</div>
        <div class="metric-value">{current_data['attention']:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
    if current_data['attention'] < 50:
        st.caption("🔴 Distracted")
    elif current_data['attention'] < 80:
        st.caption("🟡 Normal")
    else:
        st.caption("🟢 Focused")

with col3:
    st.markdown(f"""
    <div class="metric-card" style="background: linear-gradient(135deg, #ff7e5f 0%, #feb47b 100%);">
        <div class="metric-label">😴 FATIGUE</div>
        <div class="metric-value">{current_data['fatigue']:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
    if current_data['fatigue'] < 30:
        st.caption("🟢 Fresh")
    elif current_data['fatigue'] < 60:
        st.caption("🟡 Tired")
    else:
        st.caption("🔴 Exhausted")

with col4:
    st.markdown(f"""
    <div class="metric-card" style="background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);">
        <div class="metric-label">😰 STRESS</div>
        <div class="metric-value">{current_data['stress']:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
    if current_data['stress'] < 30:
        st.caption("🟢 Calm")
    elif current_data['stress'] < 60:
        st.caption("🟡 Stressed")
    else:
        st.caption("🔴 Anxious")

# Blink Rate and EAR Section
st.subheader("👁️ Eye Metrics")
col5, col6 = st.columns(2)

with col5:
    # Blink rate gauge
    fig_blink = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current_data['blink_rate'],
        title={'text': "Blinks per Minute"},
        gauge={
            'axis': {'range': [0, 60]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 10], 'color': "lightgreen"},
                {'range': [10, 20], 'color': "yellow"},
                {'range': [20, 60], 'color': "orange"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 25
            }
        }
    ))
    fig_blink.update_layout(height=250)
    st.plotly_chart(fig_blink, use_container_width=True)
    
    # Blink interpretation
    rate = current_data['blink_rate']
    if rate < 8:
        interpretation = "Deep Focus"
    elif rate < 15:
        interpretation = "Normal"
    elif rate < 25:
        interpretation = "Relaxed"
    else:
        interpretation = "Fatigue/Stress"
    st.caption(f"Interpretation: **{interpretation}**")

with col6:
    # EAR history chart
    fig_ear = go.Figure()
    fig_ear.add_trace(go.Scatter(
        x=st.session_state.data_history['time'],
        y=st.session_state.data_history['ear'],
        mode='lines',
        name='EAR',
        line=dict(color='green', width=2)
    ))
    fig_ear.add_hline(y=0.21, line_dash="dash", line_color="red",
                     annotation_text="Blink Threshold")
    
    fig_ear.update_layout(
        title="Eye Aspect Ratio (EAR) Over Time",
        xaxis_title="Time",
        yaxis_title="EAR Value",
        height=250,
        showlegend=False
    )
    st.plotly_chart(fig_ear, use_container_width=True)
    
    # Current EAR status
    ear_value = current_data['ear']
    if ear_value > 0.25:
        status = "Eyes Open"
    elif ear_value > 0.21:
        status = "Partially Open"
    else:
        status = "Eyes Closed"
    st.caption(f"Current: **{ear_value:.3f}** ({status})")

# Time Series Chart
st.subheader("📈 Cognitive States Over Time")

# Create DataFrame for plotting
df = pd.DataFrame({
    'Time': st.session_state.data_history['time'],
    'Cognitive': st.session_state.data_history['cognitive'],
    'Attention': st.session_state.data_history['attention'],
    'Fatigue': st.session_state.data_history['fatigue'],
    'Stress': st.session_state.data_history['stress']
})

# Create line chart
fig_timeline = go.Figure()
metrics = ['Cognitive', 'Attention', 'Fatigue', 'Stress']
colors = ['#667eea', '#00b09b', '#ff7e5f', '#ff416c']

for metric, color in zip(metrics, colors):
    fig_timeline.add_trace(go.Scatter(
        x=df['Time'],
        y=df[metric],
        name=metric,
        line=dict(color=color, width=2),
        mode='lines'
    ))

fig_timeline.update_layout(
    height=400,
    xaxis_title="Time",
    yaxis_title="Percentage (%)",
    hovermode='x unified',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_timeline, use_container_width=True)

# Session Summary
st.subheader("📋 Session Summary")

col7, col8, col9 = st.columns(3)

with col7:
    duration = datetime.now() - st.session_state.session_start
    minutes = duration.seconds // 60
    seconds = duration.seconds % 60
    st.metric("Duration", f"{minutes}:{seconds:02d}")

with col8:
    avg_attention = np.mean(st.session_state.data_history['attention']) if st.session_state.data_history['attention'] else 0
    st.metric("Avg Attention", f"{avg_attention:.1f}%")

with col9:
    data_points = len(st.session_state.data_history['time'])
    st.metric("Data Points", str(data_points))

# Recommendations
st.subheader("💡 Recommendations")

# Generate recommendations based on current state
recommendations = []

if current_data['attention'] < 50:
    recommendations.append("**Take a short break** - Your attention is low")
    
if current_data['fatigue'] > 60:
    recommendations.append("**Rest your eyes** - High fatigue detected")
    
if current_data['stress'] > 60:
    recommendations.append("**Try deep breathing** - Stress levels are high")
    
if current_data['blink_rate'] < 8:
    recommendations.append("**Blink more** - You're staring intensely")
    
if current_data['blink_rate'] > 25:
    recommendations.append("**Relax your eyes** - Rapid blinking indicates strain")

if not recommendations:
    recommendations.append("**All good!** - Your cognitive state is balanced")

for rec in recommendations:
    st.info(rec)

# Raw data (optional)
if show_raw_data:
    st.subheader("📊 Raw Data")
    st.dataframe(df.tail(10))

# Footer
st.divider()
st.caption(f"Last update: {datetime.now().strftime('%H:%M:%S')} • Dashboard v1.0")

# Auto-refresh
time.sleep(1.0 / update_speed)
st.rerun()
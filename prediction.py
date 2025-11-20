import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# Page config
st.set_page_config(
    page_title="✈️ Airline Satisfaction AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Load models
@st.cache_resource
def load_models():
    try:
        model = pickle.load(open('model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        encoder = pickle.load(open('label_encoder.pkl', 'rb'))
        return model, scaler, encoder
    except:
        return None, None, None


model, scaler, encoder = load_models()

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'show_confetti' not in st.session_state:
    st.session_state.show_confetti = False

# Custom CSS with modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    * {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .heading-predictor {
    text-align: center;
    font-size: 3rem !important;
    font-weight: 700;
    background: linear-gradient(135deg, #ff6a00 0%, #ee0979 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 2px 2px 8px rgba(23, 0, 12, 0.2);
    margin-bottom: 0.5rem;
    animation: fadeInDown 1s ease-out;
}

@keyframes fadeInDown {
    from {
        opacity: 0;
        transform: translateY(-20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

    .main {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 30px;
        padding: 30px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }

    h1 {
        background: linear-gradient(13deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        
        font-weight: 700;
        font-size: 3.5rem !important;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(145,2,9,0.1);
    }

    .subtitle {
        text-align: center;
        color: #700;
        font-size: 2rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }

    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px 40px;
        font-size: 18px;
        font-weight: 600;
        border-radius: 50px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }

    .prediction-card {
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin: 30px 0;
        animation: slideIn 0.5s ease-out;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
    }

    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .satisfied {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }

    .dissatisfied {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        color: white;
    }

    .metric-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }

    .metric-box:hover {
        transform: translateY(-5px);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 5px;
    }

    .service-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(145, 0, 13, 0.05);
        margin: 10px 0;
        border-left: 4px solid #667eea;
    }

    .stSelectbox label, .stNumberInput label, .stSlider label {
        font-weight: 600;
        color: #333;
    }

    .info-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 8px 20px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 5px;
    }

    .stProgress > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    hr {
        margin: 30px 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
    }

    .footer {
        text-align: center;
        padding: 20px;
        color: #600;
        font-size: 0.9rem;
        margin-top: 40px;
    }

    .emoji-large {
        font-size: 4rem;
        margin: 20px 0;
    }

    .confidence-bar {
        height: 30px;
        border-radius: 15px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        margin: 20px 0;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/airplane-take-off.png", width=180)

    page = st.radio("🎯 Navigation",
                    ["🏠 Home", "🔮 Predictor", "📊 Analytics", "📜 History", "ℹ️ About"],
                    label_visibility="collapsed")

    st.markdown("---")

    # Quick stats
    if st.session_state.history:
        st.markdown("### 📈 Session Stats")
        total = len(st.session_state.history)
        satisfied = sum(1 for h in st.session_state.history if h['prediction'] == 1)

        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{total}</div>
            <div class="metric-label">Total Predictions</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{satisfied / total * 100:.1f}%</div>
            <div class="metric-label">Satisfaction Rate</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🤖 Model Info")
    st.info("**XGBoost Classifier**")
    st.success("**Accuracy: 96.29%**")
    st.metric("Precision", "0.97")
    st.metric("Recall", "0.96")

# HOME PAGE
if page == "🏠 Home":
    st.markdown("<h1>✈️ Airline Satisfaction Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>🚀 Powered by XGBoost | 96.29% Accuracy | Real-time Predictions</p>",
                unsafe_allow_html=True)

    # Hero metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-box">
            <div class="emoji-large">🎯</div>
            <div class="metric-value">96.29%</div>
            <div class="metric-label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-box">
            <div class="emoji-large">⚡</div>
            <div class="metric-value">17</div>
            <div class="metric-label">Features</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-box">
            <div class="emoji-large">🤖</div>
            <div class="metric-value">9</div>
            <div class="metric-label">Models Tested</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-box">
            <div class="emoji-large">📊</div>
            <div class="metric-value">104K</div>
            <div class="metric-label">Training Data</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Feature highlights
    st.markdown("### ✨ Key Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="service-card">
            <div class="metric-label"><h4>🎯 High Precision Predictions</h4>
            <p>State-of-the-art XGBoost model with 97% precision rate</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="service-card">
            <div class="metric-label"><h4>📊 Real-time Analytics</h4>
            <p>Interactive visualizations and instant feedback</p>
        </div> </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="service-card">
            <div class="metric-label"><h4>💡 Smart Recommendations</h4>
            <p>AI-powered suggestions to improve satisfaction</p>
        </div> </div>
        """, unsafe_allow_html=True)

    with col2:


        st.markdown("""
        <div class="service-card">
            <div class="metric-label"><h4>📜 History Tracking</h4>
            <p>Keep track of all predictions with timestamps</p>
        </div> </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="service-card">
           <div class="metric-label"> <h4>🎨 Modern UI/UX</h4>
            <p>Beautiful, responsive interface with smooth animations</p>
        </div> </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Model comparison
    st.markdown("### 🏆 Model Performance Comparison")

    model_data = {
        'Model': ['XGBoost', 'Random Forest', 'SVC', 'Decision Tree', 'Gradient Boosting',
                  'KNN', 'AdaBoost', 'Logistic Regression', 'Gaussian NB'],
        'Accuracy': [96.29, 96.14, 95.03, 94.56, 94.18, 92.88, 91.72, 87.67, 86.62]
    }

    fig = go.Figure(data=[
        go.Bar(x=model_data['Accuracy'], y=model_data['Model'], orientation='h',
               marker=dict(color=model_data['Accuracy'],
                           colorscale='Viridis',
                           showscale=True),
               text=[f"{acc:.2f}%" for acc in model_data['Accuracy']],
               textposition='auto')
    ])

    fig.update_layout(
        title="Model Accuracy Comparison",
        xaxis_title="Accuracy (%)",
        height=450,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

# PREDICTOR PAGE
elif page == "🔮 Predictor":
    st.markdown("<h1 > Make Your Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Fill in the passenger details to predict satisfaction</p>", unsafe_allow_html=True)

    if model is None:
        st.error("⚠️ Model files not found! Please ensure all pickle files are in the directory.")
        st.stop()

    with st.form("prediction_form"):
        # Passenger Details Section
        st.markdown("### 👤 Passenger Information")
        col1, col2, col3 = st.columns(3)

        with col1:
            customer_type = st.selectbox("Customer Type 👥",
                                         ["Loyal Customer", "Disloyal Customer"])
        with col2:
            age = st.number_input("Age 🎂", 10, 100, 35)
        with col3:
            type_of_travel = st.selectbox("Travel Type ✈️",
                                          ["Business travel", "Personal travel"])

        col1, col2 = st.columns(2)
        with col1:
            travel_class = st.selectbox("Class 💺", ["Eco", "Eco Plus", "Business"])
        with col2:
            flight_distance = st.number_input("Flight Distance (miles) 🌍",
                                              30, 5000, 1000, step=50)

        st.markdown("<hr>", unsafe_allow_html=True)

        # Service Ratings Section
        st.markdown("### ⭐ Service Experience Ratings")
        st.markdown("<p style='text-align:center; color:#700;'>Rate each service from 1 (Poor) to 5 (Excellent)</p>",
                    unsafe_allow_html=True)

        # Digital Services
        with st.expander("📱 Digital Services", expanded=True):
            col1, col2, col3 = st.columns(3)
            wifi = col1.slider("WiFi Service 📶", 1, 5, 3)
            booking = col2.slider("Online Booking 💻", 1, 5, 4)
            boarding = col3.slider("Online Boarding 📲", 1, 5, 4)

        # In-flight Services
        with st.expander("✈️ In-flight Services", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            food = col1.slider("Food & Drink 🍽️", 1, 5, 3)
            entertainment = col2.slider("Entertainment 🎬", 1, 5, 3)
            service = col3.slider("On-board Service 👨‍✈️", 1, 5, 4)
            inflight_service = col4.slider("Inflight Service 🛫", 1, 5, 4)

        # Comfort Services
        with st.expander("🛋️ Comfort & Amenities", expanded=True):
            col1, col2, col3 = st.columns(3)
            seat_comfort = col1.slider("Seat Comfort 💺", 1, 5, 4)
            leg_room = col2.slider("Leg Room 🦵", 1, 5, 3)
            cleanliness = col3.slider("Cleanliness 🧹", 1, 5, 4)

        # Ground Services
        with st.expander("🏢 Ground Services", expanded=True):
            col1, col2 = st.columns(2)
            baggage = col1.slider("Baggage Handling 🧳", 1, 5, 4)
            checkin = col2.slider("Check-in Service ✅", 1, 5, 4)

        st.markdown("<hr>", unsafe_allow_html=True)

        # Submit button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submit = st.form_submit_button("🚀 Predict Satisfaction", use_container_width=True)

    if submit:
        # Progress bar animation
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(100):
            progress_bar.progress(i + 1)
            if i < 30:
                status_text.text("🔍 Analyzing passenger data...")
            elif i < 60:
                status_text.text("🤖 Running XGBoost model...")
            elif i < 90:
                status_text.text("📊 Calculating confidence score...")
            else:
                status_text.text("✅ Prediction complete!")
            time.sleep(0.01)

        progress_bar.empty()
        status_text.empty()

        # Prepare input data
        inputs = {
            'Customer Type': customer_type,
            'Age': age,
            'Type of Travel': type_of_travel,
            'Class': travel_class,
            'Flight Distance': flight_distance,
            'Inflight wifi service': wifi,
            'Ease of Online booking': booking,
            'Food and drink': food,
            'Online boarding': boarding,
            'Seat comfort': seat_comfort,
            'Inflight entertainment': entertainment,
            'On-board service': service,
            'Leg room service': leg_room,
            'Baggage handling': baggage,
            'Checkin service': checkin,
            'Inflight service': inflight_service,
            'Cleanliness': cleanliness
        }

        df = pd.DataFrame([inputs])

        # Manual encoding
        manual_encoding = {
            "Customer Type": {"Loyal Customer": 0, "Disloyal Customer": 1},
            "Type of Travel": {"Business travel": 0, "Personal travel": 1},
            "Class": {"Eco": 1, "Eco Plus": 2, "Business": 0},
        }

        for col in df.columns:
            if col in manual_encoding:
                df[col] = df[col].map(manual_encoding[col]).fillna(0).astype(int)

        # Scale and predict
        X_scaled = scaler.transform(df)
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]
        confidence = max(proba) * 100

        # Display result
        if pred == 1:
            st.markdown(f"""
            <div class="prediction-card satisfied">
                <div class="emoji-large">😊 ✈️</div>
                <h1 style="color: white;">SATISFIED PASSENGER!</h1>
                <p style="font-size: 1.3rem;">This passenger is likely to have a great experience</p>
                <div class="confidence-bar">
                    Confidence: {confidence:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f"""
            <div class="prediction-card dissatisfied">
                <div class="emoji-large">😞 ✈️</div>
                <h1 style="color: white;">DISSATISFIED PASSENGER</h1>
                <p style="font-size: 1.3rem;">Attention needed to improve passenger experience</p>
                <div class="confidence-bar" style="background: linear-gradient(90deg, #ee0979 0%, #ff6a00 100%);">
                    Confidence: {confidence:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Feature analysis
        st.markdown("### 📊 Service Analysis")

        service_scores = {
            'WiFi': wifi,
            'Booking': booking,
            'Boarding': boarding,
            'Food': food,
            'Entertainment': entertainment,
            'Seat': seat_comfort,
            'Leg Room': leg_room,
            'Cleanliness': cleanliness,
            'Baggage': baggage,
            'Check-in': checkin,
            'Service': service,
            'Inflight': inflight_service
        }

        fig = go.Figure(data=go.Scatterpolar(
            r=list(service_scores.values()),
            theta=list(service_scores.keys()),
            fill='toself',
            marker=dict(color='rgb(102, 126, 234)')
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 5])
            ),
            showlegend=False,
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Recommendations
        st.markdown("### 💡 Recommendations")

        weak_services = [k for k, v in service_scores.items() if v < 3]

        if weak_services:
            for service_name in weak_services:
                st.warning(f"🔧 **{service_name}**: Consider improvement - currently rated below average")
        else:
            st.success("✅ All services are performing well! Keep up the great work!")

        # Save to history
        st.session_state.history.append({
            'timestamp': datetime.now(),
            'prediction': pred,
            'confidence': confidence,
            'inputs': inputs
        })

# ANALYTICS PAGE
elif page == "📊 Analytics":
    st.markdown("<h1>📊 Analytics Dashboard</h1>", unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("📝 No predictions yet. Make some predictions to see analytics!")
    else:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        total = len(st.session_state.history)
        satisfied = sum(1 for h in st.session_state.history if h['prediction'] == 1)
        dissatisfied = total - satisfied
        avg_confidence = np.mean([h['confidence'] for h in st.session_state.history])

        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{total}</div>
                <div class="metric-label">Total Predictions</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{satisfied}</div>
                <div class="metric-label">Satisfied</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{dissatisfied}</div>
                <div class="metric-label">Dissatisfied</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{avg_confidence:.1f}%</div>
                <div class="metric-label">Avg Confidence</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            # Pie chart
            fig = go.Figure(data=[go.Pie(
                labels=['Satisfied', 'Dissatisfied'],
                values=[satisfied, dissatisfied],
                marker=dict(colors=['#38ef7d', '#ff6a00']),
                hole=0.4
            )])
            fig.update_layout(title="Satisfaction Distribution", height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Confidence distribution
            confidences = [h['confidence'] for h in st.session_state.history]
            fig = go.Figure(data=[go.Histogram(
                x=confidences,
                marker=dict(color='rgb(102, 126, 234)'),
                nbinsx=20
            )])
            fig.update_layout(title="Confidence Score Distribution",
                              xaxis_title="Confidence (%)",
                              yaxis_title="Frequency",
                              height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Timeline
        st.markdown("### 📈 Prediction Timeline")
        timeline_data = pd.DataFrame(st.session_state.history)
        timeline_data['result'] = timeline_data['prediction'].map({1: 'Satisfied', 0: 'Dissatisfied'})

        fig = px.scatter(timeline_data, x='timestamp', y='confidence',
                         color='result',
                         color_discrete_map={'Satisfied': '#38ef7d', 'Dissatisfied': '#ff6a00'},
                         size=[10] * len(timeline_data),
                         title="Predictions Over Time")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# HISTORY PAGE
elif page == "📜 History":
    st.markdown("<h1>📜 Prediction History</h1>", unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("📝 No predictions yet. Start making predictions to build your history!")
    else:
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.history = []
                st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)

        for i, record in enumerate(reversed(st.session_state.history)):
            result = "✅ Satisfied" if record['prediction'] == 1 else "❌ Dissatisfied"
            color = "satisfied" if record['prediction'] == 1 else "dissatisfied"

            with st.expander(
                    f"**Prediction #{len(st.session_state.history) - i}** | {record['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} | {result}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"""
                    <div class="service-card">
                        <h4>Prediction Result</h4>
                        <p><strong>Status:</strong> {result}</p>
                        <p><strong>Confidence:</strong> {record['confidence']:.1f}%</p>
                        <p><strong>Time:</strong> {record['timestamp'].strftime('%H:%M:%S')}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown("**Input Details:**")
                    st.json(record['inputs'])

# ABOUT PAGE
else:
    st.markdown("<h1>ℹ️ About This Application</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div class="service-card">
        <div class="metric-label"><h3>🎯 Project Overview</h3>
        <p>This is an advanced machine learning application that predicts airline passenger satisfaction 
        with 96.29% accuracy using the XGBoost algorithm.</p>
    </div></div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="service-card">
            <div class="metric-label"><h4>📊 Dataset</h4>
            <ul>
                <li>104,000+ passenger records</li>
                <li>23 original features</li>
                <li>17 selected features after correlation analysis</li>
                <li>Binary classification (Satisfied/Dissatisfied)</li>
            </ul>
        </div></div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="service-card">
           <div class="metric-label"> <h4>🔧 Technologies Used</h4>
            <ul>
                <li><strong>Python</strong> - Programming language</li>
                <li><strong>Streamlit</strong> - Web framework</li>
                <li><strong>XGBoost</strong> - ML algorithm</li>
                <li><strong>Scikit-learn</strong> - ML library</li>
                <li><strong>Plotly</strong> - Visualization</li>
                <li><strong>Pandas & NumPy</strong> - Data processing</li>
            </ul>
        </div></div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="service-card">
            <div class="metric-label"><h4>🤖 Model Performance</h4>
            <ul>
                <li><strong>Accuracy:</strong> 96.29%</li>
                <li><strong>Precision:</strong> 0.97</li>
                <li><strong>Recall:</strong> 0.96</li>
                <li><strong>F1-Score:</strong> 0.96</li>
                <li><strong>Training Time:</strong> ~2 minutes</li>
            </ul>
        </div></div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="service-card">
           <div class="metric-label"> <h4>✨ Key Features</h4>
            <ul>
                <li>Real-time predictions</li>
                <li>Interactive visualizations</li>
                <li>Service quality analysis</li>
                <li>Historical tracking</li>
                <li>Recommendation engine</li>
                <li>Responsive design</li>
            </ul>
        </div></div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Model comparison table
    st.markdown("### 🏆 Algorithm Comparison")

    comparison_data = pd.DataFrame({
        'Model': ['XGBoost', 'Random Forest', 'SVC', 'Decision Tree',
                  'Gradient Boosting', 'KNN', 'AdaBoost', 'Logistic Regression', 'Gaussian NB'],
        'Accuracy (%)': [96.29, 96.14, 95.03, 94.56, 94.18, 92.88, 91.72, 87.67, 86.62],
        'Precision': [0.97, 0.97, 0.95, 0.94, 0.94, 0.94, 0.92, 0.88, 0.87],
        'Recall': [0.96, 0.96, 0.95, 0.95, 0.94, 0.92, 0.91, 0.87, 0.86],
        'F1-Score': [0.96, 0.96, 0.95, 0.95, 0.94, 0.93, 0.92, 0.87, 0.86]
    })

    st.dataframe(comparison_data.style.background_gradient(cmap='Greens', subset=['Accuracy (%)']),
                 use_container_width=True, hide_index=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Features explanation
    st.markdown("### 📋 Feature Categories")

    tab1, tab2, tab3, tab4 = st.tabs(["👤 Passenger", "📱 Digital", "✈️ In-flight", "🏢 Ground"])

    with tab1:
        st.markdown("""
        <div class="service-card">
            <div class="metric-label"><h4>Passenger Demographics</h4>
            <ul>
                <li><strong>Customer Type:</strong> Loyal vs. Disloyal customer</li>
                <li><strong>Age:</strong> Passenger age (7-85 years)</li>
                <li><strong>Type of Travel:</strong> Business or Personal</li>
                <li><strong>Class:</strong> Economy, Economy Plus, or Business</li>
                <li><strong>Flight Distance:</strong> Journey length in miles</li>
            </ul>
        </div></div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("""
        <div class="service-card">
            <div class="metric-label"><h4>Digital Services</h4>
            <ul>
                <li><strong>Inflight WiFi:</strong> Quality of internet connectivity</li>
                <li><strong>Ease of Online Booking:</strong> Website/app experience</li>
                <li><strong>Online Boarding:</strong> Digital check-in process</li>
            </ul>
            <p><em>Rating Scale: 1 (Poor) to 5 (Excellent)</em></p>
        </div></div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown("""
        <div class="service-card">
            <div class="metric-label"><h4>In-flight Experience</h4>
            <ul>
                <li><strong>Seat Comfort:</strong> Comfort level of seating</li>
                <li><strong>Leg Room Service:</strong> Space availability</li>
                <li><strong>Food and Drink:</strong> Meal quality and variety</li>
                <li><strong>Inflight Entertainment:</strong> Entertainment options</li>
                <li><strong>On-board Service:</strong> Crew service quality</li>
                <li><strong>Cleanliness:</strong> Aircraft hygiene standards</li>
            </ul>
            <p><em>Rating Scale: 1 (Poor) to 5 (Excellent)</em></p>
        </div></div>
        """, unsafe_allow_html=True)

    with tab4:
        st.markdown("""
        <div class="service-card">
           <div class="metric-label"> <h4>Ground Services</h4>
            <ul>
                <li><strong>Check-in Service:</strong> Counter/kiosk experience</li>
                <li><strong>Baggage Handling:</strong> Luggage management</li>
            </ul>
            <p><em>Rating Scale: 1 (Poor) to 5 (Excellent)</em></p>
        </div></div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Data preprocessing info
    st.markdown("### 🔧 Data Preprocessing Pipeline")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="service-card">
           <div class="metric-label"> <h4>Step 1: Data Cleaning</h4>
            <ul>
                <li>Handled missing values using KNN imputation</li>
                <li>Removed duplicate records</li>
                <li>Validated data types and ranges</li>
            </ul>
        </div></div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="service-card">
            <div class="metric-label"><h4>Step 3: Feature Selection</h4>
            <ul>
                <li>Correlation analysis performed</li>
                <li>Dropped low-correlation features</li>
                <li>Removed: Gender, Gate location, Delay features</li>
            </ul>
        </div></div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="service-card">
            <div class="metric-label"><h4>Step 2: Encoding</h4>
            <ul>
                <li>Label Encoding for categorical variables</li>
                <li>Customer Type: Loyal=0, Disloyal=1</li>
                <li>Travel Type: Business=0, Personal=1</li>
                <li>Class: Business=0, Eco=1, Eco Plus=2</li>
            </ul>
        </div></div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="service-card">
            <div class="metric-label"><h4>Step 4: Scaling</h4>
            <ul>
                <li>MinMaxScaler applied to all features</li>
                <li>Normalized values to [0, 1] range</li>
                <li>Improved model convergence</li>
            </ul>
        </div></div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # How to use section
    st.markdown("### 📖 How to Use This App")

    st.markdown("""
    <div class="service-card">
        <div class="metric-label"><h4>Step-by-Step Guide:</h4>
        <ol>
            <li><strong>Navigate to Predictor:</strong> Click on 🔮 Predictor in the sidebar</li>
            <li><strong>Enter Passenger Details:</strong> Fill in demographic and flight information</li>
            <li><strong>Rate Services:</strong> Provide ratings (1-5) for each service category</li>
            <li><strong>Submit Prediction:</strong> Click the "Predict Satisfaction" button</li>
            <li><strong>Review Results:</strong> Analyze the prediction, confidence score, and recommendations</li>
            <li><strong>View Analytics:</strong> Check 📊 Analytics for overall trends</li>
            <li><strong>Check History:</strong> Access 📜 History to review past predictions</li>
        </ol>
    </div></div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Tips and tricks
    st.markdown("### 💡 Tips for Best Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="service-card">
            <div class="metric-label"><h4>🎯 Accurate Ratings</h4>
            <p>Provide honest and realistic service ratings. The model works best with genuine feedback.</p>
        </div></div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="service-card">
            <div class="metric-label"><h4>📊 Use Analytics</h4>
            <p>Review analytics regularly to identify patterns and trends in passenger satisfaction.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="service-card">
            <div class="metric-label"><h4>💬 Act on Insights</h4>
            <p>Use recommendations to improve low-rated services and enhance passenger experience.</p>
        </div></div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Contact and credits
    st.markdown("### 👥 About the Developer")

    st.markdown("""
    <div class="service-card">
        <div class="metric-label"><p>This application was developed as a machine learning project to demonstrate real-world 
        application of predictive analytics in the aviation industry.</p>
        <br>
        <p><strong>Project Goals:</strong></p>
        <ul>
            <li>Predict passenger satisfaction with high accuracy</li>
            <li>Provide actionable insights for airlines</li>
            <li>Create an intuitive, user-friendly interface</li>
            <li>Demonstrate best practices in ML deployment</li>
        </ul>
    </div></div>
    """, unsafe_allow_html=True)

    # Disclaimer
    st.warning("""
    **⚠️ Disclaimer**

    This is an educational project. Predictions should be used as supplementary insights 
    alongside other business metrics and human judgment. The model's accuracy is based on 
    historical data and may not reflect all real-world scenarios.
    """)

# Footer (appears on all pages)
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    <p>✈️ <strong>Airline Satisfaction Predictor</strong> | Powered by XGBoost & Streamlit</p>
    <p>🎯 96.29% Accuracy | 📊 17 Features | 🤖 9 Models Tested</p>
    <p>Made with ❤️ using Python | © 2025</p>
</div>
""", unsafe_allow_html=True)
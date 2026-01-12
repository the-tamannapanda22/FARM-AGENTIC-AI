# Create the main Streamlit application file
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
from streamlit_option_menu import option_menu
from dotenv import load_dotenv
import warnings

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Custom imports
from weather_service import WeatherService
from soil_monitor import soilMonitor

# Page configuration
st.set_page_config(page_title="SmartCrop AI Agent", page_icon="🌱", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 3rem; color: #2E8B57; text-align: center; margin-bottom: 2rem; font-weight: bold; text-shadow: 2px 2px 4px rgba(0,0,0,0.1); }
    .sub-header { font-size: 1.8rem; color: #228B22; margin-bottom: 1.5rem; border-bottom: 2px solid #32CD32; padding-bottom: 0.5rem; }
    .metric-card { background: linear-gradient(135deg, #f0f8f0 0%, #e8f5e8 100%); padding: 1.5rem; border-radius: 15px; border-left: 5px solid #32CD32; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 1rem; }
    .prediction-card { background: linear-gradient(135deg, #e6f3ff 0%, #cce7ff 100%); padding: 2rem; border-radius: 15px; border: 2px solid #4169E1; box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15); }
    .stButton > button { background: linear-gradient(135deg, #32CD32 0%, #228B22 100%); color: white; border-radius: 25px; border: none; padding: 0.75rem 2rem; font-size: 1rem; font-weight: bold; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); }
    .chat-message { padding: 1rem; border-radius: 10px; margin: 0.5rem 0; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); }
    .user-message { background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); margin-left: auto; width: 80%; }
    .ai-message { background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%); margin-right: auto; width: 80%; }
</style>
""", unsafe_allow_html=True)


class SmartCropApp:
    def __init__(self):
        """Initialize the SmartCrop AI Agent application"""
        try:
            self.soil_monitor = soilMonitor()
            self.weather_service = WeatherService()

            if 'chat_history' not in st.session_state: st.session_state.chat_history = []
            if 'farm_data' not in st.session_state: st.session_state.farm_data = self._load_farm_data()

        except Exception as e:
            st.error(f"Error initializing application: {e}")
            st.stop()

    def _load_farm_data(self):
        return {
            'name': os.getenv('FARM_NAME', 'Green Valley Farm'),
            'location': os.getenv('FARM_LOCATION', 'Maharashtra, India'),
            'size': float(os.getenv('FARM_SIZE', '50.0')),
            'primary_crops': ['Wheat', 'Cotton', 'Soybeans'],
            'last_updated': datetime.now()
        }

    def main(self):
        """Main application entry point"""
        st.markdown('<h1 class="main-header">🌱 SmartCrop AI Agent</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.3rem; color: #666;">AI-Powered Farming Optimization with Gemini</p>', unsafe_allow_html=True)

        selected = option_menu(
            menu_title=None,
            options=["Dashboard", "Crop Prediction", "Weather", "Soil Monitor", "AI Assistant", "Settings"],
            icons=["speedometer2", "graph-up-arrow", "cloud-sun", "geo-alt-fill", "robot", "gear-fill"],
            orientation="horizontal",
            styles={"nav-link-selected": {"background-color": "#2E8B57"}}
        )

        try:
            if selected == "Dashboard": self.show_dashboard()
            elif selected == "Crop Prediction": self.show_crop_prediction()
            elif selected == "Weather": self.show_weather_intelligence()
            elif selected == "Soil Monitor": self.show_soil_monitor()
            elif selected == "AI Assistant": self.show_ai_assistant()
            elif selected == "Settings": self.show_settings()
        except Exception as e:
            st.error(f"An error occurred in the '{selected}' tab: {e}")
            st.exception(e)

    def show_dashboard(self):
        st.markdown('<h2 class="sub-header">📊 Farm Overview Dashboard</h2>', unsafe_allow_html=True)
        soil_data = self.soil_monitor.get_latest_readings()
        weather_data = self.weather_service.get_current_weather()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🌾 Predicted Yield", f"{2340:.0f} kg/ha", "12% vs last season")
        col2.metric("🌡️ Current Temp", f"{weather_data.get('temperature', 0):.1f}°C", f"{weather_data.get('temperature', 0) - 26:+.1f}°C from optimal")
        col3.metric("💧 Soil Moisture", f"{soil_data.get('moisture', 0):.1f}%", "Optimal")
        col4.metric("🌱 Crop Health", "Excellent", f"AI Confidence: {soil_data.get('health_score', 90):.0f}%")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📈 Yield Prediction Trends")
            hist_df = pd.DataFrame({'Date': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01']), 'Predicted': [2100, 2150, 2250, 2340], 'Actual': [2050, 2120, 2280, None]})
            fig = px.line(hist_df, x='Date', y=['Predicted', 'Actual'], title="Monthly Predictions vs Actual", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("🌱 Soil Health Analysis")
            categories = ['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Moisture']
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=[soil_data.get(k,0) for k in ['nitrogen','phosphorus','potassium']] + [soil_data.get('ph',0)*10, soil_data.get('moisture',0)], theta=categories, fill='toself', name='Current'))
            fig.add_trace(go.Scatterpolar(r=[100, 65, 125, 68, 70], theta=categories, fill='toself', name='Optimal'))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<h3 class="sub-header">🎯 Smart Recommendations</h3>', unsafe_allow_html=True)
        recommendations = self.soil_monitor.get_dashboard_recommendations(soil_data, weather_data)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""<div class="metric-card" style="color: #333;"><h4>🚿 Irrigation Schedule</h4><p><strong>Next:</strong> {recommendations['irrigation']['next_time']}</p><p><strong>Duration:</strong> {recommendations['irrigation']['duration']}</p></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card" style="color: #333;"><h4>🧪 Fertilization Plan</h4><p><strong>Type:</strong> {recommendations['fertilization']['type']}</p><p><strong>Priority:</strong> {recommendations['fertilization']['priority']}</p></div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-card" style="color: #333;"><h4>🦗 Pest & Disease</h4><p><strong>Risk:</strong> {recommendations['pest_control']['risk_level']}</p><p><strong>Action:</strong> {recommendations['pest_control']['action']}</p></div>""", unsafe_allow_html=True)

    def show_crop_prediction(self):
        st.markdown('<h2 class="sub-header">🌾 AI Crop Yield Prediction</h2>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### 📝 Input Parameters")
            with st.expander("🌱 Soil & Weather", expanded=True):
                crop_type = st.selectbox("Select Crop", list(self.soil_monitor.crop_database.keys()))
                nitrogen, phosphorus, potassium = st.slider("N",0,200,85), st.slider("P",0,150,60), st.slider("K",0,250,120)
                ph_level, temperature = st.slider("pH", 5.0, 9.0, 6.8, 0.1), st.slider("Temp (°C)", 10, 45, 28)
                organic_matter, humidity = st.slider("Organic Matter (%)",0.5,8.0,3.2,0.1), st.slider("Humidity (%)",30,90,65)
                rainfall, sunshine = st.slider("Rainfall (mm)",0,500,120), st.slider("Sunshine (Hrs/Day)",4,14,8)
                area_hectares = st.number_input("Area (ha)", 1, 1000, 10)
                irrigation_type, fertilizer_used = st.selectbox("Irrigation", ["Drip", "Sprinkler", "Flood", "Rain-fed"]), st.selectbox("Fertilizer", ["Organic", "Chemical", "Mixed", "None"])

            if st.button("🔮 Predict Crop Yield", use_container_width=True):
                with st.spinner("🤖 AI is analyzing..."):
                    input_data = {'nitrogen': nitrogen, 'phosphorus': phosphorus, 'potassium': potassium, 'ph': ph_level,'organic_matter': organic_matter, 'temperature': temperature, 'humidity': humidity,'rainfall': rainfall, 'sunshine': sunshine, 'crop_type': crop_type, 'area': area_hectares,'irrigation': irrigation_type, 'fertilizer': fertilizer_used}
                    st.session_state.prediction = self.soil_monitor.predict_yield(input_data)
        with col2:
            if 'prediction' in st.session_state:
                res = st.session_state.prediction
                st.markdown(f"""<div class="prediction-card"><h3>🎯 AI Prediction Results</h3><div style="text-align: center; margin: 1rem 0;"><h1 style="color: #4169E1; font-size: 3rem; margin: 0;">{res["yield"]:.0f}</h1><h3 style="color: #666; margin: 0;">kg/ha</h3></div><hr><p><strong>Total Yield:</strong> {res["total_yield"]:.0f} kg | <strong>Confidence:</strong> {res["confidence"]:.1f}%</p><p><strong>Category:</strong> {res["category"]} | <strong>Model:</strong> {res["model"]}</p></div>""", unsafe_allow_html=True)
                st.subheader("💡 Smart Farming Recommendations")
                for rec in res["recommendations"]:
                    st.error(f"🔴 **High**: {rec['recommendation']}") if rec["priority"] == "High" else st.warning(f"🟡 **Medium**: {rec['recommendation']}") if rec["priority"] == "Medium" else st.info(f"🟢 **Low**: {rec['recommendation']}")

    def show_weather_intelligence(self):
        st.markdown('<h2 class="sub-header">🌤️ Weather Intelligence</h2>', unsafe_allow_html=True)
        forecast_df = pd.DataFrame(self.weather_service.get_7day_forecast())
        st.subheader("📅 7-Day Weather Forecast")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('Temperature Trends (°C)', 'Rainfall Forecast (mm)'))
        fig.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['max_temp'], name='Max'), row=1, col=1)
        fig.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['min_temp'], name='Min'), row=1, col=1)
        fig.add_trace(go.Bar(x=forecast_df['date'], y=forecast_df['rainfall'], name='Rainfall'), row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

    def show_soil_monitor(self):
        st.markdown('<h2 class="sub-header">🌱 Soil Health Monitoring</h2>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("📊 Current Parameters")
            soil_data = self.soil_monitor.get_latest_readings()
            st.metric("Health Score", f"{soil_data['health_score']:.0f}/100")
            st.metric("Moisture", f"{soil_data['moisture']:.1f}%")
            st.metric("pH Level", f"{soil_data['ph']:.1f}")
            st.metric("Nitrogen", f"{soil_data['nitrogen']:.0f} ppm")
        with col2:
            st.subheader("📈 30-Day Soil Trends")
            hist_df = pd.DataFrame(self.soil_monitor.get_historical_data())
            hist_df['date'] = pd.to_datetime(pd.date_range(end=datetime.now(), periods=30))
            fig = px.line(hist_df, x='date', y=['moisture', 'health_score'], title="Historical Moisture and Health Score")
            st.plotly_chart(fig, use_container_width=True)

    def show_ai_assistant(self):
        st.markdown('<h2 class="sub-header">🤖 AI Farming Assistant (Gemini)</h2>', unsafe_allow_html=True)
        for message in st.session_state.chat_history:
            with st.chat_message(message['role']):
                st.markdown(message['content'])

        if prompt := st.chat_input("Ask your farming question..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("🤖 AI is thinking..."):
                    context = {
                        'farm_data': st.session_state.farm_data,
                        'soil_data': self.soil_monitor.get_latest_readings(),
                        'weather_data': self.weather_service.get_current_weather()
                    }
                    response = self.soil_monitor.get_ai_response(prompt, context)
                    message_placeholder.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

    def show_settings(self):
        st.markdown('<h2 class="sub-header">⚙️ Settings & Configuration</h2>', unsafe_allow_html=True)
        st.info("Settings are configured via the `.env` file for this demo application.")
        
        tab1, tab2 = st.tabs(["API Status", "Farm Profile"])

        with tab1:
            st.subheader("API Connection Status")
            gemini_status = "✅ Connected" if self.soil_monitor.test_connection() else "❌ Disconnected"
            weather_status = "✅ Connected" if self.weather_service.test_connection() else "❌ Disconnected (Using Mock Data)"
            st.markdown(f"**Gemini AI Status:** {gemini_status}")
            st.markdown(f"**Weather API Status:** {weather_status}")
            
            # RECTIFIED: Updated the model name to reflect the new model being used.
            st.text_input("Model Version", value="gemini-1.5-flash", disabled=True)

        with tab2:
            st.subheader("Current Farm Profile")
            st.text_input("Farm Name", value=st.session_state.farm_data['name'], disabled=True)
            st.text_input("Location", value=st.session_state.farm_data['location'], disabled=True)
            st.multiselect("Primary Crops", options=st.session_state.farm_data['primary_crops'], default=st.session_state.farm_data['primary_crops'], disabled=True)
        
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

# Application entry point
if __name__ == "__main__":
    try:
        app = SmartCropApp()
        app.main()
    except Exception as e:
        st.error(f"A critical error occurred: {e}")
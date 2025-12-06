"""
Streamlit Frontend for Cargo Capacity Forecaster
"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Cargo Capacity Forecaster",
    page_icon="✈️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.json().get("model_loaded", False)
    except:
        return False

def main():
    """Main application."""
    st.markdown('<p class="main-header">✈️ Dynamic Cargo Capacity Forecaster</p>', unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ API is not running or model is not loaded. Please start the backend first:")
        st.code("uvicorn backend.main:app --reload")
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page",
        ["Forecast", "What-If Simulator", "Marketplace"]
    )
    
    if page == "Forecast":
        show_forecast_page()
    elif page == "What-If Simulator":
        show_simulator_page()
    elif page == "Marketplace":
        show_marketplace_page()

def show_forecast_page():
    """Forecast page."""
    st.header("📊 Cargo Capacity Forecast")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Flight Parameters")
        passenger_count = st.number_input("Passenger Count", min_value=1, max_value=500, value=150)
        month = st.number_input("Month", min_value=1, max_value=12, value=6)
        day_of_week = st.selectbox("Day of Week", 
                                   options=list(range(7)), 
                                   format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x],
                                   index=2)
        is_weekend = 1 if day_of_week >= 5 else 0
        
        holiday_flag = st.selectbox("Holiday", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        group_travel_ratio = st.slider("Group Travel Ratio", 0.0, 1.0, 0.2)
        delay_probability = st.slider("Delay Probability", 0.0, 1.0, 0.1)
        weather_index = st.slider("Weather Index", 0.0, 1.0, 0.5)
    
    with col2:
        st.subheader("Aircraft Details")
        fuel_weight = st.number_input("Fuel Weight (kg)", min_value=1000, max_value=50000, value=5000)
        fuel_price = st.number_input("Fuel Price per kg", min_value=0.5, max_value=5.0, value=0.85, step=0.01)
        cargo_price = st.number_input("Cargo Price per kg", min_value=0.5, max_value=10.0, value=1.5, step=0.01)
    
    if st.button("🚀 Get Forecast", type="primary"):
        flight_data = {
            "passenger_count": passenger_count,
            "year": 2024,
            "month": month,
            "day_of_week": day_of_week,
            "day_of_month": 15,
            "is_weekend": is_weekend,
            "group_travel_ratio": group_travel_ratio,
            "holiday_flag": holiday_flag,
            "delay_probability": delay_probability,
            "weather_index": weather_index,
            "origin_encoded": 0,
            "destination_encoded": 0,
            "tail_number_encoded": 0,
            "aircraft_type_encoded": 0,
            "fuel_weight_kg": fuel_weight,
            "fuel_price_per_kg": fuel_price,
            "cargo_price_per_kg": cargo_price
        }
        
        try:
            response = requests.post(f"{API_BASE_URL}/predict", json=flight_data)
            if response.status_code == 200:
                result = response.json()
                
                # Display results
                st.success("✅ Forecast Generated Successfully!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Predicted Baggage",
                        f"{result['predicted_baggage']:.0f} kg",
                        delta=f"±{(result['predicted_baggage_upper'] - result['predicted_baggage']) / result['predicted_baggage'] * 100:.1f}%"
                    )
                
                with col2:
                    st.metric(
                        "Remaining Cargo",
                        f"{result['remaining_cargo']:.0f} kg",
                        delta=f"±{(result['remaining_cargo_upper'] - result['remaining_cargo']) / (result['remaining_cargo'] + 1) * 100:.1f}%"
                    )
                
                with col3:
                    confidence_color = "🟢" if result['confidence'] > 0.7 else "🟡" if result['confidence'] > 0.4 else "🔴"
                    st.metric(
                        "Confidence",
                        f"{confidence_color} {result['confidence']:.1%}"
                    )
                
                # Confidence intervals visualization
                st.subheader("📈 Confidence Intervals")
                
                fig = go.Figure()
                
                # Baggage
                fig.add_trace(go.Bar(
                    name='Predicted Baggage',
                    x=['Baggage Weight'],
                    y=[result['predicted_baggage']],
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=[result['predicted_baggage_upper'] - result['predicted_baggage']],
                        arrayminus=[result['predicted_baggage'] - result['predicted_baggage_lower']]
                    ),
                    marker_color='#1f77b4'
                ))
                
                # Remaining cargo
                fig.add_trace(go.Bar(
                    name='Remaining Cargo',
                    x=['Remaining Cargo'],
                    y=[result['remaining_cargo']],
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=[result['remaining_cargo_upper'] - result['remaining_cargo']],
                        arrayminus=[result['remaining_cargo'] - result['remaining_cargo_lower']]
                    ),
                    marker_color='#2ca02c'
                ))
                
                fig.update_layout(
                    title="Predictions with 95% Confidence Intervals",
                    yaxis_title="Weight (kg)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance
                st.subheader("🔍 Top Contributing Factors")
                try:
                    importance_response = requests.get(f"{API_BASE_URL}/feature-importance")
                    if importance_response.status_code == 200:
                        importance_data = importance_response.json()
                        top_features = importance_data['top_features'][:5]
                        
                        features_df = pd.DataFrame({
                            'Feature': top_features,
                            'Importance': ['High' if i < 2 else 'Medium' if i < 4 else 'Low' for i in range(len(top_features))]
                        })
                        st.table(features_df)
                except:
                    st.info("Feature importance data unavailable")
            
            else:
                st.error(f"Error: {response.text}")
        
        except Exception as e:
            st.error(f"Failed to connect to API: {str(e)}")

def show_simulator_page():
    """What-If Simulator page."""
    st.header("🔮 What-If Simulator")
    st.markdown("Adjust scenario parameters to see how predictions change in real-time.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Scenario Parameters")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            expected_passengers = st.slider(
                "Expected Passengers",
                min_value=50,
                max_value=400,
                value=150,
                key="sim_passengers"
            )
            
            group_travel_ratio = st.slider(
                "Group Travel Ratio",
                0.0,
                1.0,
                0.2,
                step=0.05,
                key="sim_group"
            )
        
        with col_b:
            holiday_flag = st.selectbox(
                "Holiday Flag",
                options=[0, 1],
                format_func=lambda x: "Regular Day" if x == 0 else "Holiday",
                key="sim_holiday"
            )
            
            delay_probability = st.slider(
                "Delay Probability",
                0.0,
                1.0,
                0.1,
                step=0.05,
                key="sim_delay"
            )
        
        if st.button("🔄 Run Simulation", type="primary"):
            changes = {
                "expected_passengers": expected_passengers,
                "group_travel_ratio": group_travel_ratio,
                "holiday_flag": holiday_flag,
                "delay_probability": delay_probability
            }
            
            try:
                response = requests.post(
                    f"{API_BASE_URL}/simulate",
                    json={"changes": changes}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success("✅ Simulation Complete!")
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Predicted Baggage",
                            f"{result['predicted_baggage']:.0f} kg"
                        )
                    
                    with col2:
                        st.metric(
                            "Remaining Cargo",
                            f"{result['remaining_cargo']:.0f} kg"
                        )
                    
                    with col3:
                        confidence_color = "🟢" if result['confidence'] > 0.7 else "🟡" if result['confidence'] > 0.4 else "🔴"
                        st.metric(
                            "Confidence",
                            f"{confidence_color} {result['confidence']:.1%}"
                        )
                    
                    # Explanation
                    st.subheader("📋 Explanation")
                    
                    explanation = result.get('explanation', {})
                    top_factors = explanation.get('top_factors', [])
                    
                    st.markdown("**Top Contributing Factors:**")
                    for i, factor in enumerate(top_factors, 1):
                        st.markdown(f"{i}. **{factor}**")
                    
                    # Confidence intervals
                    if 'prediction_details' in result:
                        details = result['prediction_details']
                        st.markdown("**Confidence Intervals:**")
                        st.json(details)
                
                else:
                    st.error(f"Error: {response.text}")
            
            except Exception as e:
                st.error(f"Failed to connect to API: {str(e)}")
    
    with col2:
        st.subheader("💡 Tips")
        st.info("""
        **What-If Simulator Guide:**
        
        - **Passengers**: More passengers = more baggage
        - **Group Travel**: Higher ratio = more coordinated baggage
        - **Holidays**: Often increase passenger loads
        - **Delays**: Can affect cargo capacity planning
        
        Adjust parameters and click "Run Simulation" to see updated predictions!
        """)

def show_marketplace_page():
    """Marketplace page."""
    st.header("🏪 Cargo Marketplace")
    st.markdown("Generate and reserve cargo capacity slots.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Generate Slots")
        
        predicted_cargo = st.number_input(
            "Predicted Remaining Cargo (kg)",
            min_value=0.0,
            max_value=10000.0,
            value=500.0,
            step=10.0
        )
        
        confidence = st.slider(
            "Confidence Level",
            0.0,
            1.0,
            0.8,
            step=0.05
        )
        
        slot_size = st.number_input(
            "Slot Size (kg)",
            min_value=1.0,
            max_value=100.0,
            value=20.0,
            step=1.0
        )
        
        if st.button("🛒 Generate Slots", type="primary"):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/marketplace/generate-slots",
                    json={
                        "predicted_cargo": predicted_cargo,
                        "confidence": confidence,
                        "slot_size_kg": slot_size
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    slots = result['slots']
                    
                    st.success(f"✅ Generated {len(slots)} slot(s)")
                    
                    # Store slots in session state
                    st.session_state['marketplace_slots'] = slots
                    
                    # Display slots
                    st.subheader("Available Slots")
                    
                    for i, slot in enumerate(slots):
                        with st.container():
                            col_a, col_b, col_c, col_d = st.columns([2, 1, 1, 1])
                            
                            with col_a:
                                st.markdown(f"**Slot ID:** {slot['slot_id']}")
                                st.markdown(f"Weight: {slot['weight']} kg")
                            
                            with col_b:
                                risk_color = "🟢" if slot['risk_score'] < 0.3 else "🟡" if slot['risk_score'] < 0.6 else "🔴"
                                st.markdown(f"Risk: {risk_color} {slot['risk_score']:.2f}")
                            
                            with col_c:
                                st.markdown(f"**${slot['price']:.2f}**")
                                st.caption(f"${slot['price_per_kg']:.2f}/kg")
                            
                            with col_d:
                                if st.button("Reserve", key=f"reserve_{i}"):
                                    try:
                                        reserve_response = requests.post(
                                            f"{API_BASE_URL}/marketplace/reserve/{slot['slot_id']}",
                                            json={}
                                        )
                                        if reserve_response.status_code == 200:
                                            st.success(f"✅ Reserved {slot['slot_id']}")
                                            slot['status'] = 'reserved'
                                        else:
                                            st.error("Reservation failed")
                                    except Exception as e:
                                        st.error(f"Error: {str(e)}")
                            
                            st.divider()
                
                else:
                    st.error(f"Error: {response.text}")
            
            except Exception as e:
                st.error(f"Failed to connect to API: {str(e)}")
    
    with col2:
        st.subheader("📊 Statistics")
        if 'marketplace_slots' in st.session_state:
            slots = st.session_state['marketplace_slots']
            total_weight = sum(s['weight'] for s in slots)
            total_value = sum(s['price'] for s in slots)
            avg_price_per_kg = total_value / total_weight if total_weight > 0 else 0
            
            st.metric("Total Slots", len(slots))
            st.metric("Total Weight", f"{total_weight:.0f} kg")
            st.metric("Total Value", f"${total_value:.2f}")
            st.metric("Avg Price/kg", f"${avg_price_per_kg:.2f}")

if __name__ == "__main__":
    main()


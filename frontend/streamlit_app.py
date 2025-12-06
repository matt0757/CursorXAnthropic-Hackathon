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
        ["Forecast", "Cargo Optimizer", "Marketplace"]
    )
    
    if page == "Forecast":
        show_forecast_page()
    elif page == "Cargo Optimizer":
        show_optimizer_page()
    elif page == "Marketplace":
        show_marketplace_page()
    elif page == "Cargo Optimizer":
        show_optimizer_page()

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
        
        # Route information
        origin = st.text_input("Origin (optional)", placeholder="e.g., KUL", help="Airport code")
        destination = st.text_input("Destination (optional)", placeholder="e.g., LHR", help="Airport code")

        group_travel_ratio = st.slider("Group Travel Ratio", 0.0, 1.0, 0.2)
        delay_probability = st.slider("Delay Probability", 0.0, 1.0, 0.1)
        weather_index = st.slider("Weather Index", 0.0, 1.0, 0.5)
    
    with col2:
        st.subheader("Aircraft Details")
        
        # Aircraft type and tail number
        aircraft_type = st.text_input("Aircraft Type (optional)", placeholder="e.g., A330-300", help="Leave blank to use defaults")
        tail_number = st.text_input("Tail Number (optional)", placeholder="e.g., 9M-XXX", help="Leave blank to use defaults")
        
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
            "weather_index": weather_index
        }
        
        # Add optional fields if provided
        if aircraft_type:
            flight_data["aircraft_type"] = aircraft_type
        if tail_number:
            flight_data["tail_number"] = tail_number
        if origin:
            flight_data["origin"] = origin
        if destination:
            flight_data["destination"] = destination
        
        # Add default encoded values if not using actual values
        if not aircraft_type:
            flight_data["aircraft_type_encoded"] = 0
        if not tail_number:
            flight_data["tail_number_encoded"] = 0
        if not origin:
            flight_data["origin_encoded"] = 0
        if not destination:
            flight_data["destination_encoded"] = 0
        
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
                        "Predicted Cargo Demand",
                        f"{result.get('predicted_cargo_demand', 0):.0f} kg",
                        delta=f"±{(result.get('predicted_cargo_demand_upper', 0) - result.get('predicted_cargo_demand', 0)) / (result.get('predicted_cargo_demand', 1) + 1) * 100:.1f}%"
                    )
                
                with col3:
                    st.metric(
                        "Remaining Cargo Capacity",
                        f"{result['remaining_cargo']:.0f} kg",
                        delta=f"±{(result['remaining_cargo_upper'] - result['remaining_cargo']) / (result['remaining_cargo'] + 1) * 100:.1f}%"
                    )
                
                col4, col5 = st.columns(2)
                
                with col4:
                    st.metric(
                        "Predicted Cargo Volume",
                        f"{result.get('predicted_cargo_volume', 0):.1f} m³"
                    )
                
                with col5:
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

def show_optimizer_page():
    """Cargo Optimizer page - Multi-flight allocation with priority-based scheduling."""
    st.header("🔧 Multi-Flight Cargo Optimizer")
    st.markdown("Allocate cargo across multiple flights. Higher priority cargo gets earlier flights.")
    
    # Initialize session state
    if 'cargo_requests' not in st.session_state:
        st.session_state['cargo_requests'] = []
    if 'optimization_result' not in st.session_state:
        st.session_state['optimization_result'] = None
    if 'flights_loaded' not in st.session_state:
        st.session_state['flights_loaded'] = False
    
    # Check if flights are loaded
    try:
        status_response = requests.get(f"{API_BASE_URL}/flights/status")
        if status_response.status_code == 200:
            status = status_response.json()
            st.session_state['flights_loaded'] = status['flights_loaded']
            st.session_state['flight_count'] = status['flight_count']
    except:
        pass
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Flight Data Loading Section
        st.subheader("📂 Flight Data")
        
        if not st.session_state.get('flights_loaded', False):
            st.warning("⚠️ No flights loaded.")
            
            if st.button("📥 Load Future Flights (2025)", type="primary"):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/flights/load",
                        json={"filepath": "data/future_flights_2025.csv"}
                    )
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ {result['message']}")
                        st.session_state['flights_loaded'] = True
                        st.session_state['flight_count'] = result['flight_count']
                        st.rerun()
                    else:
                        st.error(f"Error: {response.json().get('detail', response.text)}")
                except Exception as e:
                    st.error(f"Failed to load flights: {str(e)}")
        else:
            st.success(f"✅ {st.session_state.get('flight_count', 0)} flights loaded")
        
        st.divider()
        
        # Flight Filter Section
        st.subheader("✈️ Flight Selection")
        
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            origin_filter = st.text_input("Origin Airport", placeholder="e.g., KUL", key="opt_origin")
            from_date = st.date_input("From Date", value=None, key="opt_from_date")
        with filter_col2:
            dest_filter = st.text_input("Destination Airport", placeholder="e.g., SIN", key="opt_dest")
            to_date = st.date_input("To Date", value=None, key="opt_to_date")
        
        # Load and display available flights
        if st.button("🔍 Load Available Flights", disabled=not st.session_state.get('flights_loaded', False)):
            try:
                params = {}
                if origin_filter:
                    params['origin'] = origin_filter
                if dest_filter:
                    params['destination'] = dest_filter
                if from_date:
                    params['from_date'] = from_date.strftime('%Y-%m-%d')
                if to_date:
                    params['to_date'] = to_date.strftime('%Y-%m-%d')
                
                response = requests.get(f"{API_BASE_URL}/flights/utilization", params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    st.session_state['available_flights'] = data['utilization']
                    if data['total'] > 0:
                        st.success(f"Found {data['total']} flights")
                    else:
                        st.warning("No flights found matching the criteria")
                else:
                    st.error(f"Error loading flights: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to API: {str(e)}")
        
        # Display available flights
        if 'available_flights' in st.session_state and st.session_state['available_flights']:
            flights_df = pd.DataFrame(st.session_state['available_flights'])
            display_cols = ['flight_number', 'flight_date', 'origin', 'destination', 
                          'available_weight_kg', 'available_volume_m3', 
                          'weight_utilization_pct', 'is_near_full']
            st.dataframe(flights_df[display_cols], use_container_width=True, height=200)
        
        st.divider()
        
        # Cargo Requests Section
        st.subheader("📦 Cargo Requests")
        
        with st.expander("➕ Add Cargo Request", expanded=True):
            req_col1, req_col2 = st.columns(2)
            
            with req_col1:
                req_weight = st.number_input("Weight (kg)", min_value=1.0, value=100.0, key="req_weight")
                req_volume = st.number_input("Volume (m³)", min_value=0.1, value=1.0, step=0.1, key="req_volume")
            
            with req_col2:
                req_priority = st.selectbox(
                    "Priority (1=Low, 5=High)", 
                    options=[1, 2, 3, 4, 5], 
                    index=2, 
                    key="req_priority",
                    help="Higher priority cargo gets allocated to earlier flights"
                )
                req_customer = st.selectbox(
                    "Customer Type", 
                    options=["standard", "premium", "spot"], 
                    key="req_customer"
                )
            
            if st.button("➕ Add Request"):
                st.session_state['cargo_requests'].append({
                    'request_id': f"REQ{len(st.session_state['cargo_requests']) + 1:03d}",
                    'weight': req_weight,
                    'volume': req_volume,
                    'priority': req_priority,
                    'customer_type': req_customer
                })
                st.rerun()
        
        # Display current requests
        if st.session_state['cargo_requests']:
            st.markdown(f"**Current Requests ({len(st.session_state['cargo_requests'])})**")
            
            requests_df = pd.DataFrame(st.session_state['cargo_requests'])
            st.dataframe(requests_df, use_container_width=True)
            
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("🗑️ Clear All Requests"):
                    st.session_state['cargo_requests'] = []
                    st.session_state['optimization_result'] = None
                    st.rerun()
            with btn_col2:
                if st.button("🔄 Reset Allocations"):
                    try:
                        requests.post(f"{API_BASE_URL}/marketplace/reset")
                        st.success("Allocations reset!")
                        if 'available_flights' in st.session_state:
                            del st.session_state['available_flights']
                    except:
                        pass
        
        st.divider()
        
        # Run Optimization
        st.subheader("🚀 Run Optimization")
        
        if st.button("🚀 Allocate Cargo to Flights", type="primary", 
                    disabled=len(st.session_state.get('cargo_requests', [])) == 0):
            try:
                # Build request
                opt_request = {
                    "cargo_requests": st.session_state['cargo_requests'],
                    "commit": True
                }
                if origin_filter:
                    opt_request['origin'] = origin_filter
                if dest_filter:
                    opt_request['destination'] = dest_filter
                if from_date:
                    opt_request['from_date'] = from_date.strftime('%Y-%m-%d')
                if to_date:
                    opt_request['to_date'] = to_date.strftime('%Y-%m-%d')
                
                response = requests.post(
                    f"{API_BASE_URL}/marketplace/optimize",
                    json=opt_request
                )
                
                if response.status_code == 200:
                    st.session_state['optimization_result'] = response.json()
                    st.rerun()
                else:
                    st.error(f"Error: {response.text}")
            
            except Exception as e:
                st.error(f"Failed to connect to API: {str(e)}")
        
        # Display Optimization Results
        if st.session_state.get('optimization_result'):
            result = st.session_state['optimization_result']
            
            st.success("✅ Optimization Complete!")
            
            # Statistics
            stats = result['statistics']
            
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            with stat_col1:
                st.metric("Total Weight Allocated", f"{stats['total_allocated_weight']:.1f} kg")
            with stat_col2:
                st.metric("Flights Used", stats['flights_used'])
            with stat_col3:
                st.metric("Allocated", f"{stats['allocated_count']}/{stats['total_requests']}")
            
            # Allocation Results by Flight
            st.subheader("📋 Allocation Results")
            
            allocations = result['allocations']
            allocated = [a for a in allocations if a['allocated']]
            rejected = [a for a in allocations if not a['allocated']]
            
            if allocated:
                st.markdown("**✅ Allocated Cargo:**")
                allocated_df = pd.DataFrame(allocated)
                display_cols = ['request_id', 'weight', 'volume', 'flight_number', 'flight_date']
                st.dataframe(allocated_df[display_cols], use_container_width=True)
            
            if rejected:
                st.markdown("**❌ Rejected (No capacity):**")
                rejected_df = pd.DataFrame(rejected)
                st.dataframe(rejected_df[['request_id', 'weight', 'volume', 'reason']], use_container_width=True)
            
            # Flight Updates
            if result.get('flight_updates'):
                st.subheader("✈️ Flight Updates (Database)")
                
                for update in result['flight_updates']:
                    with st.expander(f"Flight {update['flight_number']} on {update['flight_date']}"):
                        st.write(f"**Weight Added:** {update['weight_added']:.1f} kg")
                        st.write(f"**Volume Added:** {update['volume_added']:.1f} m³")
                        st.write(f"**Requests:** {', '.join(update['requests'])}")
                        if update.get('committed'):
                            st.success("✅ Committed to database")
            
            # Visualization
            if allocated:
                st.subheader("📊 Allocation by Flight")
                
                # Group by flight
                flight_summary = {}
                for a in allocated:
                    key = f"{a['flight_number']} ({a['flight_date']})"
                    if key not in flight_summary:
                        flight_summary[key] = {'weight': 0, 'volume': 0, 'count': 0}
                    flight_summary[key]['weight'] += a['weight']
                    flight_summary[key]['volume'] += a['volume']
                    flight_summary[key]['count'] += 1
                
                fig = go.Figure()
                
                flights = list(flight_summary.keys())
                weights = [flight_summary[f]['weight'] for f in flights]
                
                fig.add_trace(go.Bar(
                    name='Weight (kg)',
                    x=flights,
                    y=weights,
                    marker_color='#1f77b4',
                    text=[f"{w:.0f} kg" for w in weights],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title="Cargo Allocated per Flight",
                    xaxis_title="Flight",
                    yaxis_title="Weight (kg)",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💡 How It Works")
        st.info("""
        **Priority-Based Allocation:**
        
        ⭐ **Priority 5 (Highest)**
        - Gets earliest available flight
        - Premium customers
        
        ⭐ **Priority 3-4**
        - Standard allocation
        - Next available flight
        
        ⭐ **Priority 1-2 (Lowest)**
        - If flight is near full (>85%)
        - Bulky cargo moves to later flights
        
        ---
        
        **Database Updates:**
        - `gross_weight_cargo_kg` is updated
        - Available capacity recalculated
        - Changes persist across sessions
        """)
        
        st.divider()
        
        # Dynamic pricing suggestion
        st.subheader("💵 Pricing Suggestion")
        
        total_capacity = 0
        if 'available_flights' in st.session_state:
            total_capacity = sum(f['available_weight_kg'] for f in st.session_state['available_flights'])
        
        st.write(f"Total Available: **{total_capacity:.0f} kg**")
        
        pred_demand = st.number_input("Predicted Demand (kg)", min_value=0.0, value=800.0, key="pricing_demand")
        confidence = st.slider("Confidence", 0.0, 1.0, 0.8, key="pricing_confidence")
        
        if st.button("Get Pricing Suggestion"):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/marketplace/pricing-suggestion",
                    json={
                        "available_capacity": total_capacity if total_capacity > 0 else 1000,
                        "predicted_demand": pred_demand,
                        "confidence": confidence
                    }
                )
                
                if response.status_code == 200:
                    pricing = response.json()
                    
                    st.metric("Suggested Price", f"${pricing['suggested_price_per_kg']:.2f}/kg")
                    st.metric("Strategy", pricing['pricing_strategy'].upper())
                    st.metric("Demand Ratio", f"{pricing['demand_ratio']:.2f}")
                    
                    st.info(f"**Market Balance:** {pricing['demand_supply_balance']}")
                
                else:
                    st.error(f"Error: {response.text}")
            
            except Exception as e:
                st.error(f"Failed to connect to API: {str(e)}")

if __name__ == "__main__":
    main()


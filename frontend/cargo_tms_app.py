"""
Cargo TMS (Transportation Management System) Dashboard
For Airport Cargo Personnel
"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random
import uuid
import time

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="AirCargo TMS",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional TMS look
st.markdown("""
<style>
    /* Main theme */
    :root {
        --primary-green: #4CAF50;
        --dark-bg: #1a1a2e;
        --card-bg: #f8f9fa;
        --border-color: #e0e0e0;
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Header styling */
    .tms-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 0 0 10px 10px;
        margin: -1rem -1rem 1rem -1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .tms-logo {
        font-size: 1.8rem;
        font-weight: bold;
        color: #4CAF50;
    }
    
    /* Flight card */
    .flight-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e8e8e8;
        margin-bottom: 1rem;
    }
    
    .flight-route {
        display: flex;
        align-items: center;
        justify-content: space-around;
        padding: 1rem 0;
    }
    
    .airport-code {
        font-size: 2rem;
        font-weight: bold;
        color: #1a1a2e;
    }
    
    .airport-name {
        font-size: 0.85rem;
        color: #666;
    }
    
    .flight-arrow {
        font-size: 1.5rem;
        color: #4CAF50;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .status-scheduled {
        background: #e3f2fd;
        color: #1565c0;
    }
    
    .status-loading {
        background: #fff3e0;
        color: #ef6c00;
    }
    
    .status-ready {
        background: #e8f5e9;
        color: #2e7d32;
    }
    
    .status-departed {
        background: #f3e5f5;
        color: #7b1fa2;
    }
    
    /* Capacity bars */
    .capacity-bar-container {
        background: #e9ecef;
        border-radius: 10px;
        height: 20px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .capacity-bar {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    .capacity-bar-weight {
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
    }
    
    .capacity-bar-volume {
        background: linear-gradient(90deg, #2196F3, #03A9F4);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        border-left: 4px solid #4CAF50;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1a1a2e;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #666;
        text-transform: uppercase;
    }
    
    /* Table styling */
    .cargo-table {
        width: 100%;
        border-collapse: collapse;
    }
    
    .cargo-table th {
        background: #f8f9fa;
        padding: 0.75rem;
        text-align: left;
        font-weight: 600;
        border-bottom: 2px solid #dee2e6;
    }
    
    .cargo-table td {
        padding: 0.75rem;
        border-bottom: 1px solid #e9ecef;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #1a1a2e;
        border-radius: 10px;
        padding: 0.25rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        color: #e0e0e0 !important;
        background: transparent !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff !important;
        background: rgba(76, 175, 80, 0.3) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: #4CAF50 !important;
        color: white !important;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: #1a1a2e;
    }
    
    /* ULD/Container cards */
    .uld-card {
        background: #f8f9fa;
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        padding: 0.75rem;
        text-align: center;
        margin: 0.25rem;
    }
    
    .uld-card.loaded {
        background: #e8f5e9;
        border-color: #4CAF50;
    }
    
    .uld-card.pending {
        background: #fff3e0;
        border-color: #ff9800;
    }
    
    /* Aircraft visualization */
    .aircraft-section {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1rem;
        position: relative;
    }
    
    /* Priority indicators */
    .priority-high {
        color: #d32f2f;
        font-weight: bold;
    }
    
    .priority-medium {
        color: #f57c00;
    }
    
    .priority-low {
        color: #388e3c;
    }
</style>
""", unsafe_allow_html=True)

# ============= Helper Functions =============

def check_api_health():
    """Check if API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.json().get("model_loaded", False)
    except:
        return False

def get_mock_flights():
    """Generate mock flight data for demonstration."""
    airports = [
        ("EKAH", "Aarhus"),
        ("KBML", "Berlin"),
        ("OPKC", "Karachi"),
        ("FAUP", "Upington"),
        ("WSSS", "Singapore"),
        ("KLAX", "Los Angeles"),
        ("EGLL", "London Heathrow"),
        ("VHHH", "Hong Kong"),
    ]
    
    aircraft_types = ["A300-B4", "B737-800F", "A330-300F", "B777F", "A350F"]
    statuses = ["Scheduled", "Loading", "Ready", "Departed"]
    
    flights = []
    base_time = datetime.now()
    
    for i in range(8):
        origin_idx = i % len(airports)
        dest_idx = (i + 1 + random.randint(1, 3)) % len(airports)
        
        flight = {
            "flight_id": f"FL{1493018 + i}",
            "flight_number": f"AC{100 + i}",
            "origin": airports[origin_idx][0],
            "origin_name": airports[origin_idx][1],
            "destination": airports[dest_idx][0],
            "destination_name": airports[dest_idx][1],
            "aircraft_type": random.choice(aircraft_types),
            "tail_number": f"9M-{random.choice(['XXA', 'XXB', 'XXC', 'XXD'])}{i}",
            "reg": f"88{988 + i}",
            "scheduled_departure": (base_time + timedelta(hours=i * 2)).strftime("%H:%M"),
            "scheduled_arrival": (base_time + timedelta(hours=i * 2 + 4)).strftime("%H:%M"),
            "date": base_time.strftime("%d/%m/%Y"),
            "status": statuses[min(i, len(statuses) - 1)],
            "passenger_count": random.randint(120, 280),
            "max_weight": random.randint(4000, 8000),
            "max_volume": random.randint(30, 60),
            "current_cargo_weight": random.randint(1000, 3000),
            "current_cargo_volume": random.randint(8, 25),
            "baggage_estimate": random.randint(800, 2000),
            "uld_count": random.randint(4, 12),
            "priority_cargo": random.randint(0, 3),
        }
        flights.append(flight)
    
    return flights

def get_mock_cargo_requests():
    """Generate mock cargo booking requests."""
    customers = ["Shopee Express", "DHL", "FedEx", "UPS", "Amazon Logistics", "Local Shipper"]
    cargo_types = ["General", "Express", "Perishable", "Dangerous Goods", "Valuables", "Live Animals"]
    
    requests = []
    for i in range(random.randint(5, 12)):
        req = {
            "request_id": f"CRQ-{uuid.uuid4().hex[:6].upper()}",
            "customer": random.choice(customers),
            "cargo_type": random.choice(cargo_types),
            "weight": round(random.uniform(50, 500), 1),
            "volume": round(random.uniform(0.5, 5), 2),
            "priority": random.randint(1, 5),
            "revenue_per_kg": round(random.uniform(1.5, 4.0), 2),
            "status": random.choice(["Pending", "Accepted", "Loading"]),
            "deadline": (datetime.now() + timedelta(hours=random.randint(1, 8))).strftime("%H:%M"),
        }
        requests.append(req)
    
    return requests

def get_mock_uld_containers():
    """Generate mock ULD container data."""
    uld_types = ["LD3", "LD6", "LD9", "PMC", "AKE", "RKN"]
    
    containers = []
    for i in range(random.randint(8, 16)):
        container = {
            "uld_id": f"{random.choice(uld_types)}-{random.randint(1000, 9999)}",
            "type": random.choice(uld_types),
            "status": random.choice(["Empty", "Loading", "Full", "Sealed"]),
            "weight": round(random.uniform(0, 1500), 1),
            "max_weight": 1587,
            "destination": random.choice(["KBML", "OPKC", "FAUP", "WSSS"]),
            "contents": random.randint(0, 8),
        }
        containers.append(container)
    
    return containers

# ============= Main Application =============

def main():
    """Main application entry point."""
    
    # Header
    col_header1, col_header2, col_header3 = st.columns([2, 6, 2])
    
    with col_header1:
        st.markdown("### ✈️ **AirCargo TMS**")
    
    with col_header2:
        # Navigation tabs in header
        tabs = st.tabs(["🏠 Home", "📋 Mesh Flights", "🔄 Flight Services", "📦 Cargo Planning", "📊 Analytics"])
    
    with col_header3:
        st.markdown(f"**{datetime.now().strftime('%H:%M')}** | {datetime.now().strftime('%d %b %Y')}")
    
    # Check API Status
    api_status = check_api_health()
    
    # Tab content
    with tabs[0]:
        show_home_dashboard(api_status)
    
    with tabs[1]:
        show_mesh_flights()
    
    with tabs[2]:
        show_flight_services(api_status)
    
    with tabs[3]:
        show_cargo_planning(api_status)
    
    with tabs[4]:
        show_analytics(api_status)


def show_home_dashboard(api_status):
    """Home dashboard with overview."""
    
    # Real-time simulation toggle
    auto_refresh = st.toggle("🔄 Auto-refresh (3s)", value=False, key="home_auto_refresh")
    
    # API Status Banner
    if not api_status:
        st.warning("⚠️ Forecasting API offline. Running in demo mode with simulated data.")
    else:
        st.success("✅ All systems operational | Forecasting API connected")
    
    # Quick Stats Row
    st.markdown("### 📊 Today's Overview")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Active Flights", "12", delta="3 departing")
    
    with col2:
        st.metric("Total Cargo", "45,230 kg", delta="+2,100 kg")
    
    with col3:
        st.metric("Capacity Util.", "78%", delta="5%")
    
    with col4:
        st.metric("Pending Requests", "23", delta="-4")
    
    with col5:
        st.metric("Revenue Today", "$89,450", delta="+$12,300")
    
    with col6:
        st.metric("On-Time Rate", "94%", delta="2%")
    
    st.divider()
    
    # Two column layout
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.markdown("### 🛫 Active Flights")
        
        flights = get_mock_flights()[:4]
        
        for flight in flights:
            with st.container():
                # Flight card header
                fcol1, fcol2, fcol3, fcol4 = st.columns([2, 3, 2, 1])
                
                with fcol1:
                    st.markdown(f"**{flight['flight_number']}** | {flight['aircraft_type']}")
                    st.caption(f"Tail: {flight['tail_number']} | REG: {flight['reg']}")
                
                with fcol2:
                    st.markdown(f"**{flight['origin']}** → **{flight['destination']}**")
                    st.caption(f"{flight['origin_name']} to {flight['destination_name']}")
                
                with fcol3:
                    st.markdown(f"🕐 {flight['scheduled_departure']} - {flight['scheduled_arrival']}")
                    st.caption(flight['date'])
                
                with fcol4:
                    status_colors = {
                        "Scheduled": "🔵",
                        "Loading": "🟡",
                        "Ready": "🟢",
                        "Departed": "🟣"
                    }
                    st.markdown(f"{status_colors.get(flight['status'], '⚪')} **{flight['status']}**")
                
                # Capacity progress bars
                weight_pct = (flight['current_cargo_weight'] + flight['baggage_estimate']) / flight['max_weight'] * 100
                volume_pct = flight['current_cargo_volume'] / flight['max_volume'] * 100
                
                pcol1, pcol2 = st.columns(2)
                with pcol1:
                    st.progress(min(weight_pct / 100, 1.0), text=f"Weight: {weight_pct:.0f}%")
                with pcol2:
                    st.progress(min(volume_pct / 100, 1.0), text=f"Volume: {volume_pct:.0f}%")
                
                st.divider()
    
    with col_right:
        st.markdown("### 📦 Pending Cargo Requests")
        
        requests = get_mock_cargo_requests()[:5]
        
        for req in requests:
            with st.container():
                rcol1, rcol2, rcol3 = st.columns([2, 2, 1])
                
                with rcol1:
                    priority_icons = {5: "🔴", 4: "🟠", 3: "🟡", 2: "🟢", 1: "⚪"}
                    st.markdown(f"{priority_icons.get(req['priority'], '⚪')} **{req['request_id']}**")
                    st.caption(req['customer'])
                
                with rcol2:
                    st.markdown(f"**{req['weight']} kg** | {req['volume']} m³")
                    st.caption(f"{req['cargo_type']} • Due {req['deadline']}")
                
                with rcol3:
                    st.markdown(f"${req['revenue_per_kg']}/kg")
                
            st.divider()
        
        if st.button("View All Requests →", use_container_width=True):
            st.info("Navigate to Cargo Planning tab")
    
    # Auto-refresh: wait 3 seconds then rerun
    if auto_refresh:
        time.sleep(3)
        st.rerun()


def show_mesh_flights():
    """Display mesh/network flights view."""
    st.markdown("### 📋 Mesh Flights - Network Overview")
    
    # Filter controls
    fcol1, fcol2, fcol3, fcol4 = st.columns(4)
    
    with fcol1:
        st.selectbox("Origin", ["All", "EKAH", "KBML", "OPKC", "WSSS"])
    
    with fcol2:
        st.selectbox("Destination", ["All", "FAUP", "KLAX", "EGLL", "VHHH"])
    
    with fcol3:
        st.selectbox("Status", ["All", "Scheduled", "Loading", "Ready", "Departed"])
    
    with fcol4:
        st.date_input("Date", datetime.now())
    
    st.divider()
    
    # Flights table
    flights = get_mock_flights()
    
    # Convert to DataFrame
    df = pd.DataFrame(flights)
    df_display = df[['flight_number', 'origin', 'destination', 'aircraft_type', 
                     'scheduled_departure', 'status', 'passenger_count', 
                     'current_cargo_weight', 'max_weight']].copy()
    df_display.columns = ['Flight', 'Origin', 'Dest', 'Aircraft', 'Departure', 
                          'Status', 'PAX', 'Cargo (kg)', 'Max (kg)']
    
    # Add utilization column
    df_display['Util %'] = ((df['current_cargo_weight'] + df['baggage_estimate']) / df['max_weight'] * 100).round(1)
    
    # Style the dataframe
    st.dataframe(
        df_display,
        use_container_width=True,
        height=400,
        column_config={
            "Util %": st.column_config.ProgressColumn(
                "Utilization",
                help="Cargo capacity utilization",
                min_value=0,
                max_value=100,
            )
        }
    )
    
    # Selected flight details
    st.markdown("### ✈️ Selected Flight Details")
    
    selected_flight = st.selectbox(
        "Select a flight for details",
        options=[f"{f['flight_number']} - {f['origin']} → {f['destination']}" for f in flights]
    )
    
    if selected_flight:
        flight_idx = [f"{f['flight_number']} - {f['origin']} → {f['destination']}" for f in flights].index(selected_flight)
        flight = flights[flight_idx]
        
        # Flight detail tabs
        detail_tabs = st.tabs(["Summary", "Services", "Timing", "Cargo", "Plan", "ULDs", "Crew", "Weather"])
        
        with detail_tabs[0]:  # Summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Flight Information**")
                st.write(f"Flight: {flight['flight_number']}")
                st.write(f"Aircraft: {flight['aircraft_type']}")
                st.write(f"Registration: {flight['reg']}")
                st.write(f"Tail: {flight['tail_number']}")
            
            with col2:
                st.markdown("**Route**")
                st.write(f"From: {flight['origin']} ({flight['origin_name']})")
                st.write(f"To: {flight['destination']} ({flight['destination_name']})")
                st.write(f"Departure: {flight['scheduled_departure']}")
                st.write(f"Arrival: {flight['scheduled_arrival']}")
            
            with col3:
                st.markdown("**Capacity**")
                st.write(f"Passengers: {flight['passenger_count']}")
                st.write(f"Max Cargo Weight: {flight['max_weight']} kg")
                st.write(f"Max Cargo Volume: {flight['max_volume']} m³")
                st.write(f"ULD Positions: {flight['uld_count']}")
        
        with detail_tabs[3]:  # Cargo
            ccol1, ccol2 = st.columns(2)
            
            with ccol1:
                st.markdown("**Current Load**")
                st.metric("Cargo Weight", f"{flight['current_cargo_weight']} kg")
                st.metric("Baggage Estimate", f"{flight['baggage_estimate']} kg")
                st.metric("Total Load", f"{flight['current_cargo_weight'] + flight['baggage_estimate']} kg")
            
            with ccol2:
                st.markdown("**Remaining Capacity**")
                remaining = flight['max_weight'] - flight['current_cargo_weight'] - flight['baggage_estimate']
                st.metric("Available Weight", f"{remaining} kg")
                st.metric("Available Volume", f"{flight['max_volume'] - flight['current_cargo_volume']} m³")
                st.metric("Priority Cargo", f"{flight['priority_cargo']} items")
        
        with detail_tabs[5]:  # ULDs
            st.markdown("**ULD/Container Status**")
            
            containers = get_mock_uld_containers()
            
            # Display ULDs in grid
            uld_cols = st.columns(4)
            for i, container in enumerate(containers[:8]):
                with uld_cols[i % 4]:
                    status_color = {
                        "Empty": "⚪",
                        "Loading": "🟡",
                        "Full": "🟢",
                        "Sealed": "🔵"
                    }
                    st.markdown(f"""
                    **{container['uld_id']}**  
                    {status_color.get(container['status'], '⚪')} {container['status']}  
                    {container['weight']}/{container['max_weight']} kg  
                    → {container['destination']}
                    """)


def show_flight_services(api_status):
    """Flight services status and management."""
    
    # Real-time simulation toggle
    auto_refresh = st.toggle("🔄 Auto-refresh (3s)", value=False, key="services_auto_refresh")
    
    st.markdown("### 🔄 Flight Services Status")
    
    # Select flight
    flights = get_mock_flights()
    selected = st.selectbox(
        "Select Flight",
        options=[f"{f['flight_number']} | {f['origin']}-{f['destination']} | {f['scheduled_departure']}" for f in flights]
    )
    
    if selected:
        flight_idx = [f"{f['flight_number']} | {f['origin']}-{f['destination']} | {f['scheduled_departure']}" for f in flights].index(selected)
        flight = flights[flight_idx]
        
        st.divider()
        
        # Service status grid
        services = [
            ("Fueling", "✅ Complete", 100),
            ("Catering", "🔄 In Progress", 65),
            ("Cleaning", "✅ Complete", 100),
            ("Cargo Loading", "🔄 In Progress", 45),
            ("Baggage Loading", "⏳ Pending", 0),
            ("Safety Check", "✅ Complete", 100),
            ("Crew Boarding", "⏳ Pending", 0),
            ("Ground Power", "✅ Connected", 100),
        ]
        
        cols = st.columns(4)
        for i, (service, status, progress) in enumerate(services):
            with cols[i % 4]:
                st.markdown(f"**{service}**")
                st.progress(progress / 100)
                st.caption(status)
        
        st.divider()
        
        # Cargo forecast section
        st.markdown("### 📊 Cargo Capacity Forecast")
        
        if api_status:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**Flight Parameters**")
                passenger_count = st.number_input("Passenger Count", value=flight['passenger_count'], min_value=1, max_value=500)
                existing_cargo = st.number_input("Current Cargo (kg)", value=float(flight['current_cargo_weight']), min_value=0.0)
                existing_volume = st.number_input("Current Volume (m³)", value=float(flight['current_cargo_volume']), min_value=0.0)
                
                if st.button("🔮 Get AI Forecast", type="primary"):
                    try:
                        flight_data = {
                            "passenger_count": passenger_count,
                            "year": 2024,
                            "month": datetime.now().month,
                            "day_of_week": datetime.now().weekday(),
                            "day_of_month": datetime.now().day,
                            "is_weekend": 1 if datetime.now().weekday() >= 5 else 0,
                            "aircraft_type": flight['aircraft_type'],
                            "tail_number": flight['tail_number'],
                            "origin": flight['origin'],
                            "destination": flight['destination'],
                            "existing_cargo_weight_kg": existing_cargo,
                            "existing_cargo_volume_m3": existing_volume,
                            "group_travel_ratio": 0.2,
                            "holiday_flag": 0,
                            "delay_probability": 0.1,
                            "weather_index": 0.7,
                            "fuel_weight_kg": 5000,
                            "fuel_price_per_kg": 0.8,
                            "cargo_price_per_kg": 2.0,
                            "aircraft_type_encoded": 0,
                            "tail_number_encoded": 0,
                            "origin_encoded": 0,
                            "destination_encoded": 0,
                        }
                        
                        response = requests.post(f"{API_BASE_URL}/predict", json=flight_data)
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state['forecast_result'] = result
                            st.success("✅ Forecast generated!")
                        else:
                            st.error(f"API Error: {response.text}")
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")
            
            with col2:
                if 'forecast_result' in st.session_state:
                    result = st.session_state['forecast_result']
                    
                    # Display forecast results
                    mcol1, mcol2, mcol3 = st.columns(3)
                    
                    with mcol1:
                        st.metric(
                            "Predicted Baggage",
                            f"{result['predicted_baggage']:.0f} kg",
                            help="AI-predicted baggage weight based on passenger count"
                        )
                    
                    with mcol2:
                        st.metric(
                            "Cargo Demand",
                            f"{result.get('predicted_cargo_demand', 0):.0f} kg",
                            help="Predicted cargo booking demand"
                        )
                    
                    with mcol3:
                        confidence_icon = "🟢" if result['confidence'] > 0.7 else "🟡" if result['confidence'] > 0.4 else "🔴"
                        st.metric(
                            "Remaining Capacity",
                            f"{result['remaining_cargo']:.0f} kg",
                            delta=f"{confidence_icon} {result['confidence']:.0%} confidence"
                        )
                    
                    # Visualization
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        name='Baggage',
                        x=['Capacity Breakdown'],
                        y=[result['predicted_baggage']],
                        marker_color='#FF9800',
                        error_y=dict(
                            type='data',
                            array=[result['predicted_baggage_upper'] - result['predicted_baggage']],
                            arrayminus=[result['predicted_baggage'] - result['predicted_baggage_lower']]
                        )
                    ))
                    
                    fig.add_trace(go.Bar(
                        name='Current Cargo',
                        x=['Capacity Breakdown'],
                        y=[existing_cargo],
                        marker_color='#2196F3'
                    ))
                    
                    fig.add_trace(go.Bar(
                        name='Remaining Capacity',
                        x=['Capacity Breakdown'],
                        y=[result['remaining_cargo']],
                        marker_color='#4CAF50',
                        error_y=dict(
                            type='data',
                            array=[result['remaining_cargo_upper'] - result['remaining_cargo']],
                            arrayminus=[result['remaining_cargo'] - result['remaining_cargo_lower']]
                        )
                    ))
                    
                    fig.update_layout(
                        barmode='stack',
                        title="Cargo Capacity Allocation",
                        yaxis_title="Weight (kg)",
                        height=350,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("🔌 Connect to the Forecasting API for AI-powered predictions")
            st.code("uvicorn backend.main:app --reload")
    
    # Auto-refresh: wait 3 seconds then rerun
    if auto_refresh:
        time.sleep(3)
        st.rerun()


def show_cargo_planning(api_status):
    """Cargo planning and optimization."""
    st.markdown("### 📦 Cargo Planning & Optimization")
    
    # Select flight
    flights = get_mock_flights()
    selected = st.selectbox(
        "Select Flight for Planning",
        options=[f"{f['flight_number']} | {f['origin']}-{f['destination']}" for f in flights],
        key="cargo_planning_flight"
    )
    
    if selected:
        flight_idx = [f"{f['flight_number']} | {f['origin']}-{f['destination']}" for f in flights].index(selected)
        flight = flights[flight_idx]
        
        col_left, col_right = st.columns([3, 2])
        
        with col_left:
            st.markdown("#### 📋 Cargo Booking Requests")
            
            # Initialize cargo requests in session state
            if 'tms_cargo_requests' not in st.session_state:
                st.session_state['tms_cargo_requests'] = get_mock_cargo_requests()
            
            requests = st.session_state['tms_cargo_requests']
            
            # Display as editable table
            df_requests = pd.DataFrame(requests)
            
            edited_df = st.data_editor(
                df_requests[['request_id', 'customer', 'cargo_type', 'weight', 'volume', 'priority', 'revenue_per_kg', 'status']],
                use_container_width=True,
                num_rows="dynamic",
                column_config={
                    "priority": st.column_config.SelectboxColumn(
                        "Priority",
                        options=[1, 2, 3, 4, 5],
                        required=True
                    ),
                    "status": st.column_config.SelectboxColumn(
                        "Status",
                        options=["Pending", "Accepted", "Rejected", "Loading"],
                        required=True
                    )
                }
            )
            
            # Optimization controls
            st.markdown("#### 🎯 Optimization")
            
            ocol1, ocol2, ocol3 = st.columns(3)
            
            with ocol1:
                available_weight = st.number_input(
                    "Available Weight (kg)",
                    value=float(flight['max_weight'] - flight['current_cargo_weight'] - flight['baggage_estimate']),
                    min_value=0.0
                )
            
            with ocol2:
                available_volume = st.number_input(
                    "Available Volume (m³)",
                    value=float(flight['max_volume'] - flight['current_cargo_volume']),
                    min_value=0.0
                )
            
            with ocol3:
                strategy = st.selectbox(
                    "Strategy",
                    options=["balanced", "revenue_max", "utilization_max", "priority_first"],
                    format_func=lambda x: {
                        "balanced": "⚖️ Balanced",
                        "revenue_max": "💰 Max Revenue",
                        "utilization_max": "📦 Max Utilization",
                        "priority_first": "⭐ Priority First"
                    }[x]
                )
            
            if st.button("🚀 Run Optimization", type="primary"):
                if api_status:
                    try:
                        # Prepare cargo requests
                        cargo_requests = []
                        for _, row in edited_df.iterrows():
                            if row['status'] == 'Pending':
                                cargo_requests.append({
                                    'request_id': row['request_id'],
                                    'weight': float(row['weight']),
                                    'volume': float(row['volume']),
                                    'priority': int(row['priority']),
                                    'revenue_per_kg': float(row['revenue_per_kg']),
                                    'customer_type': 'standard'
                                })
                        
                        response = requests.post(
                            f"{API_BASE_URL}/marketplace/optimize",
                            json={
                                "available_weight": available_weight,
                                "available_volume": available_volume,
                                "cargo_requests": cargo_requests,
                                "strategy": strategy
                            }
                        )
                        
                        if response.status_code == 200:
                            st.session_state['optimization_result'] = response.json()
                            st.success("✅ Optimization complete!")
                        else:
                            st.error(f"API Error: {response.text}")
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")
                else:
                    st.warning("API not connected. Running simulated optimization...")
                    # Simulate optimization result
                    st.session_state['optimization_result'] = {
                        'allocations': [
                            {'request_id': r['request_id'], 'allocated': random.choice([True, True, True, False]),
                             'weight': r['weight'], 'volume': r['volume'], 'revenue': r['weight'] * r['revenue_per_kg']}
                            for r in requests[:5]
                        ],
                        'statistics': {
                            'total_revenue': sum(r['weight'] * r['revenue_per_kg'] for r in requests[:4]),
                            'weight_utilization': 72.5,
                            'volume_utilization': 65.3,
                            'allocated_count': 4,
                            'rejected_count': 1,
                            'remaining_weight': available_weight * 0.275,
                            'remaining_volume': available_volume * 0.347,
                            'strategy': strategy
                        }
                    }
        
        with col_right:
            st.markdown("#### 📊 Optimization Results")
            
            if 'optimization_result' in st.session_state:
                result = st.session_state['optimization_result']
                stats = result['statistics']
                
                # Key metrics
                st.metric("Total Revenue", f"${stats['total_revenue']:.2f}")
                
                mcol1, mcol2 = st.columns(2)
                with mcol1:
                    st.metric("Weight Util.", f"{stats['weight_utilization']:.1f}%")
                with mcol2:
                    st.metric("Volume Util.", f"{stats['volume_utilization']:.1f}%")
                
                mcol3, mcol4 = st.columns(2)
                with mcol3:
                    st.metric("Allocated", stats['allocated_count'])
                with mcol4:
                    st.metric("Rejected", stats['rejected_count'])
                
                # Utilization chart
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='Weight',
                    x=['Utilization'],
                    y=[stats['weight_utilization']],
                    marker_color='#4CAF50'
                ))
                
                fig.add_trace(go.Bar(
                    name='Volume',
                    x=['Utilization'],
                    y=[stats['volume_utilization']],
                    marker_color='#2196F3'
                ))
                
                fig.update_layout(
                    barmode='group',
                    yaxis_range=[0, 100],
                    yaxis_title="Utilization %",
                    height=250,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Allocation details
                st.markdown("**Allocations:**")
                for alloc in result['allocations']:
                    icon = "✅" if alloc['allocated'] else "❌"
                    st.caption(f"{icon} {alloc['request_id']}: {alloc['weight']} kg")
            
            # Pricing suggestion
            st.markdown("---")
            st.markdown("#### 💵 Dynamic Pricing")
            
            if api_status:
                pred_demand = st.number_input("Predicted Demand (kg)", value=800.0, key="tms_pricing")
                
                if st.button("Get Pricing Suggestion"):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/marketplace/pricing-suggestion",
                            json={
                                "available_capacity": available_weight,
                                "predicted_demand": pred_demand,
                                "confidence": 0.8
                            }
                        )
                        
                        if response.status_code == 200:
                            pricing = response.json()
                            st.metric("Suggested Price", f"${pricing['suggested_price_per_kg']:.2f}/kg")
                            st.caption(f"Strategy: {pricing['pricing_strategy'].upper()}")
                            st.caption(f"Demand Ratio: {pricing['demand_ratio']:.2f}")
                    except Exception as e:
                        st.error(str(e))


def show_analytics(api_status):
    """Analytics and reporting dashboard."""
    st.markdown("### 📊 Analytics & Performance")
    
    # Date range selector
    col1, col2, col3 = st.columns([2, 2, 4])
    
    with col1:
        start_date = st.date_input("From", datetime.now() - timedelta(days=30))
    
    with col2:
        end_date = st.date_input("To", datetime.now())
    
    st.divider()
    
    # Performance metrics
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    
    with mcol1:
        st.metric("Total Flights", "342", delta="+12 vs last month")
    
    with mcol2:
        st.metric("Cargo Transported", "1.2M kg", delta="+8.5%")
    
    with mcol3:
        st.metric("Avg. Utilization", "76.3%", delta="+3.2%")
    
    with mcol4:
        st.metric("Revenue", "$2.4M", delta="+15.7%")
    
    st.divider()
    
    # Charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("#### 📈 Capacity Utilization Trend")
        
        # Generate sample data
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        utilization_data = pd.DataFrame({
            'Date': dates,
            'Weight Util %': [random.uniform(65, 90) for _ in range(len(dates))],
            'Volume Util %': [random.uniform(55, 85) for _ in range(len(dates))]
        })
        
        fig = px.line(utilization_data, x='Date', y=['Weight Util %', 'Volume Util %'],
                      title="Daily Capacity Utilization")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        st.markdown("#### 📊 Revenue by Customer Type")
        
        revenue_data = pd.DataFrame({
            'Customer': ['Shopee Express', 'DHL', 'FedEx', 'UPS', 'Amazon', 'Others'],
            'Revenue': [450000, 380000, 320000, 290000, 410000, 550000]
        })
        
        fig = px.pie(revenue_data, values='Revenue', names='Customer',
                     title="Revenue Distribution")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Route performance
    st.markdown("#### 🌍 Route Performance")
    
    routes_data = pd.DataFrame({
        'Route': ['EKAH-KBML', 'KBML-OPKC', 'OPKC-FAUP', 'WSSS-KLAX', 'KLAX-EGLL'],
        'Flights': [45, 38, 42, 28, 35],
        'Avg Weight (kg)': [3200, 4100, 3800, 5200, 4600],
        'Avg Utilization %': [78, 82, 75, 88, 84],
        'Revenue': [125000, 156000, 142000, 198000, 176000],
        'On-Time %': [94, 91, 88, 96, 92]
    })
    
    st.dataframe(
        routes_data,
        use_container_width=True,
        column_config={
            'Avg Utilization %': st.column_config.ProgressColumn(
                'Avg Utilization',
                min_value=0,
                max_value=100
            ),
            'On-Time %': st.column_config.ProgressColumn(
                'On-Time',
                min_value=0,
                max_value=100
            ),
            'Revenue': st.column_config.NumberColumn(
                'Revenue',
                format="$%d"
            )
        }
    )
    
    # Export options
    st.divider()
    col_export1, col_export2, col_export3 = st.columns(3)
    
    with col_export1:
        if st.button("📄 Export to PDF"):
            st.info("PDF export functionality")
    
    with col_export2:
        if st.button("📊 Export to Excel"):
            st.info("Excel export functionality")
    
    with col_export3:
        if st.button("📧 Email Report"):
            st.info("Email report functionality")


if __name__ == "__main__":
    main()

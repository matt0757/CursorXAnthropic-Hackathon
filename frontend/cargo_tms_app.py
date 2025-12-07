"""
PISSTMS - Passenger Integrated Smart Scheduling TMS
Transportation Management System Dashboard for Airport Cargo Personnel
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
    page_title="PISSTMS",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============= Authentication =============

def init_session_state():
    """Initialize session state variables."""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = ""
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "🏠 Home"

def login(username: str, password: str) -> bool:
    """Validate login credentials."""
    # Demo credentials
    if username == "demo" and password == "demo":
        st.session_state.logged_in = True
        st.session_state.username = username
        return True
    return False

def logout():
    """Log out the user."""
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.current_page = "🏠 Home"

def show_login_page():
    """Display the login page."""
    st.markdown("""
    <style>
        .login-container {
            max-width: 400px;
            margin: 100px auto;
            padding: 2rem;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("## ✈️ PISSTMS")
        st.markdown("### Login")
        st.markdown("---")
        
        # Auto-fill with demo credentials
        username = st.text_input("Username", value="demo", key="login_username")
        password = st.text_input("Password", value="demo", type="password", key="login_password")
        
        st.caption("💡 Demo credentials: **demo** / **demo**")
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🔐 Login", type="primary", use_container_width=True):
                if login(username, password):
                    st.success("✅ Login successful!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")
        
        with col_btn2:
            if st.button("🚀 Demo Login", use_container_width=True):
                if login("demo", "demo"):
                    st.success("✅ Demo login successful!")
                    time.sleep(0.5)
                    st.rerun()

def show_sidebar():
    """Display the sidebar navigation."""
    with st.sidebar:
        # Logo and branding
        st.markdown("## ✈️ PISSTMS")
        st.markdown("---")
        
        # User info
        st.markdown(f"👤 **{st.session_state.username}**")
        st.caption(f"🕐 {datetime.now().strftime('%H:%M')} | {datetime.now().strftime('%d %b %Y')}")
        
        st.markdown("---")
        
        # Navigation menu
        st.markdown("### Navigation")
        
        menu_items = [
            "🏠 Home",
            "📋 Mesh Flights",
            "🔄 Flight Services",
            "📦 Cargo Planning",
            "📊 Analytics"
        ]
        
        for item in menu_items:
            if st.button(item, use_container_width=True, 
                        type="primary" if st.session_state.current_page == item else "secondary"):
                st.session_state.current_page = item
                st.rerun()
        
        st.markdown("---")
        
        # API Status
        api_status = check_api_health()
        if api_status:
            st.success("🟢 API Connected")
        else:
            st.warning("🟡 API Offline")
        
        st.markdown("---")
        
        # Logout button
        if st.button("🚪 Logout", use_container_width=True):
            logout()
            st.rerun()
        
        # Footer
        st.markdown("---")
        st.caption("© 2024 PISSTMS")
        st.caption("🇲🇾 KUL Hub Operations")
    
    return api_status

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
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdown"] {
        color: #e0e0e0;
    }
    
    [data-testid="stSidebar"] h2 {
        color: #4CAF50 !important;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .stButton > button {
        background: transparent;
        border: 1px solid #4CAF50;
        color: #e0e0e0;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(76, 175, 80, 0.3);
        color: white;
    }
    
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: #4CAF50;
        color: white;
        border: none;
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

# KUL Hub - Kuala Lumpur International Airport
KUL_HUB = ("WMKK", "Kuala Lumpur (KUL)")

# Destination airports from/to KUL
DESTINATIONS = [
    ("WSSS", "Singapore (SIN)"),
    ("VHHH", "Hong Kong (HKG)"),
    ("VTBS", "Bangkok (BKK)"),
    ("RPLL", "Manila (MNL)"),
    ("WIII", "Jakarta (CGK)"),
    ("VDPP", "Phnom Penh (PNH)"),
    ("ZGGG", "Guangzhou (CAN)"),
    ("RJTT", "Tokyo (HND)"),
]

# Aircraft cargo zones
CARGO_ZONES = {
    "FWD": {"name": "Forward Hold", "position": "Zone A", "max_weight": 2000},
    "AFT": {"name": "Aft Hold", "position": "Zone B", "max_weight": 2500},
    "BULK": {"name": "Bulk Cargo", "position": "Zone C", "max_weight": 1000},
    "MAIN": {"name": "Main Deck", "position": "Zone D", "max_weight": 5000},
}

def check_api_health():
    """Check if API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.json().get("model_loaded", False)
    except:
        return False

def get_mock_flights():
    """Generate mock flight data - all from/to KUL Malaysia with multiple flights per route."""
    aircraft_types = ["A330-300F", "B777F", "A350F", "B747-8F"]
    statuses = ["Scheduled", "Loading", "Ready", "Departed"]
    
    flights = []
    base_time = datetime.now().replace(hour=6, minute=0, second=0, microsecond=0)
    
    # Define routes with multiple flights per day
    # Format: (destination, direction, times_per_day)
    routes = [
        (DESTINATIONS[0], "outbound", 3),  # Singapore - 3 flights out
        (DESTINATIONS[0], "inbound", 2),   # Singapore - 2 flights in
        (DESTINATIONS[1], "outbound", 2),  # Hong Kong - 2 flights out
        (DESTINATIONS[1], "inbound", 2),   # Hong Kong - 2 flights in
        (DESTINATIONS[2], "outbound", 2),  # Bangkok - 2 flights out
        (DESTINATIONS[2], "inbound", 1),   # Bangkok - 1 flight in
        (DESTINATIONS[3], "outbound", 1),  # Manila - 1 flight out
        (DESTINATIONS[4], "inbound", 1),   # Jakarta - 1 flight in
    ]
    
    flight_idx = 0
    for dest, direction, count in routes:
        for i in range(count):
            if direction == "outbound":
                origin = KUL_HUB
                destination = dest
                flight_prefix = "MH"
            else:
                origin = dest
                destination = KUL_HUB
                flight_prefix = "MH"
            
            # Spread flights throughout the day
            hour_offset = (flight_idx * 2) % 18  # Flights from 6 AM to midnight
            departure_time = base_time + timedelta(hours=hour_offset)
            
            # Flight duration based on destination
            duration_hours = random.randint(2, 5)
            arrival_time = departure_time + timedelta(hours=duration_hours)
            
            # Determine status based on time
            now = datetime.now()
            if departure_time > now + timedelta(hours=2):
                status = "Scheduled"
            elif departure_time > now:
                status = "Loading"
            elif departure_time > now - timedelta(hours=1):
                status = "Ready"
            else:
                status = random.choice(["Loading", "Ready", "Departed"])
            
            aircraft = random.choice(aircraft_types)
            max_weight = {"A330-300F": 6000, "B777F": 8000, "A350F": 7000, "B747-8F": 10000}[aircraft]
            max_volume = {"A330-300F": 45, "B777F": 60, "A350F": 50, "B747-8F": 80}[aircraft]
            
            flight = {
                "flight_id": f"FL{1493018 + flight_idx}",
                "flight_number": f"{flight_prefix}{800 + flight_idx}",
                "origin": origin[0],
                "origin_name": origin[1],
                "destination": destination[0],
                "destination_name": destination[1],
                "direction": direction,
                "route_key": f"{origin[0]}-{destination[0]}",
                "aircraft_type": aircraft,
                "tail_number": f"9M-{random.choice(['MRA', 'MRB', 'MRC', 'MRD', 'MRE'])}{flight_idx}",
                "reg": f"9M{flight_idx:03d}",
                "scheduled_departure": departure_time.strftime("%H:%M"),
                "scheduled_arrival": arrival_time.strftime("%H:%M"),
                "date": base_time.strftime("%d/%m/%Y"),
                "status": status,
                "passenger_count": random.randint(180, 350),
                "max_weight": max_weight,
                "max_volume": max_volume,
                "current_cargo_weight": random.randint(1000, int(max_weight * 0.5)),
                "current_cargo_volume": random.randint(8, int(max_volume * 0.5)),
                "baggage_estimate": random.randint(1000, 2500),
                "uld_count": random.randint(6, 14),
                "priority_cargo": random.randint(0, 5),
                "zones": list(CARGO_ZONES.keys()),
            }
            flights.append(flight)
            flight_idx += 1
    
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
    """Generate mock ULD container data for KUL routes."""
    uld_types = ["LD3", "LD6", "LD9", "PMC", "AKE", "RKN"]
    destinations = [d[0] for d in DESTINATIONS[:5]]
    
    containers = []
    for i in range(random.randint(10, 18)):
        dest = random.choice(destinations)
        container = {
            "uld_id": f"{random.choice(uld_types)}-{random.randint(1000, 9999)}",
            "type": random.choice(uld_types),
            "status": random.choice(["Empty", "Loading", "Full", "Sealed"]),
            "weight": round(random.uniform(0, 1500), 1),
            "max_weight": 1587,
            "destination": dest,
            "destination_name": next((d[1] for d in DESTINATIONS if d[0] == dest), dest),
            "contents": random.randint(0, 8),
            "zone": random.choice(list(CARGO_ZONES.keys())),
        }
        containers.append(container)
    
    return containers

def optimize_cargo_across_flights(cargo_requests, flights, strategy="balanced"):
    """Optimize cargo allocation across multiple flights going to the same destination."""
    allocations = []
    
    # Group flights by destination
    flights_by_dest = {}
    for flight in flights:
        if flight['direction'] == 'outbound':  # Only outbound from KUL
            dest = flight['destination']
            if dest not in flights_by_dest:
                flights_by_dest[dest] = []
            flights_by_dest[dest].append(flight)
    
    # Sort cargo by priority (high to low) and revenue
    sorted_cargo = sorted(cargo_requests, key=lambda x: (-x['priority'], -x['revenue_per_kg']))
    
    for cargo in sorted_cargo:
        if cargo['status'] != 'Pending':
            continue
            
        dest = cargo['destination']
        if dest not in flights_by_dest:
            allocations.append({
                **cargo,
                "allocated": False,
                "reason": "No flights to destination",
                "assigned_flight": None,
                "assigned_zone": None,
            })
            continue
        
        # Find best flight based on strategy
        available_flights = flights_by_dest[dest]
        best_flight = None
        best_zone = None
        
        for flight in available_flights:
            remaining_weight = flight['max_weight'] - flight['current_cargo_weight'] - flight['baggage_estimate']
            remaining_volume = flight['max_volume'] - flight['current_cargo_volume']
            
            if cargo['weight'] <= remaining_weight and cargo['volume'] <= remaining_volume:
                # Assign to zone based on cargo type and weight
                if cargo['cargo_type'] in ['Express', 'Pharmaceuticals'] or cargo['priority'] >= 4:
                    zone = "FWD"  # Priority cargo in forward hold
                elif cargo['weight'] > 500:
                    zone = "MAIN" if "MAIN" in flight['zones'] else "AFT"
                elif cargo['cargo_type'] == 'Perishable':
                    zone = "AFT"  # Perishables in aft hold (temperature controlled)
                else:
                    zone = random.choice(["AFT", "BULK"])
                
                if best_flight is None:
                    best_flight = flight
                    best_zone = zone
                elif strategy == "earliest":
                    if flight['scheduled_departure'] < best_flight['scheduled_departure']:
                        best_flight = flight
                        best_zone = zone
                elif strategy == "balanced":
                    # Prefer less loaded flights
                    current_util = (flight['current_cargo_weight'] / flight['max_weight'])
                    best_util = (best_flight['current_cargo_weight'] / best_flight['max_weight'])
                    if current_util < best_util:
                        best_flight = flight
                        best_zone = zone
        
        if best_flight:
            allocations.append({
                **cargo,
                "allocated": True,
                "assigned_flight": best_flight['flight_number'],
                "assigned_flight_time": best_flight['scheduled_departure'],
                "assigned_zone": best_zone,
                "zone_name": CARGO_ZONES[best_zone]['name'],
            })
            # Update flight capacity (simulation)
            best_flight['current_cargo_weight'] += cargo['weight']
            best_flight['current_cargo_volume'] += cargo['volume']
        else:
            allocations.append({
                **cargo,
                "allocated": False,
                "reason": "No capacity available",
                "assigned_flight": None,
                "assigned_zone": None,
            })
    
    return allocations

# ============= Main Application =============

def main():
    """Main application entry point."""
    
    # Initialize session state
    init_session_state()
    
    # Check if logged in
    if not st.session_state.logged_in:
        show_login_page()
        return
    
    # Show sidebar and get API status
    api_status = show_sidebar()
    
    # Page content based on current page
    current_page = st.session_state.current_page
    
    if current_page == "🏠 Home":
        show_home_dashboard(api_status)
    elif current_page == "📋 Mesh Flights":
        show_mesh_flights()
    elif current_page == "🔄 Flight Services":
        show_flight_services(api_status)
    elif current_page == "📦 Cargo Planning":
        show_cargo_planning(api_status)
    elif current_page == "📊 Analytics":
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
        
        pending_reqs = get_mock_cargo_requests()[:5]
        
        for req in pending_reqs:
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
    st.markdown("### 📋 Mesh Flights - KUL Hub Network")
    st.caption("🇲🇾 All flights operate from/to Kuala Lumpur International Airport (WMKK)")
    
    # Filter controls
    fcol1, fcol2, fcol3, fcol4 = st.columns(4)
    
    dest_options = ["All"] + [f"{d[0]} - {d[1]}" for d in DESTINATIONS]
    
    with fcol1:
        direction_filter = st.selectbox("Direction", ["All", "Outbound (KUL→)", "Inbound (→KUL)"])
    
    with fcol2:
        dest_filter = st.selectbox("Destination", dest_options)
    
    with fcol3:
        status_filter = st.selectbox("Status", ["All", "Scheduled", "Loading", "Ready", "Departed"])
    
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
    """Cargo planning and optimization across multiple flights to same destination."""
    st.markdown("### 📦 Multi-Flight Cargo Planning & Zone Assignment")
    st.caption("🇲🇾 Optimize cargo across all KUL outbound flights to the same destination")
    
    # Get all flights
    flights = get_mock_flights()
    
    # Group outbound flights by destination
    outbound_flights = [f for f in flights if f['direction'] == 'outbound']
    destinations = {}
    for f in outbound_flights:
        dest = f['destination']
        dest_name = f['destination_name']
        if dest not in destinations:
            destinations[dest] = {'name': dest_name, 'flights': []}
        destinations[dest]['flights'].append(f)
    
    # Destination selector
    dest_options = [f"{code} - {info['name']} ({len(info['flights'])} flights)" 
                    for code, info in destinations.items()]
    
    selected_dest = st.selectbox(
        "🎯 Select Destination (to see all flights)",
        options=dest_options,
        key="cargo_planning_dest"
    )
    
    if selected_dest:
        dest_code = selected_dest.split(" - ")[0]
        dest_flights = destinations[dest_code]['flights']
        
        # Show flights to this destination
        st.markdown(f"#### ✈️ Flights to {dest_code} ({len(dest_flights)} available)")
        
        flight_cols = st.columns(len(dest_flights))
        for i, flight in enumerate(dest_flights):
            with flight_cols[i]:
                weight_pct = (flight['current_cargo_weight'] + flight['baggage_estimate']) / flight['max_weight'] * 100
                remaining = flight['max_weight'] - flight['current_cargo_weight'] - flight['baggage_estimate']
                
                st.markdown(f"**{flight['flight_number']}**")
                st.caption(f"🕐 {flight['scheduled_departure']}")
                st.caption(f"✈️ {flight['aircraft_type']}")
                st.progress(min(weight_pct / 100, 1.0))
                st.caption(f"📦 {remaining:,.0f} kg available")
        
        st.divider()
        
        col_left, col_right = st.columns([3, 2])
        
        with col_left:
            st.markdown("#### 📋 Cargo Booking Requests")
            
            # Initialize cargo requests with destination in session state
            if 'tms_cargo_requests' not in st.session_state or st.session_state.get('cargo_dest') != dest_code:
                cargo_reqs = get_mock_cargo_requests()
                # Add destination to each cargo request
                for req in cargo_reqs:
                    req['destination'] = dest_code
                st.session_state['tms_cargo_requests'] = cargo_reqs
                st.session_state['cargo_dest'] = dest_code
            
            cargo_requests = st.session_state['tms_cargo_requests']
            
            # Display as editable table
            df_requests = pd.DataFrame(cargo_requests)
            
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
            st.markdown("#### 🎯 Multi-Flight Optimization")
            
            ocol1, ocol2 = st.columns(2)
            
            with ocol1:
                strategy = st.selectbox(
                    "Optimization Strategy",
                    options=["balanced", "earliest", "revenue_max", "priority_first"],
                    format_func=lambda x: {
                        "balanced": "⚖️ Balanced (spread load evenly)",
                        "earliest": "⏰ Earliest Flight First",
                        "revenue_max": "💰 Max Revenue",
                        "priority_first": "⭐ Priority First"
                    }[x]
                )
            
            with ocol2:
                # Calculate total available capacity across all flights
                total_available = sum(
                    f['max_weight'] - f['current_cargo_weight'] - f['baggage_estimate']
                    for f in dest_flights
                )
                st.metric("Total Available Capacity", f"{total_available:,.0f} kg", 
                         delta=f"Across {len(dest_flights)} flights")
            
            if st.button("🚀 Optimize Across All Flights", type="primary"):
                # Prepare cargo requests with destination
                pending_cargo = []
                for _, row in edited_df.iterrows():
                    if row['status'] == 'Pending':
                        pending_cargo.append({
                            'request_id': row['request_id'],
                            'customer': row['customer'],
                            'cargo_type': row['cargo_type'],
                            'weight': float(row['weight']),
                            'volume': float(row['volume']),
                            'priority': int(row['priority']),
                            'revenue_per_kg': float(row['revenue_per_kg']),
                            'status': 'Pending',
                            'destination': dest_code
                        })
                
                # Calculate naive scenario (all cargo to first flight only)
                first_flight = dest_flights[0]
                first_flight_capacity = first_flight['max_weight'] - first_flight['current_cargo_weight'] - first_flight['baggage_estimate']
                total_cargo_weight = sum(c['weight'] for c in pending_cargo)
                
                naive_overflow = max(0, total_cargo_weight - first_flight_capacity)
                naive_wasted = sum(
                    f['max_weight'] - f['current_cargo_weight'] - f['baggage_estimate']
                    for f in dest_flights[1:]
                )
                
                st.session_state['naive_scenario'] = {
                    'first_flight': first_flight['flight_number'],
                    'first_capacity': first_flight_capacity,
                    'total_cargo': total_cargo_weight,
                    'overflow': naive_overflow,
                    'wasted_capacity': naive_wasted
                }
                
                # Run multi-flight optimization
                # Create a copy of flights to avoid modifying the originals
                import copy
                flights_copy = copy.deepcopy(dest_flights)
                
                allocations = optimize_cargo_across_flights(pending_cargo, flights_copy, strategy)
                
                st.session_state['multi_flight_allocations'] = allocations
                st.session_state['optimized_dest'] = dest_code
                st.success("✅ Multi-flight optimization complete!")
                st.rerun()
        
        with col_right:
            st.markdown("#### 🗺️ Cargo Zone Guide")
            
            for zone_id, zone_info in CARGO_ZONES.items():
                st.markdown(f"**{zone_id}** - {zone_info['name']}")
                st.caption(f"{zone_info['position']} | Max: {zone_info['max_weight']:,} kg")
            
            st.markdown("---")
            st.markdown("#### 📍 Zone Assignment Logic")
            st.caption("🔴 Priority 4-5 & Express → **FWD** (First out)")
            st.caption("🟠 Heavy cargo (>500kg) → **MAIN/AFT**")
            st.caption("🟡 Perishables → **AFT** (Temperature)")
            st.caption("🟢 General cargo → **AFT/BULK**")
            
            st.markdown("---")
            st.markdown("#### 💡 Why Multi-Flight?")
            st.caption("Without optimization, all cargo goes to the first available flight, causing:")
            st.caption("❌ Overflow & rejected cargo")
            st.caption("❌ Wasted capacity on later flights")
            st.caption("✅ Optimization distributes load across all flights!")
        
        # Show optimization results
        if 'multi_flight_allocations' in st.session_state and st.session_state.get('optimized_dest') == dest_code:
            st.divider()
            
            allocations = st.session_state['multi_flight_allocations']
            allocated = [a for a in allocations if a.get('allocated')]
            not_allocated = [a for a in allocations if not a.get('allocated')]
            
            total_allocated = sum(a['weight'] for a in allocated)
            total_rejected = sum(a['weight'] for a in not_allocated)
            
            # Show Before vs After comparison only if there's meaningful data
            if 'naive_scenario' in st.session_state:
                naive = st.session_state['naive_scenario']
                
                st.markdown("### 📊 Optimization Results")
                
                # Only show comparison if there would have been overflow without optimization
                if naive['overflow'] > 0:
                    st.markdown("#### 🔄 Before vs After Optimization")
                    
                    comp_col1, comp_col2 = st.columns(2)
                    
                    with comp_col1:
                        st.markdown("**❌ Single Flight Approach**")
                        st.markdown(f"*All {naive['total_cargo']:,.0f} kg → {naive['first_flight']}*")
                        st.error(f"🚫 **{naive['overflow']:,.0f} kg** REJECTED (overflow)")
                        st.caption(f"First flight capacity: {naive['first_capacity']:,.0f} kg")
                        st.warning(f"💨 **{naive['wasted_capacity']:,.0f} kg** unused on other flights")
                    
                    with comp_col2:
                        st.markdown("**✅ Multi-Flight Optimization**")
                        flights_used = len(set(a['assigned_flight'] for a in allocated))
                        st.markdown(f"*Distributed across {flights_used} flight(s)*")
                        
                        if total_rejected > 0:
                            st.warning(f"⚠️ **{total_rejected:,.0f} kg** rejected (all flights full)")
                        else:
                            st.success(f"✅ **ALL {total_allocated:,.0f} kg** allocated!")
                        
                        # Calculate improvement
                        improvement = naive['overflow'] - total_rejected
                        if improvement > 0:
                            st.success(f"📈 **{improvement:,.0f} kg MORE** cargo accommodated!")
                    
                    st.divider()
                else:
                    # No overflow - simple summary
                    st.markdown("#### ✅ Allocation Summary")
                    
                    summary_col1, summary_col2, summary_col3 = st.columns(3)
                    
                    with summary_col1:
                        st.metric("Total Cargo", f"{naive['total_cargo']:,.0f} kg")
                    
                    with summary_col2:
                        flights_used = len(set(a['assigned_flight'] for a in allocated)) if allocated else 0
                        st.metric("Flights Used", f"{flights_used}")
                    
                    with summary_col3:
                        st.metric("Status", "✅ All Allocated" if total_rejected == 0 else f"⚠️ {total_rejected:.0f} kg rejected")
                    
                    if total_rejected == 0:
                        st.success(f"✅ All **{len(allocated)} shipments** ({total_allocated:,.0f} kg) successfully allocated!")
                    
                    st.divider()
            
            st.markdown("### 📋 Flight & Zone Assignments")
            
            allocations = st.session_state['multi_flight_allocations']
            allocated = [a for a in allocations if a.get('allocated')]
            not_allocated = [a for a in allocations if not a.get('allocated')]
            
            # Summary metrics
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            with mcol1:
                st.metric("Allocated", len(allocated))
            with mcol2:
                st.metric("Not Allocated", len(not_allocated))
            with mcol3:
                total_allocated_weight = sum(a['weight'] for a in allocated)
                st.metric("Total Weight", f"{total_allocated_weight:,.0f} kg")
            with mcol4:
                total_revenue = sum(a['weight'] * a['revenue_per_kg'] for a in allocated)
                st.metric("Est. Revenue", f"${total_revenue:,.0f}")
            
            # Group by flight for display
            st.markdown("#### ✅ Allocated Shipments by Flight & Zone")
            
            if allocated:
                # Group by flight
                by_flight = {}
                for alloc in allocated:
                    flight_num = alloc['assigned_flight']
                    if flight_num not in by_flight:
                        by_flight[flight_num] = {'FWD': [], 'AFT': [], 'BULK': [], 'MAIN': []}
                    zone = alloc.get('assigned_zone', 'BULK')
                    by_flight[flight_num][zone].append(alloc)
                
                for flight_num, zones in by_flight.items():
                    flight_time = next((a['assigned_flight_time'] for a in allocated if a['assigned_flight'] == flight_num), "")
                    
                    with st.expander(f"✈️ **{flight_num}** - Departure: {flight_time}", expanded=True):
                        zone_cols = st.columns(4)
                        
                        for i, (zone_id, zone_cargo) in enumerate(zones.items()):
                            with zone_cols[i]:
                                zone_weight = sum(c['weight'] for c in zone_cargo)
                                st.markdown(f"**{zone_id}** ({zone_weight:.0f} kg)")
                                
                                if zone_cargo:
                                    for cargo in zone_cargo:
                                        priority_icon = "🔴" if cargo['priority'] >= 4 else "🟡" if cargo['priority'] >= 3 else "🟢"
                                        st.caption(f"{priority_icon} {cargo['request_id']}")
                                        st.caption(f"   {cargo['weight']} kg | {cargo['customer'][:15]}")
                                else:
                                    st.caption("Empty")
            
            # Show not allocated
            if not_allocated:
                st.markdown("#### ❌ Not Allocated (No Capacity)")
                for alloc in not_allocated:
                    st.caption(f"❌ {alloc['request_id']}: {alloc['weight']} kg - {alloc.get('reason', 'No capacity')}")


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
    st.markdown("#### 🌍 KUL Hub Route Performance")
    
    routes_data = pd.DataFrame({
        'Route': ['KUL-SIN', 'KUL-HKG', 'KUL-BKK', 'KUL-MNL', 'KUL-CGK', 'SIN-KUL', 'HKG-KUL'],
        'Direction': ['Outbound', 'Outbound', 'Outbound', 'Outbound', 'Outbound', 'Inbound', 'Inbound'],
        'Daily Flights': [3, 2, 2, 1, 1, 2, 2],
        'Avg Weight (kg)': [4200, 5100, 3800, 3200, 4600, 3800, 4900],
        'Avg Utilization %': [82, 78, 75, 68, 84, 72, 80],
        'Revenue': [185000, 210000, 142000, 98000, 176000, 145000, 195000],
        'On-Time %': [96, 92, 88, 94, 90, 93, 91]
    })
    
    st.dataframe(
        routes_data,
        use_container_width=True,
        column_config={
            'Daily Flights': st.column_config.NumberColumn('Daily Flights', format="%d ✈️"),
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

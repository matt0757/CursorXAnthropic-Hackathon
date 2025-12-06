"""
Flight Database - Manages flight data with available cargo capacity.
Structured for future database implementation (same columns as sample dataset).

NOTE: This does NOT auto-load historical flight data. 
Future flights dataset should be loaded via load_flights_from_csv() method.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
from dataclasses import dataclass, asdict
import glob


@dataclass
class Flight:
    """Flight record matching sample dataset columns."""
    flight_number: str
    flight_date: str  # YYYY-MM-DD format
    origin: str
    destination: str
    tail_number: str
    aircraft_type: str
    gross_weight_cargo_kg: float = 0.0  # Current cargo weight
    gross_volume_cargo_m3: float = 0.0  # Current cargo volume
    passenger_count: int = 0
    baggage_weight_kg: float = 0.0
    fuel_weight_kg: float = 0.0
    fuel_price_per_kg: float = 0.0
    cargo_price_per_kg: float = 0.0
    # Computed fields (not in original dataset, for optimization)
    max_cargo_weight_kg: float = 0.0
    max_cargo_volume_m3: float = 0.0
    available_weight_kg: float = 0.0
    available_volume_m3: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


class FlightDatabase:
    """
    Flight database for managing cargo allocation.
    Designed to be easily replaced with actual database (SQLAlchemy, etc.)
    
    Usage:
        db = FlightDatabase()
        
        # Option 1: Load from CSV file (future flights dataset)
        db.load_flights_from_csv("path/to/future_flights.csv")
        
        # Option 2: Add flights manually
        db.add_flight(Flight(...))
    """
    
    # Expected columns for flight data (matches sample dataset)
    FLIGHT_COLUMNS = [
        'flight_number', 'flight_date', 'origin', 'destination',
        'tail_number', 'aircraft_type', 'gross_weight_cargo_kg',
        'gross_volume_cargo_m3', 'passenger_count', 'baggage_weight_kg',
        'fuel_weight_kg', 'fuel_price_per_kg', 'cargo_price_per_kg',
        'max_cargo_weight_kg', 'max_cargo_volume_m3',
        'available_weight_kg', 'available_volume_m3'
    ]
    
    def __init__(self, data_dir: str = None, auto_load: bool = False):
        """
        Initialize flight database.
        
        Args:
            data_dir: Directory containing data files
            auto_load: If True, loads sample data (for testing). Default False.
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data"
        else:
            data_dir = Path(data_dir)
        
        self.data_dir = data_dir
        self.flights_df: pd.DataFrame = None
        self.aircraft_capacities: Dict[str, Dict] = {}
        
        # Load aircraft capacities (these are static aircraft specs)
        self._load_aircraft_capacities()
        
        # Initialize empty flights DataFrame
        self._init_empty_flights()
        
        # Only auto-load if explicitly requested (for testing)
        if auto_load:
            self._load_sample_flights()
    
    def _init_empty_flights(self):
        """Initialize empty flights DataFrame with correct columns."""
        self.flights_df = pd.DataFrame(columns=self.FLIGHT_COLUMNS)
    
    def _load_aircraft_capacities(self):
        """Load aircraft capacity data from aircraft tail CSV."""
        capacity_files = glob.glob(str(self.data_dir / "*aircraft tail*.csv"))
        
        if capacity_files:
            df = pd.read_csv(capacity_files[0])
            df.columns = df.columns.str.strip()
            
            for _, row in df.iterrows():
                tail = row['Tail Number'].strip()
                self.aircraft_capacities[tail] = {
                    'max_weight_kg': row['Max Weight (Lower Deck) (kg)'],
                    'max_volume_m3': row['Max Volume (Lower Deck) (m³)'],
                    'aircraft_type': row['Aircraft Type']
                }
    
    def _load_sample_flights(self):
        """Load sample flight data (for testing only)."""
        flight_files = glob.glob(str(self.data_dir / "*sample_2024*.csv"))
        
        if not flight_files:
            flight_files = glob.glob(str(self.data_dir / "*sample*.csv"))
        
        if flight_files:
            self.load_flights_from_csv(flight_files[0])
    
    def load_flights_from_csv(self, filepath: str):
        """
        Load flights from a CSV file.
        
        Expected columns (matching sample dataset):
        - flight_number, flight_date, origin, destination
        - tail_number, aircraft_type
        - gross_weight_cargo_kg, gross_volume_cargo_m3
        - passenger_count, baggage_weight_kg
        - fuel_weight_kg, fuel_price_per_kg, cargo_price_per_kg
        
        Args:
            filepath: Path to CSV file with flight data
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Flight data file not found: {filepath}")
        
        self.flights_df = pd.read_csv(filepath)
        self.flights_df.columns = self.flights_df.columns.str.strip()
        
        # Ensure date is properly formatted
        self.flights_df['flight_date'] = pd.to_datetime(
            self.flights_df['flight_date']
        ).dt.strftime('%Y-%m-%d')
        
        # Initialize cargo columns if not present (for future flights with no cargo yet)
        if 'gross_weight_cargo_kg' not in self.flights_df.columns:
            self.flights_df['gross_weight_cargo_kg'] = 0.0
        if 'gross_volume_cargo_m3' not in self.flights_df.columns:
            self.flights_df['gross_volume_cargo_m3'] = 0.0
        
        # Calculate available capacity for each flight
        self._calculate_available_capacity()
        
        return len(self.flights_df)
    
    def add_flight(self, flight: Flight) -> bool:
        """
        Add a single flight to the database.
        
        Args:
            flight: Flight dataclass instance
            
        Returns:
            True if added successfully
        """
        flight_dict = flight.to_dict()
        
        # Calculate max capacity from aircraft
        tail = flight.tail_number
        if tail in self.aircraft_capacities:
            flight_dict['max_cargo_weight_kg'] = self.aircraft_capacities[tail]['max_weight_kg']
            flight_dict['max_cargo_volume_m3'] = self.aircraft_capacities[tail]['max_volume_m3']
        else:
            # Default conservative estimate
            flight_dict['max_cargo_weight_kg'] = 2000
            flight_dict['max_cargo_volume_m3'] = 50
        
        # Calculate available capacity
        flight_dict['available_weight_kg'] = max(0, 
            flight_dict['max_cargo_weight_kg'] - flight_dict['gross_weight_cargo_kg'])
        flight_dict['available_volume_m3'] = max(0,
            flight_dict['max_cargo_volume_m3'] - flight_dict['gross_volume_cargo_m3'])
        
        # Add to DataFrame
        self.flights_df = pd.concat([
            self.flights_df, 
            pd.DataFrame([flight_dict])
        ], ignore_index=True)
        
        return True
    
    def add_flights_batch(self, flights: List[Dict]) -> int:
        """
        Add multiple flights at once.
        
        Args:
            flights: List of flight dictionaries
            
        Returns:
            Number of flights added
        """
        for flight_dict in flights:
            flight = Flight(**{k: v for k, v in flight_dict.items() if k in Flight.__dataclass_fields__})
            self.add_flight(flight)
        
        return len(flights)
    
    def _calculate_available_capacity(self):
        """Calculate available cargo capacity for each flight."""
        if len(self.flights_df) == 0:
            return
        
        max_weights = []
        max_volumes = []
        available_weights = []
        available_volumes = []
        
        for _, row in self.flights_df.iterrows():
            tail = str(row.get('tail_number', '')).strip() if pd.notna(row.get('tail_number')) else ''
            
            if tail in self.aircraft_capacities:
                max_weight = self.aircraft_capacities[tail]['max_weight_kg']
                max_volume = self.aircraft_capacities[tail]['max_volume_m3']
            else:
                # Default conservative estimate
                max_weight = 2000
                max_volume = 50
            
            # Current cargo on flight (default to 0 for future flights)
            current_weight = row.get('gross_weight_cargo_kg', 0) or 0
            current_volume = row.get('gross_volume_cargo_m3', 0) or 0
            
            # Available = Max - Current
            available_weight = max(0, max_weight - current_weight)
            available_volume = max(0, max_volume - current_volume)
            
            max_weights.append(max_weight)
            max_volumes.append(max_volume)
            available_weights.append(available_weight)
            available_volumes.append(available_volume)
        
        self.flights_df['max_cargo_weight_kg'] = max_weights
        self.flights_df['max_cargo_volume_m3'] = max_volumes
        self.flights_df['available_weight_kg'] = available_weights
        self.flights_df['available_volume_m3'] = available_volumes
    
    def get_flight_count(self) -> int:
        """Get total number of flights in database."""
        return len(self.flights_df)
    
    def is_empty(self) -> bool:
        """Check if database has no flights."""
        return len(self.flights_df) == 0
    
    def get_available_flights(
        self,
        origin: Optional[str] = None,
        destination: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        min_weight_available: float = 0,
        min_volume_available: float = 0
    ) -> List[Dict]:
        """
        Get available flights with capacity, sorted by date (earliest first).
        
        Args:
            origin: Filter by origin airport
            destination: Filter by destination airport
            from_date: Filter flights from this date (YYYY-MM-DD)
            to_date: Filter flights until this date (YYYY-MM-DD)
            min_weight_available: Minimum available weight required
            min_volume_available: Minimum available volume required
            
        Returns:
            List of flight dictionaries sorted by date
        """
        if self.is_empty():
            return []
        
        df = self.flights_df.copy()
        
        # Apply filters
        if origin:
            df = df[df['origin'].str.upper() == origin.upper()]
        if destination:
            df = df[df['destination'].str.upper() == destination.upper()]
        if from_date:
            df = df[df['flight_date'] >= from_date]
        if to_date:
            df = df[df['flight_date'] <= to_date]
        
        # Filter by available capacity
        df = df[df['available_weight_kg'] >= min_weight_available]
        df = df[df['available_volume_m3'] >= min_volume_available]
        
        # Sort by date (earliest first)
        df = df.sort_values('flight_date')
        
        return df.to_dict('records')
    
    def get_flight(self, flight_number: str, flight_date: str) -> Optional[Dict]:
        """Get a specific flight by number and date."""
        if self.is_empty():
            return None
        
        mask = (
            (self.flights_df['flight_number'] == flight_number) &
            (self.flights_df['flight_date'] == flight_date)
        )
        
        rows = self.flights_df[mask]
        if len(rows) > 0:
            return rows.iloc[0].to_dict()
        return None
    
    def update_flight_cargo(
        self,
        flight_number: str,
        flight_date: str,
        cargo_weight_to_add: float,
        cargo_volume_to_add: float
    ) -> Tuple[bool, str]:
        """
        Update flight's cargo (allocate cargo to flight).
        Updates gross_weight_cargo_kg and gross_volume_cargo_m3.
        
        Args:
            flight_number: Flight number
            flight_date: Flight date (YYYY-MM-DD)
            cargo_weight_to_add: Weight to add (kg)
            cargo_volume_to_add: Volume to add (m³)
            
        Returns:
            Tuple of (success, message)
        """
        if self.is_empty():
            return False, "No flights in database"
        
        mask = (
            (self.flights_df['flight_number'] == flight_number) &
            (self.flights_df['flight_date'] == flight_date)
        )
        
        idx = self.flights_df[mask].index
        
        if len(idx) == 0:
            return False, f"Flight {flight_number} on {flight_date} not found"
        
        idx = idx[0]
        
        # Check if there's enough capacity
        available_weight = self.flights_df.at[idx, 'available_weight_kg']
        available_volume = self.flights_df.at[idx, 'available_volume_m3']
        
        if cargo_weight_to_add > available_weight:
            return False, f"Insufficient weight capacity. Available: {available_weight:.1f}kg, Requested: {cargo_weight_to_add:.1f}kg"
        
        if cargo_volume_to_add > available_volume:
            return False, f"Insufficient volume capacity. Available: {available_volume:.1f}m³, Requested: {cargo_volume_to_add:.1f}m³"
        
        # Update cargo (this updates gross_weight_cargo_kg in the database)
        self.flights_df.at[idx, 'gross_weight_cargo_kg'] += cargo_weight_to_add
        self.flights_df.at[idx, 'gross_volume_cargo_m3'] += cargo_volume_to_add
        self.flights_df.at[idx, 'available_weight_kg'] -= cargo_weight_to_add
        self.flights_df.at[idx, 'available_volume_m3'] -= cargo_volume_to_add
        
        return True, f"Successfully allocated {cargo_weight_to_add:.1f}kg / {cargo_volume_to_add:.1f}m³ to {flight_number}"
    
    def get_flight_utilization(self, flight_number: str, flight_date: str) -> Optional[Dict]:
        """Get utilization statistics for a flight."""
        flight = self.get_flight(flight_number, flight_date)
        
        if not flight:
            return None
        
        max_weight = flight['max_cargo_weight_kg']
        max_volume = flight['max_cargo_volume_m3']
        current_weight = flight['gross_weight_cargo_kg']
        current_volume = flight['gross_volume_cargo_m3']
        
        weight_utilization = (current_weight / max_weight * 100) if max_weight > 0 else 0
        volume_utilization = (current_volume / max_volume * 100) if max_volume > 0 else 0
        
        return {
            'flight_number': flight_number,
            'flight_date': flight_date,
            'weight_utilization_pct': round(weight_utilization, 1),
            'volume_utilization_pct': round(volume_utilization, 1),
            'current_weight_kg': current_weight,
            'max_weight_kg': max_weight,
            'available_weight_kg': flight['available_weight_kg'],
            'current_volume_m3': current_volume,
            'max_volume_m3': max_volume,
            'available_volume_m3': flight['available_volume_m3'],
            'is_near_full': weight_utilization >= 85 or volume_utilization >= 85
        }
    
    def get_routes(self) -> List[Dict]:
        """Get unique routes (origin-destination pairs)."""
        if self.is_empty():
            return []
        routes = self.flights_df[['origin', 'destination']].drop_duplicates()
        return routes.to_dict('records')
    
    def save_to_csv(self, filepath: str = None):
        """Save current state to CSV (for persistence)."""
        if filepath is None:
            filepath = self.data_dir / "flights_updated.csv"
        
        # Save only original columns (database-compatible)
        db_columns = [
            'flight_number', 'flight_date', 'origin', 'destination',
            'tail_number', 'aircraft_type', 'gross_weight_cargo_kg',
            'gross_volume_cargo_m3', 'passenger_count', 'baggage_weight_kg',
            'fuel_weight_kg', 'fuel_price_per_kg', 'cargo_price_per_kg'
        ]
        
        # Only include columns that exist
        cols_to_save = [c for c in db_columns if c in self.flights_df.columns]
        
        self.flights_df[cols_to_save].to_csv(filepath, index=False)
        return str(filepath)
    
    def clear_all_flights(self):
        """Remove all flights from database."""
        self._init_empty_flights()
    
    def reset_allocations(self):
        """Reset cargo allocations to zero for all flights."""
        if not self.is_empty():
            self.flights_df['gross_weight_cargo_kg'] = 0.0
            self.flights_df['gross_volume_cargo_m3'] = 0.0
            self._calculate_available_capacity()

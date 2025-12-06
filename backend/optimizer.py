"""
Cargo Allocation Optimizer - Efficient cargo slot allocation and revenue optimization.
"""
from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass

@dataclass
class CargoRequest:
    """Cargo booking request."""
    request_id: str
    weight: float  # kg
    volume: float  # m³
    priority: int  # 1-5, higher = more priority
    revenue_per_kg: float
    customer_type: str  # 'premium', 'standard', 'spot'

@dataclass
class AllocationResult:
    """Result of cargo allocation."""
    request_id: str
    allocated: bool
    weight: float
    volume: float
    revenue: float
    slot_ids: List[str]

class CargoOptimizer:
    """Optimizer for efficient cargo allocation."""
    
    def __init__(self):
        """Initialize optimizer."""
        self.CARGO_DENSITY = 200  # kg/m³ average
        
    def optimize_allocation(
        self,
        available_weight: float,
        available_volume: float,
        cargo_requests: List[CargoRequest],
        strategy: str = 'revenue_max'
    ) -> Tuple[List[AllocationResult], Dict]:
        """
        Optimize cargo allocation using various strategies.
        
        Args:
            available_weight: Available weight capacity (kg)
            available_volume: Available volume capacity (m³)
            cargo_requests: List of cargo booking requests
            strategy: Optimization strategy
                - 'revenue_max': Maximize total revenue
                - 'utilization_max': Maximize capacity utilization
                - 'priority_first': Prioritize by customer priority
                - 'balanced': Balance revenue, utilization, and priority
                
        Returns:
            Tuple of (allocation results, optimization stats)
        """
        if strategy == 'revenue_max':
            return self._optimize_revenue_max(available_weight, available_volume, cargo_requests)
        elif strategy == 'utilization_max':
            return self._optimize_utilization_max(available_weight, available_volume, cargo_requests)
        elif strategy == 'priority_first':
            return self._optimize_priority_first(available_weight, available_volume, cargo_requests)
        elif strategy == 'balanced':
            return self._optimize_balanced(available_weight, available_volume, cargo_requests)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _can_fit(self, weight: float, volume: float, 
                 remaining_weight: float, remaining_volume: float) -> bool:
        """Check if cargo can fit in remaining capacity."""
        return weight <= remaining_weight and volume <= remaining_volume
    
    def _optimize_revenue_max(
        self,
        available_weight: float,
        available_volume: float,
        cargo_requests: List[CargoRequest]
    ) -> Tuple[List[AllocationResult], Dict]:
        """
        Maximize revenue using greedy knapsack approach.
        
        This is a dual-constraint knapsack problem (weight AND volume).
        We use a greedy approach: sort by revenue per unit of binding resource.
        """
        # Calculate revenue per unit of binding resource
        def revenue_score(req: CargoRequest) -> float:
            # Determine binding constraint for each request
            weight_ratio = req.weight / available_weight if available_weight > 0 else float('inf')
            volume_ratio = req.volume / available_volume if available_volume > 0 else float('inf')
            
            # Use the tighter constraint
            binding_ratio = max(weight_ratio, volume_ratio)
            
            # Revenue per unit of binding resource
            total_revenue = req.weight * req.revenue_per_kg
            return total_revenue / binding_ratio if binding_ratio > 0 else 0
        
        # Sort by revenue score (descending)
        sorted_requests = sorted(cargo_requests, key=revenue_score, reverse=True)
        
        # Greedy allocation
        allocations = []
        remaining_weight = available_weight
        remaining_volume = available_volume
        total_revenue = 0
        
        for req in sorted_requests:
            if self._can_fit(req.weight, req.volume, remaining_weight, remaining_volume):
                # Allocate
                revenue = req.weight * req.revenue_per_kg
                allocations.append(AllocationResult(
                    request_id=req.request_id,
                    allocated=True,
                    weight=req.weight,
                    volume=req.volume,
                    revenue=revenue,
                    slot_ids=[f"slot_{req.request_id}"]
                ))
                
                remaining_weight -= req.weight
                remaining_volume -= req.volume
                total_revenue += revenue
            else:
                # Cannot fit
                allocations.append(AllocationResult(
                    request_id=req.request_id,
                    allocated=False,
                    weight=0,
                    volume=0,
                    revenue=0,
                    slot_ids=[]
                ))
        
        # Statistics
        weight_utilization = (available_weight - remaining_weight) / available_weight if available_weight > 0 else 0
        volume_utilization = (available_volume - remaining_volume) / available_volume if available_volume > 0 else 0
        
        stats = {
            'total_revenue': total_revenue,
            'weight_utilization': weight_utilization,
            'volume_utilization': volume_utilization,
            'allocated_count': sum(1 for a in allocations if a.allocated),
            'rejected_count': sum(1 for a in allocations if not a.allocated),
            'remaining_weight': remaining_weight,
            'remaining_volume': remaining_volume,
            'strategy': 'revenue_max'
        }
        
        return allocations, stats
    
    def _optimize_utilization_max(
        self,
        available_weight: float,
        available_volume: float,
        cargo_requests: List[CargoRequest]
    ) -> Tuple[List[AllocationResult], Dict]:
        """
        Maximize capacity utilization using bin packing approach.
        """
        # Sort by size (descending) - largest first
        def size_score(req: CargoRequest) -> float:
            return max(
                req.weight / available_weight if available_weight > 0 else 0,
                req.volume / available_volume if available_volume > 0 else 0
            )
        
        sorted_requests = sorted(cargo_requests, key=size_score, reverse=True)
        
        # Greedy allocation
        allocations = []
        remaining_weight = available_weight
        remaining_volume = available_volume
        total_revenue = 0
        
        for req in sorted_requests:
            if self._can_fit(req.weight, req.volume, remaining_weight, remaining_volume):
                revenue = req.weight * req.revenue_per_kg
                allocations.append(AllocationResult(
                    request_id=req.request_id,
                    allocated=True,
                    weight=req.weight,
                    volume=req.volume,
                    revenue=revenue,
                    slot_ids=[f"slot_{req.request_id}"]
                ))
                
                remaining_weight -= req.weight
                remaining_volume -= req.volume
                total_revenue += revenue
            else:
                allocations.append(AllocationResult(
                    request_id=req.request_id,
                    allocated=False,
                    weight=0,
                    volume=0,
                    revenue=0,
                    slot_ids=[]
                ))
        
        weight_utilization = (available_weight - remaining_weight) / available_weight if available_weight > 0 else 0
        volume_utilization = (available_volume - remaining_volume) / available_volume if available_volume > 0 else 0
        
        stats = {
            'total_revenue': total_revenue,
            'weight_utilization': weight_utilization,
            'volume_utilization': volume_utilization,
            'allocated_count': sum(1 for a in allocations if a.allocated),
            'rejected_count': sum(1 for a in allocations if not a.allocated),
            'remaining_weight': remaining_weight,
            'remaining_volume': remaining_volume,
            'strategy': 'utilization_max'
        }
        
        return allocations, stats
    
    def _optimize_priority_first(
        self,
        available_weight: float,
        available_volume: float,
        cargo_requests: List[CargoRequest]
    ) -> Tuple[List[AllocationResult], Dict]:
        """
        Prioritize by customer priority, then revenue.
        """
        # Sort by priority (descending), then revenue (descending)
        sorted_requests = sorted(
            cargo_requests,
            key=lambda r: (r.priority, r.weight * r.revenue_per_kg),
            reverse=True
        )
        
        # Greedy allocation
        allocations = []
        remaining_weight = available_weight
        remaining_volume = available_volume
        total_revenue = 0
        
        for req in sorted_requests:
            if self._can_fit(req.weight, req.volume, remaining_weight, remaining_volume):
                revenue = req.weight * req.revenue_per_kg
                allocations.append(AllocationResult(
                    request_id=req.request_id,
                    allocated=True,
                    weight=req.weight,
                    volume=req.volume,
                    revenue=revenue,
                    slot_ids=[f"slot_{req.request_id}"]
                ))
                
                remaining_weight -= req.weight
                remaining_volume -= req.volume
                total_revenue += revenue
            else:
                allocations.append(AllocationResult(
                    request_id=req.request_id,
                    allocated=False,
                    weight=0,
                    volume=0,
                    revenue=0,
                    slot_ids=[]
                ))
        
        weight_utilization = (available_weight - remaining_weight) / available_weight if available_weight > 0 else 0
        volume_utilization = (available_volume - remaining_volume) / available_volume if available_volume > 0 else 0
        
        stats = {
            'total_revenue': total_revenue,
            'weight_utilization': weight_utilization,
            'volume_utilization': volume_utilization,
            'allocated_count': sum(1 for a in allocations if a.allocated),
            'rejected_count': sum(1 for a in allocations if not a.allocated),
            'remaining_weight': remaining_weight,
            'remaining_volume': remaining_volume,
            'strategy': 'priority_first'
        }
        
        return allocations, stats
    
    def _optimize_balanced(
        self,
        available_weight: float,
        available_volume: float,
        cargo_requests: List[CargoRequest]
    ) -> Tuple[List[AllocationResult], Dict]:
        """
        Balanced approach: combine revenue, priority, and utilization.
        
        Score = (revenue_weight * revenue_score) + 
                (priority_weight * priority_score) +
                (utilization_weight * utilization_score)
        """
        # Weights for multi-objective optimization
        REVENUE_WEIGHT = 0.5
        PRIORITY_WEIGHT = 0.3
        UTILIZATION_WEIGHT = 0.2
        
        # Normalize scores
        max_revenue = max((r.weight * r.revenue_per_kg for r in cargo_requests), default=1)
        max_priority = max((r.priority for r in cargo_requests), default=5)
        
        def balanced_score(req: CargoRequest) -> float:
            # Revenue score (normalized)
            revenue = req.weight * req.revenue_per_kg
            revenue_score = revenue / max_revenue if max_revenue > 0 else 0
            
            # Priority score (normalized)
            priority_score = req.priority / max_priority if max_priority > 0 else 0
            
            # Utilization score (how well it uses capacity)
            weight_ratio = req.weight / available_weight if available_weight > 0 else 0
            volume_ratio = req.volume / available_volume if available_volume > 0 else 0
            utilization_score = (weight_ratio + volume_ratio) / 2
            
            # Combined score
            return (REVENUE_WEIGHT * revenue_score +
                    PRIORITY_WEIGHT * priority_score +
                    UTILIZATION_WEIGHT * utilization_score)
        
        # Sort by balanced score (descending)
        sorted_requests = sorted(cargo_requests, key=balanced_score, reverse=True)
        
        # Greedy allocation
        allocations = []
        remaining_weight = available_weight
        remaining_volume = available_volume
        total_revenue = 0
        
        for req in sorted_requests:
            if self._can_fit(req.weight, req.volume, remaining_weight, remaining_volume):
                revenue = req.weight * req.revenue_per_kg
                allocations.append(AllocationResult(
                    request_id=req.request_id,
                    allocated=True,
                    weight=req.weight,
                    volume=req.volume,
                    revenue=revenue,
                    slot_ids=[f"slot_{req.request_id}"]
                ))
                
                remaining_weight -= req.weight
                remaining_volume -= req.volume
                total_revenue += revenue
            else:
                allocations.append(AllocationResult(
                    request_id=req.request_id,
                    allocated=False,
                    weight=0,
                    volume=0,
                    revenue=0,
                    slot_ids=[]
                ))
        
        weight_utilization = (available_weight - remaining_weight) / available_weight if available_weight > 0 else 0
        volume_utilization = (available_volume - remaining_volume) / available_volume if available_volume > 0 else 0
        
        stats = {
            'total_revenue': total_revenue,
            'weight_utilization': weight_utilization,
            'volume_utilization': volume_utilization,
            'allocated_count': sum(1 for a in allocations if a.allocated),
            'rejected_count': sum(1 for a in allocations if not a.allocated),
            'remaining_weight': remaining_weight,
            'remaining_volume': remaining_volume,
            'strategy': 'balanced'
        }
        
        return allocations, stats
    
    def suggest_pricing(
        self,
        available_capacity: float,
        predicted_demand: float,
        confidence: float,
        base_price: float = 2.0
    ) -> Dict:
        """
        Suggest dynamic pricing based on supply/demand.
        
        Args:
            available_capacity: Available cargo capacity (kg)
            predicted_demand: Predicted cargo demand (kg)
            confidence: Prediction confidence (0-1)
            base_price: Base price per kg
            
        Returns:
            Dictionary with pricing suggestions
        """
        # Demand-to-supply ratio
        demand_ratio = predicted_demand / available_capacity if available_capacity > 0 else 0
        
        # Price multiplier based on demand
        if demand_ratio < 0.5:
            # Low demand - discount to fill capacity
            price_multiplier = 0.8
            pricing_strategy = "discount"
        elif demand_ratio < 0.8:
            # Moderate demand - normal pricing
            price_multiplier = 1.0
            pricing_strategy = "normal"
        elif demand_ratio < 1.2:
            # High demand - premium pricing
            price_multiplier = 1.3
            pricing_strategy = "premium"
        else:
            # Very high demand - surge pricing
            price_multiplier = 1.5
            pricing_strategy = "surge"
        
        # Adjust for confidence
        confidence_multiplier = 0.9 + (confidence * 0.2)  # 0.9 to 1.1
        
        # Final price
        suggested_price = base_price * price_multiplier * confidence_multiplier
        
        return {
            'suggested_price_per_kg': round(suggested_price, 2),
            'base_price': base_price,
            'price_multiplier': round(price_multiplier, 2),
            'confidence_multiplier': round(confidence_multiplier, 2),
            'pricing_strategy': pricing_strategy,
            'demand_ratio': round(demand_ratio, 2),
            'demand_supply_balance': 'oversupply' if demand_ratio < 0.8 else 'balanced' if demand_ratio < 1.2 else 'shortage'
        }

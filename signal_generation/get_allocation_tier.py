from typing import  Dict, Callable

def get_allocation_tier_1(s_score: float, prev_allocation: float, prev_s_score: float,
                          is_decreasing_trend: bool) -> float:
    """Standard mean-reversion strategy with stop-loss.

    Args:
        s_score (float): Current s-score.
        prev_allocation (float): Previous allocation percentage.
        prev_s_score (float): Previous s-score.
        is_decreasing_trend (bool): Whether the s-score is decreasing.

    Returns:
        float: Allocation percentage.
    """
    if s_score > 2.0 or s_score < -2.0:  # Stop-loss
        return 0.0
    if s_score > 1.25:
        if prev_allocation == 0:
            return 1.0  # Enter position
        if s_score > prev_s_score and prev_allocation > 0:
            return min(prev_allocation + 0.2, 1.5)  # Increase allocation
        if s_score < prev_s_score and prev_allocation > 0:
            return max(prev_allocation - 0.2, 0.0)  # Decrease allocation
    elif prev_allocation > 0 and s_score < -1.0:
        return 0.0  # Exit position
    return prev_allocation

def get_allocation_tier_2(s_score: float, prev_allocation: float, prev_s_score: float,
                          is_decreasing_trend: bool) -> float:
    """Aggressive mean-reversion strategy with tighter stop-loss.

    Args:
        s_score (float): Current s-score.
        prev_allocation (float): Previous allocation percentage.
        prev_s_score (float): Previous s-score.
        is_decreasing_trend (bool): Whether the s-score is decreasing.
        peak_s_score (float): Peak s-score since entering the position.

    Returns:
        float: Allocation percentage.
    """
    if s_score > 2.0 or s_score < -1.5:  # Tighter stop-loss
        return 0.0
    if s_score > 1.5:
        return 1.2
    if s_score > 1.2:
        return 1.0
    if s_score > 1.0:
        return 0.8
    if s_score > 0.75:
        return 0.6
    if prev_allocation > 0 and s_score < 0.5:
        return max(0.0, prev_allocation - 0.2)
    elif prev_allocation > 0 and s_score < -1.25:
        return 0.0  # Exit position
    return prev_allocation

def get_allocation_tier_3(s_score: float, prev_allocation: float, prev_s_score: float,
                          is_decreasing_trend: bool) -> float:
    """Trend-following strategy with stop-loss.

    Args:
        s_score (float): Current s-score.
        prev_allocation (float): Previous allocation percentage.
        prev_s_score (float): Previous s-score.
        is_decreasing_trend (bool): Whether the s-score is decreasing.

    Returns:
        float: Allocation percentage.
    """
    if (s_score > prev_s_score and s_score > 0.5):
        return 1.0
    if prev_allocation > 0:
        if s_score > 0.5 and s_score < prev_s_score:
            return max(0.0,prev_allocation-0.2) 
        if s_score > 0 and s_score < prev_s_score:
            return min(0.5, prev_allocation - 0.2)
        elif s_score < -0.5:
            return 0 # Stop-loss on trend reversal
    return prev_allocation


def get_allocation_tier_4(s_score: float, prev_allocation: float, prev_s_score: float,
                          is_decreasing_trend: bool) -> float:
    """Hybrid mean-reversion and trend-following strategy.

    Args:
        s_score (float): Current s-score.
        prev_allocation (float): Previous allocation percentage.
        prev_s_score (float): Previous s-score.
        is_decreasing_trend (bool): Whether the s-score is decreasing.

    Returns:
        float: Allocation percentage.
    """
    if s_score > 2.5 or s_score < -2.0:  # Stop-loss
        return 0.0
    if s_score > 1.5:
        if s_score > prev_s_score:
            return 1.0
        if s_score < prev_s_score and prev_allocation > 0:
            return max(0.0, prev_allocation - 0.2)
    if s_score > 1.25:
        return 0.8
    if s_score > 1.0:
        return 0.6
    if prev_allocation > 0 and s_score < 1.0:
        return max(0.0, prev_allocation - 0.2)
    elif prev_allocation > 0 and s_score < -1.25:
        return 0.0  # Exit position
    return prev_allocation

allocation_functions: Dict[int, Callable] = {
    1: get_allocation_tier_1,
    2: get_allocation_tier_2,
    3: get_allocation_tier_3,
    4: get_allocation_tier_4,
}

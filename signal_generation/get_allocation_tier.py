from typing import  Dict, Callable

def get_allocation_tier_1(s_score: float, prev_allocation: float, prev_s_score: float,
                          is_decreasing_trend: bool, peak_s_score: float) -> float:
    """Standard mean-reversion strategy with stop-loss.

    Args:
        s_score (float): Current s-score.
        prev_allocation (float): Previous allocation percentage.
        prev_s_score (float): Previous s-score.
        is_decreasing_trend (bool): Whether the s-score is decreasing.
        peak_s_score (float): Peak s-score since entering the position.

    Returns:
        float: Allocation percentage.
    """
    if s_score > 2.5 or s_score < -2.0:  # Stop-loss
        return 0.0
    if s_score > 2.0 or s_score < -1.5:
        return 0.0
    if prev_allocation > 0 and peak_s_score > 0 and s_score < peak_s_score * 0.8:  # Trailing take-profit
        return 0.0
    if s_score > 1.5 and s_score > prev_s_score:
        return 1.2
    if s_score > 1.25 and s_score > prev_s_score:
        return 1.0
    if s_score > 1.0:
        return 0.8
    if prev_allocation > 0:
        if s_score < 1.0 and s_score < prev_s_score:
            return max(0.0, prev_allocation - 0.1)
        if s_score > prev_s_score and is_decreasing_trend:
            return max(0.0, prev_allocation - 0.2)
        return prev_allocation
    return 0.0

def get_allocation_tier_2(s_score: float, prev_allocation: float, prev_s_score: float,
                          is_decreasing_trend: bool, peak_s_score: float) -> float:
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
    if s_score > 2.2 or s_score < -1.8:  # Tighter stop-loss
        return 0.0
    if prev_allocation > 0 and peak_s_score > 0 and s_score < peak_s_score * 0.85:  # Trailing take-profit
        return 0.0
    if s_score > 2.0:
        return 1.5
    if s_score > 1.5:
        return 1.2
    if s_score > 1.0:
        return 1.0
    if prev_allocation > 0 and s_score < 1.0:
        return max(0.0, prev_allocation - 0.2)
    return 0.0

def get_allocation_tier_3(s_score: float, prev_allocation: float, prev_s_score: float,
                          is_decreasing_trend: bool, peak_s_score: float) -> float:
    """Conservative mean-reversion strategy with stop-loss.

    Args:
        s_score (float): Current s-score.
        prev_allocation (float): Previous allocation percentage.
        prev_s_score (float): Previous s-score.
        is_decreasing_trend (bool): Whether the s-score is decreasing.
        peak_s_score (float): Peak s-score since entering the position.

    Returns:
        float: Allocation percentage.
    """
    if s_score > 1.8 or s_score < -1.2:  # Stop-loss
        return 0.0
    if prev_allocation > 0 and peak_s_score > 0 and s_score < peak_s_score * 0.9:  # Trailing take-profit
        return 0.0
    if s_score > 1.0:
        return 0.5
    if s_score > 0.5:
        return 0.3
    if prev_allocation > 0 and s_score < 0.5:
        return max(0.0, prev_allocation - 0.05)
    return 0.0

def get_allocation_tier_4(s_score: float, prev_allocation: float, prev_s_score: float,
                          is_decreasing_trend: bool, peak_s_score: float) -> float:
    """Trend-following strategy with stop-loss.

    Args:
        s_score (float): Current s-score.
        prev_allocation (float): Previous allocation percentage.
        prev_s_score (float): Previous s-score.
        is_decreasing_trend (bool): Whether the s-score is decreasing.
        peak_s_score (float): Peak s-score since entering the position.

    Returns:
        float: Allocation percentage.
    """
    if prev_allocation > 0:
        if (s_score > 0 and s_score < prev_s_score * 0.5) or (s_score < 0 and s_score > prev_s_score * 1.5):  # Stop-loss
            return 0.0
        if peak_s_score > 0 and s_score < peak_s_score * 0.8:  # Trailing take-profit
            return 0.0
    if s_score > prev_s_score and s_score > 0:
        return 0.8
    if s_score < prev_s_score and s_score < 0:
        return 0.8
    return 0.0

def get_allocation_tier_5(s_score: float, prev_allocation: float, prev_s_score: float,
                          is_decreasing_trend: bool, peak_s_score: float) -> float:
    """Momentum-based strategy with stop-loss.

    Args:
        s_score (float): Current s-score.
        prev_allocation (float): Previous allocation percentage.
        prev_s_score (float): Previous s-score.
        is_decreasing_trend (bool): Whether the s-score is decreasing.
        peak_s_score (float): Peak s-score since entering the position.

    Returns:
        float: Allocation percentage.
    """
    if prev_allocation > 0:
        if (s_score > 0 and s_score < 0) or (s_score < 0 and s_score > 0):  # Stop-loss on momentum reversal
            return 0.0
        if peak_s_score > 0 and s_score < peak_s_score * 0.85:  # Trailing take-profit
            return 0.0
    if (s_score > 0 and prev_s_score > 0) or (s_score < 0 and prev_s_score < 0):
        return 0.8
    return 0.0

def get_allocation_tier_6(s_score: float, prev_allocation: float, prev_s_score: float,
                          is_decreasing_trend: bool, peak_s_score: float, sigma: float = 1.0) -> float:
    """Volatility-adjusted mean-reversion strategy.

    Args:
        s_score (float): Current s-score.
        prev_allocation (float): Previous allocation percentage.
        prev_s_score (float): Previous s-score.
        is_decreasing_trend (bool): Whether the s-score is decreasing.
        peak_s_score (float): Peak s-score since entering the position.
        sigma (float, optional): Volatility estimate from OU process. Defaults to 1.0.

    Returns:
        float: Allocation percentage.
    """
    normalized_s_score = s_score / sigma if sigma > 0 else s_score
    if normalized_s_score > 2.0 or normalized_s_score < -1.5:  # Stop-loss
        return 0.0
    if prev_allocation > 0 and peak_s_score > 0 and s_score < peak_s_score * 0.9:  # Trailing take-profit
        return 0.0
    if normalized_s_score > 1.5:
        return 1.0
    if normalized_s_score > 1.0:
        return 0.7
    if prev_allocation > 0 and normalized_s_score < 1.0:
        return max(0.0, prev_allocation - 0.1)
    return 0.0

def get_allocation_tier_7(s_score: float, prev_allocation: float, prev_s_score: float,
                          is_decreasing_trend: bool, peak_s_score: float) -> float:
    """Hybrid mean-reversion and trend-following strategy.

    Args:
        s_score (float): Current s-score.
        prev_allocation (float): Previous allocation percentage.
        prev_s_score (float): Previous s-score.
        is_decreasing_trend (bool): Whether the s-score is decreasing.
        peak_s_score (float): Peak s-score since entering the position.

    Returns:
        float: Allocation percentage.
    """
    if s_score > 2.5 or s_score < -2.0:  # Stop-loss
        return 0.0
    if prev_allocation > 0 and peak_s_score > 0 and s_score < peak_s_score * 0.8:  # Trailing take-profit
        return 0.0
    if abs(s_score) > 1.5 and ((s_score > 0 and s_score > prev_s_score) or (s_score < 0 and s_score < prev_s_score)):
        return 0.9  # Trend-following
    if 0.5 < abs(s_score) < 1.5:
        return 0.6  # Mean-reversion
    if prev_allocation > 0 and abs(s_score) < 0.5:
        return max(0.0, prev_allocation - 0.1)
    return 0.0

allocation_functions: Dict[int, Callable] = {
    1: get_allocation_tier_1,
    2: get_allocation_tier_2,
    3: get_allocation_tier_3,
    4: get_allocation_tier_4,
    5: get_allocation_tier_5,
    6: get_allocation_tier_6,
    7: get_allocation_tier_7,
}

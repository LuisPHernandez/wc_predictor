def points_for_prediction(pred_home, pred_away, actual_home, actual_away):
    """
    Returns points (0-3) earned for a single match prediction
    under the pool's scoring rules.

    Parameters
    ----------
    pred_home   : int  — predicted goals for home/team1
    pred_away   : int  — predicted goals for away/team2
    actual_home : int  — actual goals for home/team1
    actual_away : int  — actual goals for away/team2
    """

    def outcome(h, a):
        if h > a:   return 'home'
        elif a > h: return 'away'
        else:       return 'draw'

    pred_outcome   = outcome(pred_home, pred_away)
    actual_outcome = outcome(actual_home, actual_away)

    # Wrong winner → 0 points
    if pred_outcome != actual_outcome:
        return 0

    # Exact scoreline → 3 points
    if pred_home == actual_home and pred_away == actual_away:
        return 3

    # Draw: correct draw but wrong goals → 1 point
    # (2 pts impossible: there's no single "winner's goals" to match)
    if actual_outcome == 'draw':
        return 1

    # Win: check if the winner's goal count matches
    if actual_outcome == 'home':
        return 2 if pred_home == actual_home else 1
    else:  # away win
        return 2 if pred_away == actual_away else 1
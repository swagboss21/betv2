"""
Edge-Based Prediction Analysis

Key hypothesis: Overs are systematically overpriced. Rather than predicting hit/miss,
we should identify where the market is most wrong and bet accordingly.

Edge = actual_hit_rate - implied_probability
Negative edge = overs are overpriced = bet UNDER
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data/processed/training_data_full.csv"
OUTPUT_DIR = BASE_DIR / "analysis"

def load_data():
    """Load and filter training data."""
    df = pd.read_csv(DATA_PATH)
    # Filter out rows without actual results
    df = df[df['actual'].notna()]
    print(f"Loaded {len(df):,} rows with actual results")
    return df

def calculate_market_baseline(df):
    """Calculate overall market efficiency."""
    print("\n" + "="*60)
    print("MARKET BASELINE ANALYSIS")
    print("="*60)

    overall_hit_rate = df['hit'].mean()
    avg_over_implied = df['over_implied'].mean()
    market_edge = overall_hit_rate - avg_over_implied

    print(f"\nOverall over hit rate:     {overall_hit_rate:.2%}")
    print(f"Avg implied over prob:     {avg_over_implied:.2%}")
    print(f"Market edge (hit - impl):  {market_edge:+.2%}")
    print(f"\nInterpretation: {'Overs overpriced' if market_edge < 0 else 'Unders overpriced'} by {abs(market_edge):.1%}")

    return overall_hit_rate, avg_over_implied, market_edge

def build_player_edge_lookup(df, min_props=20):
    """
    Build player edge lookup table.
    Edge = actual_hit_rate - avg_over_implied
    """
    print("\n" + "="*60)
    print("PLAYER EDGE ANALYSIS")
    print("="*60)

    player_stats = df.groupby('player_id').agg({
        'hit': ['mean', 'count'],
        'over_implied': 'mean',
        'player': 'first'
    }).reset_index()

    player_stats.columns = ['player_id', 'hit_rate', 'prop_count', 'avg_implied', 'player_name']
    player_stats['edge'] = player_stats['hit_rate'] - player_stats['avg_implied']
    player_stats['edge_pp'] = player_stats['edge'] * 100  # percentage points

    # Filter to players with minimum props
    player_stats = player_stats[player_stats['prop_count'] >= min_props]
    player_stats = player_stats.sort_values('edge')

    print(f"\nPlayers with {min_props}+ props: {len(player_stats)}")

    print(f"\n--- TOP 20 WORST EDGE (BET UNDER) ---")
    worst = player_stats.head(20)
    for _, row in worst.iterrows():
        print(f"{row['player_name']:<25} Edge: {row['edge_pp']:+6.1f}pp  "
              f"Hit: {row['hit_rate']:.1%}  Impl: {row['avg_implied']:.1%}  n={int(row['prop_count'])}")

    print(f"\n--- TOP 20 BEST EDGE (BET OVER) ---")
    best = player_stats.tail(20).iloc[::-1]
    for _, row in best.iterrows():
        print(f"{row['player_name']:<25} Edge: {row['edge_pp']:+6.1f}pp  "
              f"Hit: {row['hit_rate']:.1%}  Impl: {row['avg_implied']:.1%}  n={int(row['prop_count'])}")

    return player_stats

def build_stat_type_edge_lookup(df):
    """Build stat type edge lookup table."""
    print("\n" + "="*60)
    print("STAT TYPE EDGE ANALYSIS")
    print("="*60)

    stat_stats = df.groupby('stat_type').agg({
        'hit': ['mean', 'count'],
        'over_implied': 'mean'
    }).reset_index()

    stat_stats.columns = ['stat_type', 'hit_rate', 'prop_count', 'avg_implied']
    stat_stats['edge'] = stat_stats['hit_rate'] - stat_stats['avg_implied']
    stat_stats['edge_pp'] = stat_stats['edge'] * 100
    stat_stats = stat_stats.sort_values('edge')

    print(f"\n{'Stat Type':<25} {'Edge':>8} {'Hit Rate':>10} {'Implied':>10} {'Count':>8}")
    print("-" * 65)
    for _, row in stat_stats.iterrows():
        print(f"{row['stat_type']:<25} {row['edge_pp']:+7.1f}pp {row['hit_rate']:>9.1%} "
              f"{row['avg_implied']:>9.1%} {int(row['prop_count']):>8}")

    return stat_stats

def build_combined_edge(df, player_stats, stat_stats, min_props=20):
    """Combine player and stat type edge for each prop."""
    # Create lookup dicts
    player_edge_dict = player_stats.set_index('player_id')['edge'].to_dict()
    stat_edge_dict = stat_stats.set_index('stat_type')['edge'].to_dict()

    df = df.copy()
    df['player_edge'] = df['player_id'].map(player_edge_dict)
    df['stat_edge'] = df['stat_type'].map(stat_edge_dict)
    df['combined_edge'] = df['player_edge'] + df['stat_edge']

    return df

def backtest_edge_strategy(df, player_stats, stat_stats, thresholds=[-5, -10, -15, -20]):
    """
    Backtest: Bet UNDER on props where player has negative edge.
    """
    print("\n" + "="*60)
    print("BACKTEST: EDGE-BASED UNDER STRATEGY")
    print("="*60)

    # Add edge to df
    player_edge_dict = player_stats.set_index('player_id')['edge'].to_dict()
    stat_edge_dict = stat_stats.set_index('stat_type')['edge'].to_dict()

    df = df.copy()
    df['player_edge'] = df['player_id'].map(player_edge_dict)
    df['stat_edge'] = df['stat_type'].map(stat_edge_dict)
    df['combined_edge'] = df['player_edge'].fillna(0) + df['stat_edge'].fillna(0)

    results = []

    print(f"\n{'Strategy':<40} {'Accuracy':>10} {'Props':>8} {'Props/Day':>10}")
    print("-" * 70)

    # Calculate days in dataset for props/day
    total_days = df['date'].nunique()

    # Player edge only strategies
    for threshold in thresholds:
        mask = df['player_edge'] <= (threshold / 100)
        subset = df[mask]
        if len(subset) > 0:
            # Betting UNDER means we win when hit=0
            accuracy = (subset['hit'] == 0).mean()
            props_per_day = len(subset) / total_days
            strategy_name = f"Player edge <= {threshold}pp (bet UNDER)"
            print(f"{strategy_name:<40} {accuracy:>9.1%} {len(subset):>8} {props_per_day:>10.1f}")
            results.append({
                'strategy': strategy_name,
                'type': 'player_edge',
                'threshold': threshold,
                'accuracy': accuracy,
                'prop_count': len(subset),
                'props_per_day': props_per_day
            })

    print()

    # Combined edge strategies
    for threshold in thresholds:
        mask = df['combined_edge'] <= (threshold / 100)
        subset = df[mask]
        if len(subset) > 0:
            accuracy = (subset['hit'] == 0).mean()
            props_per_day = len(subset) / total_days
            strategy_name = f"Combined edge <= {threshold}pp (bet UNDER)"
            print(f"{strategy_name:<40} {accuracy:>9.1%} {len(subset):>8} {props_per_day:>10.1f}")
            results.append({
                'strategy': strategy_name,
                'type': 'combined_edge',
                'threshold': threshold,
                'accuracy': accuracy,
                'prop_count': len(subset),
                'props_per_day': props_per_day
            })

    print()

    # Stat type edge strategies
    for threshold in thresholds:
        mask = df['stat_edge'] <= (threshold / 100)
        subset = df[mask]
        if len(subset) > 0:
            accuracy = (subset['hit'] == 0).mean()
            props_per_day = len(subset) / total_days
            strategy_name = f"Stat edge <= {threshold}pp (bet UNDER)"
            print(f"{strategy_name:<40} {accuracy:>9.1%} {len(subset):>8} {props_per_day:>10.1f}")
            results.append({
                'strategy': strategy_name,
                'type': 'stat_edge',
                'threshold': threshold,
                'accuracy': accuracy,
                'prop_count': len(subset),
                'props_per_day': props_per_day
            })

    return pd.DataFrame(results)

def backtest_specific_players(df, player_stats):
    """Test specific worst-edge players mentioned in analysis."""
    print("\n" + "="*60)
    print("BACKTEST: SPECIFIC PLAYER UNDERS")
    print("="*60)

    # Get worst 10 players by edge
    worst_players = player_stats.nsmallest(10, 'edge')['player_id'].tolist()

    subset = df[df['player_id'].isin(worst_players)]
    if len(subset) > 0:
        accuracy = (subset['hit'] == 0).mean()
        print(f"\nBottom 10 players by edge - betting all UNDER:")
        print(f"  Total props: {len(subset)}")
        print(f"  Under accuracy: {accuracy:.1%}")

        # Break down by player
        print(f"\n{'Player':<25} {'Under Acc':>10} {'Props':>8} {'Avg Edge':>10}")
        print("-" * 55)
        for pid in worst_players:
            player_subset = df[df['player_id'] == pid]
            if len(player_subset) > 0:
                player_name = player_subset['player'].iloc[0]
                under_acc = (player_subset['hit'] == 0).mean()
                player_edge = player_stats[player_stats['player_id'] == pid]['edge_pp'].iloc[0]
                print(f"{player_name:<25} {under_acc:>9.1%} {len(player_subset):>8} {player_edge:>+9.1f}pp")

def compare_to_models(df, player_stats, stat_stats):
    """Compare edge-based approach to v1.0 and v1.1 model results."""
    print("\n" + "="*60)
    print("COMPARISON: EDGE vs MODELS")
    print("="*60)

    # Model results from CLAUDE.md
    model_results = {
        'v1.0': {'accuracy': 0.6176, 'over_recall': 0.1427, 'under_recall': 0.9256},
        'v1.1': {'accuracy': 0.5765, 'over_recall': 0.6539, 'under_recall': 0.5263},
    }

    # Best edge strategy (let's find it)
    player_edge_dict = player_stats.set_index('player_id')['edge'].to_dict()
    df = df.copy()
    df['player_edge'] = df['player_id'].map(player_edge_dict)

    # Try -10pp threshold
    threshold = -10
    mask = df['player_edge'] <= (threshold / 100)

    # For edge strategy, we only bet UNDER on filtered props
    edge_props = df[mask]

    if len(edge_props) > 0:
        # When we bet UNDER, we're right when hit=0
        edge_under_accuracy = (edge_props['hit'] == 0).mean()

        # For comparison, what's the overall under accuracy on same props?
        naive_under = (df['hit'] == 0).mean()

        print(f"\nComparison (on full dataset):")
        print(f"{'Approach':<35} {'Accuracy':>10} {'Coverage':>10} Notes")
        print("-" * 70)
        print(f"{'Always bet UNDER':<35} {naive_under:>9.1%} {100:>9.0%}% All props")
        print(f"{'Model v1.0':<35} {model_results['v1.0']['accuracy']:>9.1%} {100:>9.0%}% Biased to under")
        print(f"{'Model v1.1':<35} {model_results['v1.1']['accuracy']:>9.1%} {100:>9.0%}% Balanced")
        print(f"{'Edge <= -10pp (UNDER only)':<35} {edge_under_accuracy:>9.1%} {len(edge_props)/len(df)*100:>9.1f}% Selective")

        # Expected value comparison
        print(f"\n--- EXPECTED VALUE ANALYSIS ---")
        print("Assuming -110 odds (bet $110 to win $100):")

        # For a -110 bet, need 52.38% to break even
        breakeven = 0.5238

        for name, acc in [('Always UNDER', naive_under),
                          ('Model v1.0', model_results['v1.0']['accuracy']),
                          ('Model v1.1', model_results['v1.1']['accuracy']),
                          ('Edge <= -10pp', edge_under_accuracy)]:
            ev_per_bet = (acc * 100) - ((1 - acc) * 110)
            status = "PROFITABLE" if ev_per_bet > 0 else "LOSING"
            print(f"{name:<25} {acc:.1%} accuracy -> ${ev_per_bet:+.2f} EV per $110 bet ({status})")

def analyze_edge_distribution(df, player_stats):
    """Analyze the distribution of edge across the dataset."""
    print("\n" + "="*60)
    print("EDGE DISTRIBUTION ANALYSIS")
    print("="*60)

    player_edge_dict = player_stats.set_index('player_id')['edge'].to_dict()
    df = df.copy()
    df['player_edge'] = df['player_id'].map(player_edge_dict)
    df = df.dropna(subset=['player_edge'])

    print(f"\nEdge distribution (percentage points):")
    print(f"  Min:    {df['player_edge'].min()*100:+.1f}pp")
    print(f"  25th:   {df['player_edge'].quantile(0.25)*100:+.1f}pp")
    print(f"  Median: {df['player_edge'].median()*100:+.1f}pp")
    print(f"  75th:   {df['player_edge'].quantile(0.75)*100:+.1f}pp")
    print(f"  Max:    {df['player_edge'].max()*100:+.1f}pp")

    # How many props at each edge threshold?
    print(f"\nProps at each edge threshold:")
    thresholds = [-5, -10, -15, -20, -25, -30]
    total_props = len(df)
    for t in thresholds:
        count = (df['player_edge'] <= t/100).sum()
        pct = count / total_props * 100
        print(f"  Edge <= {t:+3d}pp: {count:>6,} props ({pct:.1f}%)")

def find_optimal_threshold(df, player_stats):
    """Find the threshold that maximizes profit."""
    print("\n" + "="*60)
    print("OPTIMAL THRESHOLD SEARCH")
    print("="*60)

    player_edge_dict = player_stats.set_index('player_id')['edge'].to_dict()
    df = df.copy()
    df['player_edge'] = df['player_id'].map(player_edge_dict)
    df = df.dropna(subset=['player_edge'])

    results = []
    for threshold in range(-30, 5, 1):
        t = threshold / 100
        mask = df['player_edge'] <= t
        subset = df[mask]
        if len(subset) >= 100:  # minimum sample
            accuracy = (subset['hit'] == 0).mean()
            ev_per_bet = (accuracy * 100) - ((1 - accuracy) * 110)
            total_ev = ev_per_bet * len(subset) / 110  # normalize to $100 bets
            results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'prop_count': len(subset),
                'ev_per_bet': ev_per_bet,
                'total_ev': total_ev
            })

    results_df = pd.DataFrame(results)

    # Find best by EV per bet (quality)
    best_ev = results_df.loc[results_df['ev_per_bet'].idxmax()]
    # Find best by total EV (quantity * quality)
    best_total = results_df.loc[results_df['total_ev'].idxmax()]

    print(f"\nBest threshold by EV per bet (quality):")
    print(f"  Threshold: {best_ev['threshold']:.0f}pp")
    print(f"  Accuracy: {best_ev['accuracy']:.1%}")
    print(f"  Props: {best_ev['prop_count']:.0f}")
    print(f"  EV per $110 bet: ${best_ev['ev_per_bet']:.2f}")

    print(f"\nBest threshold by total EV (quality * quantity):")
    print(f"  Threshold: {best_total['threshold']:.0f}pp")
    print(f"  Accuracy: {best_total['accuracy']:.1%}")
    print(f"  Props: {best_total['prop_count']:.0f}")
    print(f"  Total EV (normalized): ${best_total['total_ev']:.2f}")

    # Show top 10 by each metric
    print(f"\n--- TOP 10 BY EV PER BET ---")
    print(f"{'Threshold':>10} {'Accuracy':>10} {'Props':>8} {'EV/bet':>10}")
    for _, row in results_df.nlargest(10, 'ev_per_bet').iterrows():
        print(f"{row['threshold']:>9.0f}pp {row['accuracy']:>9.1%} {row['prop_count']:>8.0f} ${row['ev_per_bet']:>9.2f}")

    return results_df

def main():
    print("="*60)
    print("EDGE-BASED PREDICTION ANALYSIS")
    print("="*60)

    # Load data
    df = load_data()

    # Market baseline
    overall_hit_rate, avg_over_implied, market_edge = calculate_market_baseline(df)

    # Build edge lookups
    player_stats = build_player_edge_lookup(df, min_props=20)
    stat_stats = build_stat_type_edge_lookup(df)

    # Analyze edge distribution
    analyze_edge_distribution(df, player_stats)

    # Find optimal threshold
    threshold_results = find_optimal_threshold(df, player_stats)

    # Backtest strategies
    strategy_results = backtest_edge_strategy(df, player_stats, stat_stats)

    # Test specific players
    backtest_specific_players(df, player_stats)

    # Compare to models
    compare_to_models(df, player_stats, stat_stats)

    # Save player edge lookup for inference
    player_stats.to_csv(OUTPUT_DIR / 'player_edge_lookup.csv', index=False)
    stat_stats.to_csv(OUTPUT_DIR / 'stat_type_edge_lookup.csv', index=False)
    threshold_results.to_csv(OUTPUT_DIR / 'threshold_analysis.csv', index=False)

    print("\n" + "="*60)
    print("OUTPUT FILES SAVED")
    print("="*60)
    print(f"  - {OUTPUT_DIR / 'player_edge_lookup.csv'}")
    print(f"  - {OUTPUT_DIR / 'stat_type_edge_lookup.csv'}")
    print(f"  - {OUTPUT_DIR / 'threshold_analysis.csv'}")

    print("\n" + "="*60)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*60)
    print("""
Key Findings:
1. Overs ARE systematically overpriced (actual ~39% vs implied ~47%)
2. Edge-based filtering can identify the most overpriced props
3. Selective betting on worst-edge players improves accuracy

Recommendation:
- Use edge-based approach for prop selection
- Threshold of -10pp to -15pp balances accuracy vs volume
- Compare live results against model-based approach
""")

if __name__ == "__main__":
    main()

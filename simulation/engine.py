"""
Monte Carlo Simulation Engine for NBA player props.

Chains game, minutes, and stats models to produce player stat distributions.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from .models import (
    PlayerFeatures, GameContext, StatDistribution,
    PlayerPrediction, PropBet, PropAnalysis, SimulationResult
)
from .feature_transformer import FeatureTransformer
from .injury_adjustment import InjuryAdjustmentModule
from .edge_calculator import EdgeCalculator


class MonteCarloEngine:
    """
    Monte Carlo simulation engine for NBA player props.

    Usage:
        engine = MonteCarloEngine("models/")
        result = engine.simulate_game("LAL", "BOS")
        print(result.players["LeBron James"].pts.mean)
    """

    # Stats we can simulate
    STAT_TYPES = ["pts", "reb", "ast", "stl", "blk", "tov", "fg3m"]

    # Stats that are discrete counts (use Poisson distribution instead of Normal)
    # These are low-count stats where normal distribution is inappropriate
    COUNT_STATS = {"fg3m", "stl", "blk"}

    def __init__(self, models_dir: str = "models/"):
        """
        Load all trained models.

        Args:
            models_dir: Directory containing pkl model files
        """
        self.models_dir = Path(models_dir)

        # Load models
        self._load_models()

        # Initialize components
        self.transformer = FeatureTransformer()
        self.injury_module = InjuryAdjustmentModule()
        self.edge_calc = EdgeCalculator()

        print(f"MonteCarloEngine initialized with models from {models_dir}")

    def _load_models(self):
        """Load all model artifacts from disk."""
        # Game model
        with open(self.models_dir / "game_model.pkl", "rb") as f:
            gm = pickle.load(f)
            self.game_model = gm["model"]
            self.game_variance = gm.get("variance_lookup", {})

        # Minutes model
        with open(self.models_dir / "minutes_model.pkl", "rb") as f:
            mm = pickle.load(f)
            self.minutes_model = mm["model"]
            self.minutes_variance = mm.get("variance_lookup", {})
            self.minutes_tiers = mm.get("tier_definitions", {})

        # Stats models (dict of 7 models)
        with open(self.models_dir / "stats_models.pkl", "rb") as f:
            self.stats_models = pickle.load(f)

        print(f"Loaded models: game, minutes, stats ({list(self.stats_models.keys())})")

    def _get_game_variance(self, prediction: float) -> float:
        """Get std dev for game score prediction."""
        for bucket_label, data in self.game_variance.items():
            if bucket_label == "overall":
                continue
            try:
                if "-" in str(bucket_label):
                    low, high = map(int, str(bucket_label).split("-"))
                    if low <= prediction < high:
                        return data.get("std", 8.0)
            except (ValueError, TypeError):
                continue

        # Fallback
        return self.game_variance.get("overall", {}).get("std", 8.0)

    def _get_minutes_variance(self, tier: str) -> float:
        """Get std dev for minutes prediction based on player tier."""
        tier_data = self.minutes_variance.get(tier, {})
        if isinstance(tier_data, dict):
            return tier_data.get("std", 5.0)
        return 5.0

    def _get_stat_variance(self, stat: str, prediction: float) -> float:
        """Get std dev for stat prediction."""
        if stat not in self.stats_models:
            return 3.0

        variance_lookup = self.stats_models[stat].get("variance_lookup", {})

        for (low, high), std in variance_lookup.items():
            if low <= prediction < high:
                return std

        # Fallback to last bucket
        if variance_lookup:
            return list(variance_lookup.values())[-1]
        return 3.0

    def _determine_tier(self, min_szn_avg: float) -> str:
        """Determine player tier from season minutes average."""
        for tier, (low, high) in self.minutes_tiers.items():
            if low <= min_szn_avg < high:
                return tier
        return "Rotation"

    def simulate_game(
        self,
        home_team: str,
        away_team: str,
        injuries: Dict[str, str] = None,
        n_simulations: int = 10000
    ) -> SimulationResult:
        """
        Run full Monte Carlo simulation for a game.

        Args:
            home_team: Home team abbreviation (e.g., "LAL")
            away_team: Away team abbreviation (e.g., "BOS")
            injuries: Dict of player_name -> status (e.g., {"AD": "OUT"})
            n_simulations: Number of simulations to run

        Returns:
            SimulationResult with distributions for all players
        """
        print(f"Simulating {away_team} @ {home_team} ({n_simulations} sims)...")

        # Get game context
        context = self.transformer.get_game_context(home_team, away_team)

        # Get rosters
        print("Fetching rosters...")
        home_roster = self.transformer.get_roster(home_team)
        away_roster = self.transformer.get_roster(away_team)

        print(f"  Home roster: {len(home_roster)} players")
        print(f"  Away roster: {len(away_roster)} players")

        # Apply injury adjustments
        if injuries:
            print(f"Applying injuries: {injuries}")
            home_roster = self.injury_module.adjust_roster(home_roster, injuries)
            away_roster = self.injury_module.adjust_roster(away_roster, injuries)

        # Run simulations
        print("Running simulations...")
        sim_results = self._run_simulations(
            context, home_roster, away_roster, n_simulations
        )

        # Aggregate results
        print("Aggregating results...")
        return self._aggregate_results(
            sim_results, context, home_roster + away_roster, n_simulations
        )

    def _run_simulations(
        self,
        context: GameContext,
        home_roster: List[PlayerFeatures],
        away_roster: List[PlayerFeatures],
        n_simulations: int
    ) -> Dict:
        """Run the Monte Carlo simulation loop."""
        # Pre-compute game score predictions
        game_features = context.build_game_features().reshape(1, -1)
        game_pred = self.game_model.predict(game_features)[0]

        # Handle multi-output (home_pts, away_pts) or single output
        if hasattr(game_pred, "__len__") and len(game_pred) == 2:
            home_pred, away_pred = game_pred
        else:
            # Single output model - estimate
            home_pred = float(game_pred)
            away_pred = home_pred - 3  # Home advantage ~3 pts

        home_std = self._get_game_variance(home_pred)
        away_std = self._get_game_variance(away_pred)

        # Storage for all simulations
        results = {
            "home_scores": np.zeros(n_simulations),
            "away_scores": np.zeros(n_simulations),
            "players": {}
        }

        # Initialize player storage
        all_players = home_roster + away_roster
        for player in all_players:
            results["players"][player.player_name] = {
                "minutes": np.zeros(n_simulations),
                "pts": np.zeros(n_simulations),
                "reb": np.zeros(n_simulations),
                "ast": np.zeros(n_simulations),
                "stl": np.zeros(n_simulations),
                "blk": np.zeros(n_simulations),
                "tov": np.zeros(n_simulations),
                "fg3m": np.zeros(n_simulations)
            }

        # Run simulations
        for i in range(n_simulations):
            # Sample game scores
            home_score = max(70, np.random.normal(home_pred, home_std))
            away_score = max(70, np.random.normal(away_pred, away_std))
            results["home_scores"][i] = home_score
            results["away_scores"][i] = away_score

            # Simulate each player
            for player in home_roster:
                self._simulate_player(
                    results, player, context, is_home=True, sim_idx=i
                )

            for player in away_roster:
                self._simulate_player(
                    results, player, context, is_home=False, sim_idx=i
                )

        return results

    def _simulate_player(
        self,
        results: Dict,
        player: PlayerFeatures,
        context: GameContext,
        is_home: bool,
        sim_idx: int
    ):
        """Simulate a single player for one iteration."""
        # Get rest days and opponent defense
        rest_days = context.home_rest_days if is_home else context.away_rest_days
        opp_defense = context.away_opp_pts_L10 if is_home else context.home_opp_pts_L10

        # Simulate minutes
        minutes_features = self.transformer.build_minutes_features(
            player, rest_days, is_home
        ).reshape(1, -1)

        min_pred = self.minutes_model.predict(minutes_features)[0]
        tier = self._determine_tier(player.min_szn_avg)
        min_std = self._get_minutes_variance(tier)

        sim_minutes = np.random.normal(min_pred, min_std)
        sim_minutes = np.clip(sim_minutes, 0, 48)
        results["players"][player.player_name]["minutes"][sim_idx] = sim_minutes

        # Simulate each stat
        for stat in self.STAT_TYPES:
            stat_features = self.transformer.build_stats_features(
                player, stat, opp_defense, is_home
            ).reshape(1, -1)

            stat_model = self.stats_models[stat]["model"]
            stat_pred = stat_model.predict(stat_features)[0]
            stat_std = self._get_stat_variance(stat, stat_pred)

            # Minutes adjustment: only for extreme deviations from expected
            # The model already accounts for expected minutes via min_L5_avg feature,
            # so we only adjust for significant deviations (injury scenarios, etc.)
            if player.min_L5_avg > 0:
                minutes_ratio = sim_minutes / player.min_L5_avg
                if minutes_ratio < 0.8:
                    # DNP or injury - scale down proportionally
                    minutes_factor = minutes_ratio
                elif minutes_ratio > 1.2:
                    # Extended minutes (teammate injury) - modest boost
                    minutes_factor = 1.0 + 0.25 * (minutes_ratio - 1.0)
                else:
                    # Normal variation (0.8-1.2x) - no adjustment
                    minutes_factor = 1.0
            else:
                minutes_factor = 1.0

            # Use appropriate distribution based on stat type
            if stat in self.COUNT_STATS:
                # Poisson distribution for discrete count data (fg3m, stl, blk)
                # Lambda must be > 0 for Poisson
                lambda_param = max(0.01, stat_pred * minutes_factor)
                sim_stat = float(np.random.poisson(lambda_param))
            else:
                # Normal distribution for higher-volume stats (pts, reb, ast, tov)
                sim_stat = np.random.normal(stat_pred * minutes_factor, stat_std)
                sim_stat = max(0, sim_stat)  # Can't be negative

            results["players"][player.player_name][stat][sim_idx] = sim_stat

    def _aggregate_results(
        self,
        sim_results: Dict,
        context: GameContext,
        all_players: List[PlayerFeatures],
        n_simulations: int
    ) -> SimulationResult:
        """Aggregate simulation results into distributions."""
        # Score distributions
        home_score_dist = StatDistribution.from_simulations(
            sim_results["home_scores"]
        )
        away_score_dist = StatDistribution.from_simulations(
            sim_results["away_scores"]
        )

        # Player predictions
        player_predictions = {}

        for player in all_players:
            player_data = sim_results["players"].get(player.player_name)
            if not player_data:
                continue

            player_predictions[player.player_name] = PlayerPrediction(
                player_id=player.player_id,
                player_name=player.player_name,
                team_abbr=player.team_abbr,
                minutes=StatDistribution.from_simulations(player_data["minutes"]),
                pts=StatDistribution.from_simulations(player_data["pts"]),
                reb=StatDistribution.from_simulations(player_data["reb"]),
                ast=StatDistribution.from_simulations(player_data["ast"]),
                stl=StatDistribution.from_simulations(player_data["stl"]),
                blk=StatDistribution.from_simulations(player_data["blk"]),
                tov=StatDistribution.from_simulations(player_data["tov"]),
                fg3m=StatDistribution.from_simulations(player_data["fg3m"])
            )

        # Copy raw simulations for empirical probability calculation
        raw_simulations = {
            player_name: {
                stat: player_data[stat].copy()
                for stat in self.STAT_TYPES
            }
            for player_name, player_data in sim_results["players"].items()
        }

        return SimulationResult(
            home_team=context.home_team_abbr,
            away_team=context.away_team_abbr,
            game_date=context.game_date,
            n_simulations=n_simulations,
            home_score=home_score_dist,
            away_score=away_score_dist,
            players=player_predictions,
            raw_simulations=raw_simulations
        )

    def simulate_player_prop(
        self,
        player_name: str,
        stat: str,
        line: float,
        opponent: str,
        is_home: bool = True,
        over_odds: str = "-110",
        under_odds: str = "-110",
        injuries: Dict[str, str] = None,
        n_simulations: int = 10000
    ) -> PropAnalysis:
        """
        Simulate a single player prop bet.

        Args:
            player_name: Player's full name
            stat: Stat type (pts, reb, ast, etc.)
            line: Betting line (e.g., 25.5)
            opponent: Opponent team abbreviation
            is_home: Whether player's team is home
            over_odds: American odds for over
            under_odds: American odds for under
            injuries: Injury dict (optional)
            n_simulations: Number of simulations

        Returns:
            PropAnalysis with recommendation
        """
        print(f"Simulating {player_name} {stat} o/u {line}...")

        # Get player features
        player = self.transformer.get_player_features(player_name)
        if not player:
            raise ValueError(f"Could not find player: {player_name}")

        # Determine teams
        home_team = player.team_abbr if is_home else opponent
        away_team = opponent if is_home else player.team_abbr

        # Get game context
        context = self.transformer.get_game_context(home_team, away_team)

        # Apply injury adjustments if needed
        if injuries and player.player_name in injuries:
            raise ValueError(f"{player_name} is marked as {injuries[player.player_name]}")

        # Run simulations for this player only
        opp_defense = context.away_opp_pts_L10 if is_home else context.home_opp_pts_L10
        rest_days = context.home_rest_days if is_home else context.away_rest_days

        simulated_stats = np.zeros(n_simulations)

        for i in range(n_simulations):
            # Simulate minutes
            minutes_features = self.transformer.build_minutes_features(
                player, rest_days, is_home
            ).reshape(1, -1)

            min_pred = self.minutes_model.predict(minutes_features)[0]
            tier = self._determine_tier(player.min_szn_avg)
            min_std = self._get_minutes_variance(tier)
            sim_minutes = np.clip(np.random.normal(min_pred, min_std), 0, 48)

            # Simulate stat
            stat_features = self.transformer.build_stats_features(
                player, stat, opp_defense, is_home
            ).reshape(1, -1)

            stat_model = self.stats_models[stat]["model"]
            stat_pred = stat_model.predict(stat_features)[0]
            stat_std = self._get_stat_variance(stat, stat_pred)

            # Minutes adjustment: only for extreme deviations
            if player.min_L5_avg > 0:
                minutes_ratio = sim_minutes / player.min_L5_avg
                if minutes_ratio < 0.8:
                    minutes_factor = minutes_ratio
                elif minutes_ratio > 1.2:
                    minutes_factor = 1.0 + 0.25 * (minutes_ratio - 1.0)
                else:
                    minutes_factor = 1.0
            else:
                minutes_factor = 1.0

            # Use appropriate distribution based on stat type
            if stat in self.COUNT_STATS:
                lambda_param = max(0.01, stat_pred * minutes_factor)
                sim_stat = float(np.random.poisson(lambda_param))
            else:
                sim_stat = max(0, np.random.normal(stat_pred * minutes_factor, stat_std))
            simulated_stats[i] = sim_stat

        # Analyze prop
        prop = PropBet(
            player_name=player_name,
            stat=stat,
            line=line,
            over_odds=over_odds,
            under_odds=under_odds
        )

        return self.edge_calc.analyze_prop(simulated_stats, prop)

    def get_player_distribution(
        self,
        player_name: str,
        result: SimulationResult
    ) -> Optional[PlayerPrediction]:
        """Get prediction for a specific player from simulation result."""
        return result.players.get(player_name)

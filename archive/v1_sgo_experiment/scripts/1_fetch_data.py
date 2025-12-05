#!/usr/bin/env python3
"""
Fetch historical NBA finalized events from SGO API.

This script pulls all completed NBA games with player props and results.
Designed to work within the free tier limit (2,500 entities/month).

Usage:
    python scripts/fetch_historical_sgo.py [--season 2024-25] [--dry-run]

Output:
    data/raw/sgo_historical_SEASON.json
"""

import os
import json
import time
import argparse
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load API key
load_dotenv()
API_KEY = os.getenv('SGO_API_KEY') or os.getenv('sgo_api')
if API_KEY:
    API_KEY = API_KEY.strip().strip("'\"")

if not API_KEY:
    print("ERROR: API key not found in .env")
    exit(1)

BASE_URL = 'https://api.sportsgameodds.com/v2/events'
USAGE_URL = 'https://api.sportsgameodds.com/v2/account/usage'

# Season date ranges
SEASONS = {
    '2024-25': {'start': '2024-10-22', 'end': '2025-04-15'},
    '2023-24': {'start': '2023-10-24', 'end': '2024-04-14'},
}


def get_usage():
    """Get current API usage."""
    headers = {'X-Api-Key': API_KEY}
    resp = requests.get(USAGE_URL, headers=headers)
    data = resp.json()
    limits = data['data']['rateLimits']['per-month']
    return {
        'max': int(limits['max-entities']),
        'current': int(limits['current-entities']),
        'remaining': int(limits['max-entities']) - int(limits['current-entities'])
    }


def fetch_events(starts_after: str, starts_before: str, limit: int = 100) -> list:
    """
    Fetch all finalized NBA events within date range.
    Uses cursor-based pagination.
    """
    headers = {'X-Api-Key': API_KEY}
    all_events = []
    cursor = None
    page = 1

    while True:
        params = {
            'leagueID': 'NBA',
            'startsAfter': starts_after,
            'startsBefore': starts_before,
            'finalized': 'true',
            'limit': limit,
        }
        if cursor:
            params['cursor'] = cursor

        print(f"  Page {page}: fetching up to {limit} events...")

        try:
            resp = requests.get(BASE_URL, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  ERROR: {e}")
            break

        events = data.get('data', [])
        if not events:
            break

        all_events.extend(events)
        print(f"    Got {len(events)} events (total: {len(all_events)})")

        cursor = data.get('nextCursor')
        if not cursor:
            break

        page += 1
        time.sleep(0.3)  # Rate limiting

    return all_events


def main():
    parser = argparse.ArgumentParser(description='Fetch historical NBA data from SGO')
    parser.add_argument('--season', choices=['2024-25', '2023-24', 'both'],
                        default='2024-25', help='Which season to fetch')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be fetched without making API calls')
    args = parser.parse_args()

    # Check usage first
    usage = get_usage()
    print("=== SGO API Usage ===")
    print(f"Monthly limit:  {usage['max']:,} entities")
    print(f"Current usage:  {usage['current']:,} entities")
    print(f"Remaining:      {usage['remaining']:,} entities")
    print()

    # Determine seasons to fetch
    if args.season == 'both':
        seasons_to_fetch = ['2024-25', '2023-24']
    else:
        seasons_to_fetch = [args.season]

    # Estimate games needed
    estimated_games = sum(400 if s == '2024-25' else 1230 for s in seasons_to_fetch)
    print(f"Estimated games needed: ~{estimated_games}")

    if estimated_games > usage['remaining']:
        print(f"WARNING: May exceed remaining entities ({usage['remaining']})")
        if not args.dry_run:
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                return

    if args.dry_run:
        print("\n[DRY RUN] Would fetch these seasons:")
        for season in seasons_to_fetch:
            dates = SEASONS[season]
            print(f"  {season}: {dates['start']} to {dates['end']}")
        return

    # Fetch each season
    for season in seasons_to_fetch:
        dates = SEASONS[season]
        print(f"\n=== Fetching {season} Season ===")
        print(f"Date range: {dates['start']} to {dates['end']}")

        events = fetch_events(dates['start'], dates['end'])

        if not events:
            print(f"No events found for {season}")
            continue

        # Calculate stats
        total_odds = sum(len(e.get('odds', {})) for e in events)
        player_props = sum(
            1 for e in events
            for o in e.get('odds', {}).values()
            if o.get('playerID') and 'game-ou' in o.get('oddID', '')
        )

        print(f"\n=== {season} Results ===")
        print(f"Games fetched: {len(events)}")
        print(f"Total odds: {total_odds:,}")
        print(f"Player props (over/under): {player_props:,}")

        # Save to file
        output = {
            'season': season,
            'fetched_at': datetime.now().isoformat(),
            'date_range': dates,
            'event_count': len(events),
            'events': events
        }

        output_path = Path(__file__).parent.parent / 'data' / 'raw' / f'sgo_historical_{season.replace("-", "")}.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(output, f, default=str)

        print(f"Saved to: {output_path}")

    # Final usage check
    usage = get_usage()
    print(f"\n=== Final Usage ===")
    print(f"Remaining: {usage['remaining']:,} entities")


if __name__ == '__main__':
    main()

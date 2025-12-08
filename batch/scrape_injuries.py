#!/usr/bin/env python3
"""
Injury scraper for ESPN data.

Run hourly or before precompute:
    python batch/scrape_injuries.py

This populates the injuries table with current player statuses.
"""
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Database connection
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:brain123@localhost:5432/brain")

# ESPN URL
ESPN_INJURIES_URL = "https://www.espn.com/nba/injuries"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

# Team name to abbreviation mapping
TEAM_ABBREVS = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards"
}

# Reverse mapping (lowercase team name -> abbrev)
TEAM_NAME_TO_ABBREV = {v.lower(): k for k, v in TEAM_ABBREVS.items()}


def get_db():
    """Get database connection."""
    import psycopg2
    from psycopg2.extras import RealDictCursor
    conn = psycopg2.connect(DATABASE_URL)
    conn.cursor_factory = RealDictCursor
    return conn


def scrape_espn() -> dict[str, list[dict]]:
    """
    Scrape ESPN NBA injury page.

    Returns:
        Dict mapping team abbrev to list of injuries:
        {
            "LAL": [
                {"player": "Anthony Davis", "status": "OUT", "injury": "Knee"},
                ...
            ],
            ...
        }
    """
    print(f"Scraping ESPN injuries from {ESPN_INJURIES_URL}...")

    try:
        response = requests.get(ESPN_INJURIES_URL, headers=HEADERS, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching ESPN: {e}")
        return {}

    soup = BeautifulSoup(response.text, "html.parser")
    injuries_by_team = {}

    # Find all team injury tables
    # ESPN uses ResponsiveTable class for injury tables
    tables = soup.find_all("div", class_="ResponsiveTable")

    for table in tables:
        # Get team name from inside the table wrapper (ESPN structure)
        team_header = table.find("span", class_=lambda c: c and "injuries__teamName" in c)
        if not team_header:
            # Try Table__Title div
            title_div = table.find("div", class_="Table__Title")
            if title_div:
                team_header = title_div

        if not team_header:
            continue

        team_name = team_header.get_text(strip=True).lower()

        # Map team name to abbreviation
        team_abbrev = None
        for name, abbrev in TEAM_NAME_TO_ABBREV.items():
            if name in team_name or team_name in name:
                team_abbrev = abbrev
                break

        if not team_abbrev:
            # Try partial match
            for full_name, abbrev in TEAM_ABBREVS.items():
                if any(word in team_name for word in full_name.lower().split()):
                    team_abbrev = abbrev
                    break

        if not team_abbrev:
            print(f"  Warning: Could not map team name '{team_name}'")
            continue

        # Parse injury rows
        rows = table.find_all("tr", class_="Table__TR")
        team_injuries = []

        for row in rows:
            cells = row.find_all("td")
            # ESPN columns: Name, Position, Return Date, Status, Description
            if len(cells) >= 4:
                player_name = cells[0].get_text(strip=True)
                status = cells[3].get_text(strip=True).upper()
                injury = cells[4].get_text(strip=True) if len(cells) > 4 else ""

                # Normalize status
                if "OUT" in status:
                    status = "OUT"
                elif "DOUBTFUL" in status:
                    status = "DOUBTFUL"
                elif "QUESTIONABLE" in status or "GTD" in status:
                    status = "QUESTIONABLE"
                elif "PROBABLE" in status:
                    status = "PROBABLE"
                elif "DAY-TO-DAY" in status:
                    status = "QUESTIONABLE"

                if player_name and status:
                    team_injuries.append({
                        "player": player_name,
                        "status": status,
                        "injury": injury
                    })

        if team_injuries:
            injuries_by_team[team_abbrev] = team_injuries

    total_injuries = sum(len(inj) for inj in injuries_by_team.values())
    print(f"  Found {total_injuries} injuries across {len(injuries_by_team)} teams")

    return injuries_by_team


def update_injuries_table(injuries: dict[str, list[dict]]) -> int:
    """
    Update injuries table with scraped data.

    Clears ESPN injuries and inserts fresh data.
    Preserves USER-sourced injuries (manual overrides).

    Args:
        injuries: Dict from scrape_espn()

    Returns:
        Number of injuries inserted
    """
    if not injuries:
        print("No injuries to update")
        return 0

    conn = get_db()
    cursor = conn.cursor()

    # Clear old ESPN injuries (preserve USER overrides)
    cursor.execute("DELETE FROM injuries WHERE source = 'ESPN'")
    deleted = cursor.rowcount
    print(f"  Cleared {deleted} old ESPN injuries")

    # Insert fresh injuries
    count = 0
    for team_abbrev, team_injuries in injuries.items():
        for injury in team_injuries:
            cursor.execute("""
                INSERT INTO injuries (player_name, team, status, injury, source, updated_at)
                VALUES (%s, %s, %s, %s, 'ESPN', NOW())
                ON CONFLICT (player_name, team) DO UPDATE SET
                    status = EXCLUDED.status,
                    injury = EXCLUDED.injury,
                    source = EXCLUDED.source,
                    updated_at = NOW()
            """, (
                injury["player"],
                team_abbrev,
                injury["status"],
                injury["injury"]
            ))
            count += 1

    conn.commit()
    cursor.close()
    conn.close()

    print(f"  Inserted {count} injuries")
    return count


def set_injury(player_name: str, status: str, team: str, injury: str = "") -> None:
    """
    Manually set a player's injury status (USER override).

    Args:
        player_name: Full player name
        status: OUT, DOUBTFUL, QUESTIONABLE, PROBABLE
        team: 3-letter team code
        injury: Injury description
    """
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO injuries (player_name, team, status, injury, source, updated_at)
        VALUES (%s, %s, %s, %s, 'USER', NOW())
        ON CONFLICT (player_name, team) DO UPDATE SET
            status = EXCLUDED.status,
            injury = EXCLUDED.injury,
            source = 'USER',
            updated_at = NOW()
    """, (player_name, team, status.upper(), injury))

    conn.commit()
    cursor.close()
    conn.close()

    print(f"Set {player_name} ({team}) as {status}")


def main() -> None:
    """Run injury scrape and update database."""
    start = datetime.now()
    print(f"{'='*60}")
    print(f"Starting injury scrape at {start}")
    print(f"{'='*60}")

    injuries = scrape_espn()

    if injuries:
        count = update_injuries_table(injuries)
        print(f"\nUpdated {count} injury records")

        # Print summary by team
        print("\nInjury Summary:")
        for team, team_injuries in sorted(injuries.items()):
            players = ", ".join(f"{i['player']} ({i['status']})" for i in team_injuries[:3])
            if len(team_injuries) > 3:
                players += f" +{len(team_injuries)-3} more"
            print(f"  {team}: {players}")
    else:
        print("\nNo injuries found or error occurred")

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n{'='*60}")
    print(f"Complete in {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

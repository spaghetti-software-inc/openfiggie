#!/usr/bin/env python3
"""
Monte Carlo Simulation of a Figgie Round Using PFN (Portable Figgie Notation)
with Randomized Initial Deal, Bayesian Inference, and an Interactive REPL
to allow the user (P1) to play with the bots. Debugging and raw PFN output
are suppressed in this interactive mode.
Author: [Your Name]
Date: 2025-02-15
"""

import random
import math
import toml  # pip install toml

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# Create a global console object.
console = Console()

# --- Helper Functions --- #

def logistic(x):
    """Compute the logistic function."""
    return 1.0 / (1.0 + math.exp(-x))

def valuation_given_candidate(card_suit, candidate_goal):
    """
    Returns the valuation for a card of suit `card_suit` given that the
    candidate goal suit is `candidate_goal`.
      - If the card suit equals the candidate goal, value is 30.
      - If the card suit is the same color as candidate goal, value is 20.
      - Otherwise, value is 10.
    """
    black_suits = ["Spades", "Clubs"]
    red_suits   = ["Hearts", "Diamonds"]
    if card_suit == candidate_goal:
        return 30
    elif (card_suit in black_suits and candidate_goal in black_suits) or \
         (card_suit in red_suits and candidate_goal in red_suits):
        return 20
    else:
        return 10

def expected_value(card_suit, beliefs):
    """
    Computes the expected value of a card of suit `card_suit` based on
    a player's current belief distribution.
    """
    ev = 0
    for candidate, prob in beliefs.items():
        ev += prob * valuation_given_candidate(card_suit, candidate)
    return ev

def update_beliefs(beliefs, card_suit, price, sigma=3.0):
    """
    Updates the belief distribution based on an observed trade of a card of suit
    `card_suit` at price `price` using a Gaussian likelihood.
    """
    new_beliefs = {}
    total = 0.0
    for candidate, prob in beliefs.items():
        v_candidate = valuation_given_candidate(card_suit, candidate)
        likelihood = math.exp(-((price - v_candidate) ** 2) / (2 * sigma**2))
        new_beliefs[candidate] = prob * likelihood
        total += new_beliefs[candidate]
    if total > 0:
        for candidate in new_beliefs:
            new_beliefs[candidate] /= total
        return new_beliefs
    else:
        return {c: 1.0 / len(beliefs) for c in beliefs}

# --- Player Class --- #

class Player:
    def __init__(self, name, hand, money):
        self.name = name
        self.hand = hand  # e.g., ["♠1", "♣5", ...]
        self.money = money
        # Initialize uniform belief over possible goal suits.
        self.beliefs = {"Spades": 0.25, "Clubs": 0.25, "Hearts": 0.25, "Diamonds": 0.25}

# --- Global Game State Initialization --- #

# PFN configuration (hidden to players; actual goal suit is used for outcome)
config = {
    "FiggieGame": {
        "Title": "Interactive Figgie Round",
        "GameID": "G12345",
        "Players": 4,
        "Date": "2025-02-15",
        "GameDuration": 60.0,  # seconds
        "GameVariant": "Standard"
    },
    "DeckSetup": {
        "GoalSuitColor": "Black",  # used in deck design
        "GoalSuit": "Spades",       # actual goal suit (secret during play)
        "Distribution": {
            "Spades": 10,
            "Clubs": 12,
            "Hearts": 10,
            "Diamonds": 8
        }
    }
}

# Use Unicode suit symbols.
suit_unicode_map = {"Spades": "♠", "Clubs": "♣", "Hearts": "♥", "Diamonds": "♦"}
# Reverse mapping.
unicode_to_name = {"♠": "Spades", "♣": "Clubs", "♥": "Hearts", "♦": "Diamonds"}

# Build the deck based on the distribution.
deck_distribution = config["DeckSetup"]["Distribution"]
deck = []
for suit, count in deck_distribution.items():
    symbol = suit_unicode_map[suit]
    for i in range(1, count + 1):
        deck.append(symbol + str(i))
random.shuffle(deck)

# Deal cards in round-robin.
num_players = config["FiggieGame"]["Players"]
initial_deal = {f"P{i+1}": [] for i in range(num_players)}
for idx, card in enumerate(deck):
    player_key = f"P{(idx % num_players) + 1}"
    initial_deal[player_key].append(card)
# Prepare a string version for PFN (not printed).
deal_str = {p: ",".join(cards) for p, cards in initial_deal.items()}

# Create player objects.
players = {}
INITIAL_MONEY = 350
for pname, cards in initial_deal.items():
    players[pname] = Player(pname, cards.copy(), INITIAL_MONEY)

# Global game parameters.
current_time = 0.0
game_duration = config["FiggieGame"]["GameDuration"]
pot_amount = 100
trade_events = []
trade_index = 1
actual_goal_suit = config["DeckSetup"]["GoalSuit"]

# For interactive play, designate P1 as the human player.
HUMAN_PLAYER = "P1"

# --- REPL Functions --- #

def show_help():
    help_text = """
Available commands:
  advance (a)   - Simulate the next trade candidate event.
  status  (s)   - Show current game status.
  help    (h)   - Show this help message.
  quit    (q)   - Quit the game.
    """
    console.print(Panel(help_text.strip(), title="Help", style="cyan"))

def show_status():
    status_table = Table(title="Current Status", show_edge=True)
    status_table.add_column("Player", style="bold")
    status_table.add_column("Money", justify="right")
    status_table.add_column("Hand")
    status_table.add_column("Beliefs", style="dim")
    for pname, player in players.items():
        status_table.add_row(
            pname,
            str(player.money),
            ", ".join(player.hand),
            str(player.beliefs)
        )
    status_table.add_row("Time", f"{current_time:.2f}", "", "")
    console.print(status_table)

def simulate_trade_event():
    """Simulate one candidate trade event; if HUMAN_PLAYER is involved, prompt for decision."""
    global current_time, trade_index
    dt = random.expovariate(1 / 5.0)
    current_time += dt
    if current_time >= game_duration:
        return None  # Signal game time is over.

    # Select a seller with at least one card.
    seller_candidates = [p for p in players.values() if p.hand]
    if not seller_candidates:
        return "No seller available."
    seller = random.choice(seller_candidates)

    # Select a random card from seller.
    card = random.choice(seller.hand)
    symbol = card[0]
    suit = unicode_to_name[symbol]

    # Select a buyer different from seller.
    buyer_candidates = [p for p in players.values() if p.name != seller.name]
    if not buyer_candidates:
        return "No buyer available."
    buyer = random.choice(buyer_candidates)

    # Compute expected values.
    buyer_ev = expected_value(suit, buyer.beliefs)
    seller_ev = expected_value(suit, seller.beliefs)
    noise = random.uniform(-2, 2)
    candidate_price = max(1, (buyer_ev + seller_ev) / 2 + noise)

    # Check buyer funds.
    if buyer.money < candidate_price:
        return f"T={current_time:.2f}: Trade candidate skipped – Buyer {buyer.name} lacks funds."

    # Compute acceptance probabilities.
    alpha = 0.5; beta = 0.5
    buyer_prob = logistic(alpha * (buyer_ev - candidate_price))
    seller_prob = logistic(beta * (candidate_price - seller_ev))
    buyer_roll = random.random()
    seller_roll = random.random()

    # Prepare candidate trade message.
    message = (f"T={current_time:.2f}: Seller {seller.name} offers {card} ({suit}) | "
               f"Candidate Price: {candidate_price:.2f} | Buyer EV: {buyer_ev:.2f}, Seller EV: {seller_ev:.2f} | "
               f"Buyer Prob: {buyer_prob:.2f}, Seller Prob: {seller_prob:.2f} | "
               f"Rolls: Buyer {buyer_roll:.2f}, Seller {seller_roll:.2f}")

    # If HUMAN_PLAYER is involved, override the decision.
    if buyer.name == HUMAN_PLAYER or seller.name == HUMAN_PLAYER:
        console.print(Text(message, style="yellow"))
        decision = console.input("[bold blue]You are involved in this trade. Accept it? (y/n): [/bold blue]").strip().lower()
        if decision == "y":
            accepted = True
        else:
            accepted = False
    else:
        # Bots decide automatically.
        accepted = (buyer_roll < buyer_prob and seller_roll < seller_prob)

    if accepted:
        seller.hand.remove(card)
        buyer.hand.append(card)
        buyer.money -= candidate_price
        seller.money += candidate_price
        # Record the trade event.
        event = {
            "TradeIndex": trade_index,
            "T": round(current_time, 2),
            "Buyer": buyer.name,
            "Seller": seller.name,
            "Suit": suit,
            "Card": card,
            "Price": round(candidate_price, 2)
        }
        trade_events.append(event)
        trade_index += 1
        result_msg = f"Trade Executed: {buyer.name} buys {card} ({suit}) from {seller.name} at {candidate_price:.2f}."
        # After a trade, all players update their beliefs.
        for p in players.values():
            p.beliefs = update_beliefs(p.beliefs, suit, candidate_price, sigma=3.0)
        return result_msg
    else:
        return (f"Trade Rejected: Buyer {buyer.name} (roll {buyer_roll:.2f} vs. {buyer_prob:.2f}) "
                f"or Seller {seller.name} (roll {seller_roll:.2f} vs. {seller_prob:.2f}) declined.")

# --- Main Interactive REPL Loop --- #

def repl():
    console.rule("[bold blue]Interactive Figgie REPL[/bold blue]")
    show_help()
    while current_time < game_duration:
        cmd = console.input("[bold green]REPL> [/bold green]").strip().lower()
        if cmd in ["advance", "a"]:
            event_result = simulate_trade_event()
            if event_result is None:
                console.print("[bold red]Game time has expired.[/bold red]")
                break
            console.print(Text(event_result, style="magenta"))
        elif cmd in ["status", "s"]:
            show_status()
        elif cmd in ["help", "h"]:
            show_help()
        elif cmd in ["quit", "q"]:
            console.print("[bold red]Quitting the game.[/bold red]")
            break
        else:
            console.print("[bold red]Unknown command. Type 'help' for available commands.[/bold red]")

    # After time expires or quit.
    console.rule("[bold green]Final Results[/bold green]")
    # Compute goal card counts.
    goal_counts = {}
    for p in players.values():
        count = sum(1 for card in p.hand if unicode_to_name[card[0]] == actual_goal_suit)
        goal_counts[p.name] = count
    max_count = max(goal_counts.values()) if goal_counts else 0
    winners = [name for name, count in goal_counts.items() if count == max_count]

    share = pot_amount / len(winners) if winners else 0
    for p in players.values():
        if p.name in winners:
            p.money += share

    result_table = Table(title="Game Results", show_edge=True)
    result_table.add_column("Player", style="bold")
    result_table.add_column("Final Bank", justify="right")
    result_table.add_column("Goal Cards", justify="right")
    result_table.add_column("Beliefs", style="dim")
    for pname, player in players.items():
        result_table.add_row(
            pname,
            str(player.money),
            str(goal_counts.get(pname, 0)),
            str(player.beliefs)
        )
    console.print(result_table)
    console.print(f"[bold blue]Winners: {', '.join(winners)}[/bold blue]")

# --- Run the REPL --- #

def main():
    console.rule("[bold blue]Welcome to Figgie![/bold blue]")
    repl()

if __name__ == "__main__":
    main()

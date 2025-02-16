#!/usr/bin/env python3
"""
Turn-Based Interactive Simulation of Figgie (Portable Figgie Notation)
with Bayesian Inference and a Rich-Powered Unicode Interface.
In this version, the human (P1) gets a chance each turn to interact with each bot
by proposing trades or passing. The human's current hand and money are displayed
at the very start and whenever you are prompted.
A command-line option (--hide-opponents) allows the user to hide the other players' hands.
Author: [Your Name]
Date: 2025-02-15
"""

import random
import math
import sys
import argparse

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# Create a global rich console.
console = Console()

# Global flag for hiding opponents' hands.
HIDE_OPPONENTS = False

# --- Helper Functions ---

def logistic(x):
    """Return the logistic function value for x."""
    return 1.0 / (1.0 + math.exp(-x))

def valuation_given_candidate(card_suit, candidate_goal):
    """
    Return the value of a card of suit `card_suit` if candidate_goal is the valuable suit.
      - 30 if card_suit equals candidate_goal.
      - 20 if card_suit is the same color as candidate_goal.
      - 10 otherwise.
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
    Compute expected value of a card of suit `card_suit` based on a player's beliefs.
    """
    ev = 0
    for candidate, prob in beliefs.items():
        ev += prob * valuation_given_candidate(card_suit, candidate)
    return ev

def update_beliefs(beliefs, card_suit, price, sigma=3.0):
    """
    Update a belief distribution given an observed trade of a card of suit `card_suit`
    at a given price. Uses a Gaussian likelihood.
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

# --- Player Class ---

class Player:
    def __init__(self, name, hand, money):
        self.name = name
        self.hand = hand[:]  # e.g., ["♠1", "♣5", ...]
        self.money = money
        # Each player starts with uniform beliefs over the four candidate goal suits.
        self.beliefs = {"Spades": 0.25, "Clubs": 0.25, "Hearts": 0.25, "Diamonds": 0.25}

# --- Global Game Setup ---

# Game configuration (hidden to players)
config = {
    "FiggieGame": {
        "Title": "Turn-Based Figgie Round",
        "GameID": "G12345",
        "Players": 4,
        "Date": "2025-02-15",
        "Turns": 10,  # Maximum number of turns.
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

# Use Unicode symbols for suits.
suit_unicode_map = {"Spades": "♠", "Clubs": "♣", "Hearts": "♥", "Diamonds": "♦"}
# Reverse mapping: from Unicode symbol to full name.
unicode_to_name = {"♠": "Spades", "♣": "Clubs", "♥": "Hearts", "♦": "Diamonds"}

# Build deck from distribution.
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
    key = f"P{(idx % num_players) + 1}"
    initial_deal[key].append(card)

# Create Player objects.
INITIAL_MONEY = 350
players = {}
for pname, cards in initial_deal.items():
    players[pname] = Player(pname, cards, INITIAL_MONEY)

# Global parameters.
turn_number = 1
pot_amount = 100
# For outcome purposes, we know the actual goal suit.
actual_goal_suit = config["DeckSetup"]["GoalSuit"]

# Designate P1 as the human player.
HUMAN_PLAYER = "P1"

# --- Trade Mechanism Functions ---

def human_propose_trade(opponent):
    """
    Let the human (P1) propose a trade with a given opponent.
    Displays P1's current hand and money.
    Returns a message string.
    """
    human = players[HUMAN_PLAYER]
    hand_str = ", ".join(human.hand)
    console.print(Panel(f"Your current hand:\n{hand_str}\nMoney: {human.money}", title="Your Hand", style="bold green"))
    
    console.print(Panel(f"Your turn to propose a trade with {opponent.name}.", style="bold blue"))
    action = console.input("[bold]Do you want to [B]uy or [S]ell? (B/S): [/bold]").strip().lower()
    if action not in ("b", "s"):
        return "Invalid action. Trade cancelled."
    
    suit = console.input("[bold]Enter the suit (Spades, Clubs, Hearts, Diamonds): [/bold]").strip().capitalize()
    if suit not in ["Spades", "Clubs", "Hearts", "Diamonds"]:
        return "Invalid suit. Trade cancelled."
    
    try:
        price = float(console.input("[bold]Enter your proposed price: [/bold]").strip())
    except ValueError:
        return "Invalid price. Trade cancelled."
    
    alpha = 0.5
    beta = 0.5
    
    if action == "b":
        opponent_cards = [card for card in opponent.hand if unicode_to_name[card[0]] == suit]
        if not opponent_cards:
            return f"{opponent.name} has no {suit} cards. Trade cannot proceed."
        bot_ev = expected_value(suit, opponent.beliefs)
        accept_prob = logistic(beta * (price - bot_ev))
        roll = random.random()
        if roll < accept_prob:
            traded_card = opponent_cards[0]
            opponent.hand.remove(traded_card)
            human.hand.append(traded_card)
            human.money -= price
            opponent.money += price
            for p in players.values():
                p.beliefs = update_beliefs(p.beliefs, suit, price, sigma=3.0)
            return f"Trade Executed: You bought {traded_card} ({suit}) from {opponent.name} at {price:.2f}."
        else:
            return f"Trade Rejected by {opponent.name} (roll {roll:.2f} vs. accept prob {accept_prob:.2f})."
    
    else:
        human_cards = [card for card in human.hand if unicode_to_name[card[0]] == suit]
        if not human_cards:
            return f"You have no {suit} cards to sell. Trade cannot proceed."
        bot_ev = expected_value(suit, opponent.beliefs)
        accept_prob = logistic(alpha * (bot_ev - price))
        roll = random.random()
        if roll < accept_prob:
            traded_card = human_cards[0]
            human.hand.remove(traded_card)
            opponent.hand.append(traded_card)
            human.money += price
            opponent.money -= price
            for p in players.values():
                p.beliefs = update_beliefs(p.beliefs, suit, price, sigma=3.0)
            return f"Trade Executed: You sold {traded_card} ({suit}) to {opponent.name} at {price:.2f}."
        else:
            return f"Trade Rejected by {opponent.name} (roll {roll:.2f} vs. accept prob {accept_prob:.2f})."

def bot_propose_trade(opponent):
    """
    Have the bot (opponent) propose a trade to the human.
    Returns a message string.
    """
    action = random.choice(["buy", "sell"])
    if action == "buy":
        human_cards = players[HUMAN_PLAYER].hand
        if human_cards:
            card = random.choice(human_cards)
            suit = unicode_to_name[card[0]]
        else:
            suit = random.choice(["Spades", "Clubs", "Hearts", "Diamonds"])
        bot_ev = expected_value(suit, opponent.beliefs)
        price = bot_ev + random.uniform(0, 2)
        response = console.input(f"[bold]Bot {opponent.name} proposes to BUY your {suit} card for {price:.2f}. Accept? (y/n): [/bold]").strip().lower()
        if response == "y":
            human_cards = [card for card in players[HUMAN_PLAYER].hand if unicode_to_name[card[0]] == suit]
            if not human_cards:
                return f"You have no {suit} cards to sell. Trade cancelled."
            traded_card = human_cards[0]
            players[HUMAN_PLAYER].hand.remove(traded_card)
            opponent.hand.append(traded_card)
            players[HUMAN_PLAYER].money += price
            opponent.money -= price
            for p in players.values():
                p.beliefs = update_beliefs(p.beliefs, suit, price, sigma=3.0)
            return f"Trade Executed: You sold {traded_card} ({suit}) to {opponent.name} at {price:.2f}."
        else:
            return f"You declined Bot {opponent.name}'s proposal to buy your {suit} card."
    else:
        bot_cards = [card for card in opponent.hand]
        if bot_cards:
            card = random.choice(bot_cards)
            suit = unicode_to_name[card[0]]
        else:
            suit = random.choice(["Spades", "Clubs", "Hearts", "Diamonds"])
        bot_ev = expected_value(suit, opponent.beliefs)
        price = bot_ev - random.uniform(0, 2)
        response = console.input(f"[bold]Bot {opponent.name} proposes to SELL you a {suit} card for {price:.2f}. Buy? (y/n): [/bold]").strip().lower()
        if response == "y":
            bot_cards = [card for card in opponent.hand if unicode_to_name[card[0]] == suit]
            if not bot_cards:
                return f"Bot {opponent.name} has no {suit} cards to sell. Trade cancelled."
            traded_card = bot_cards[0]
            opponent.hand.remove(traded_card)
            players[HUMAN_PLAYER].hand.append(traded_card)
            players[HUMAN_PLAYER].money -= price
            opponent.money += price
            for p in players.values():
                p.beliefs = update_beliefs(p.beliefs, suit, price, sigma=3.0)
            return f"Trade Executed: You bought {traded_card} ({suit}) from {opponent.name} at {price:.2f}."
        else:
            return f"You declined Bot {opponent.name}'s proposal to sell a {suit} card."

# --- REPL / Turn Loop Functions ---

def show_status(turn):
    status_table = Table(title=f"Status at Turn {turn}", show_edge=True)
    status_table.add_column("Player", style="bold")
    status_table.add_column("Money", justify="right")
    status_table.add_column("Hand")
    status_table.add_column("Beliefs", style="dim")
    for pname, player in players.items():
        if HIDE_OPPONENTS and pname != HUMAN_PLAYER:
            hand_str = "Hidden"
        else:
            hand_str = ", ".join(player.hand)
        status_table.add_row(
            pname,
            str(player.money),
            hand_str,
            str(player.beliefs)
        )
    console.print(status_table)

def turn_loop():
    global turn_number
    total_turns = config["FiggieGame"]["Turns"]
    opponents = [players[p] for p in players if p != HUMAN_PLAYER]
    while turn_number <= total_turns:
        console.rule(f"[bold blue]Turn {turn_number}[/bold blue]")
        for opponent in opponents:
            console.print(Panel(f"Interaction with {opponent.name}", style="bold magenta"))
            choice = console.input(f"[bold]Do you want to [T]rade with {opponent.name} or [P]ass? (T/P): [/bold]").strip().lower()
            if choice.startswith("t"):
                result = human_propose_trade(opponent)
                console.print(Text(result, style="green"))
            else:
                result = bot_propose_trade(opponent)
                console.print(Text(result, style="cyan"))
        show_status(turn_number)
        cont = console.input("[bold green]Proceed to next turn? (y/n): [/bold green]").strip().lower()
        if cont != "y":
            break
        turn_number += 1

    console.rule("[bold green]Game Over[/bold green]")
    goal_counts = {}
    for p in players.values():
        count = sum(1 for card in p.hand if unicode_to_name[card[0]] == actual_goal_suit)
        goal_counts[p.name] = count
    max_count = max(goal_counts.values()) if goal_counts else 0
    winners = [name for name, count in goal_counts.items() if count == max_count]
    if winners:
        share = pot_amount / len(winners)
        for p in players.values():
            if p.name in winners:
                p.money += share

    final_table = Table(title="Final Results", show_edge=True)
    final_table.add_column("Player", style="bold")
    final_table.add_column("Final Bank", justify="right")
    final_table.add_column("Goal Cards", justify="right")
    final_table.add_column("Beliefs", style="dim")
    for pname, player in players.items():
        if HIDE_OPPONENTS and pname != HUMAN_PLAYER:
            hand_str = "Hidden"
        else:
            hand_str = ", ".join(player.hand)
        final_table.add_row(
            pname,
            str(player.money),
            str(goal_counts.get(pname, 0)),
            str(player.beliefs)
        )
    console.print(final_table)
    console.print(f"[bold blue]Winners: {', '.join(winners)}[/bold blue]")

# --- Main Program Entry Point ---

def main():
    global HIDE_OPPONENTS
    # Process command-line arguments.
    import argparse
    parser = argparse.ArgumentParser(description="Turn-Based Figgie Game")
    parser.add_argument("--hide-opponents", action="store_true", help="Hide other players' hands from display")
    args = parser.parse_args()
    HIDE_OPPONENTS = args.hide_opponents

    console.rule("[bold blue]Welcome to Turn-Based Figgie![/bold blue]")
    console.print(Panel("In this game, you (P1) will interact each turn with each bot.\n"
                        "You may propose a trade (buy or sell) or pass—in which case the bot will propose a trade to you.\n"
                        "Your current hand and money are displayed whenever you are prompted.\n"
                        "Try to secure cards in the secret goal suit!\n"
                        "Type your responses when prompted.", style="bold"))
    # Show the human player's initial hand.
    human = players[HUMAN_PLAYER]
    hand_str = ", ".join(human.hand)
    console.print(Panel(f"Your initial hand:\n{hand_str}\nMoney: {human.money}", title="Your Hand", style="bold green"))
    turn_loop()
    console.print(Panel("Thank you for playing Figgie!", style="bold green"))

if __name__ == "__main__":
    main()

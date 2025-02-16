#!/usr/bin/env python3
"""
Turn-Based Interactive Simulation of Figgie with Pydantic Game State Models
and a Rich-Powered Unicode Interface.
In this version:
  - The game state is represented by Pydantic models.
  - The human player's hand is summarized as counts per suit (using colors for red/black).
  - Real inventories of cards and cash are tracked so no trader can exceed their limits.
  - Single-character responses are used for suit input.
  - Trade executed messages now use Unicode suit symbols (without arbitrary labels).
  - After each executed trade (whether by you or a bot), your updated hand and money are shown.
  - A command-line option (--hide-opponents) hides opponents' hands.
Author: [Your Name]
Date: 2025-02-15
"""

import random
import math
import sys
import argparse
from typing import List, Dict

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from pydantic import BaseModel

# --- Pydantic Models ---

class GameMetadata(BaseModel):
    title: str
    game_id: str
    players: int
    date: str
    turns: int
    game_variant: str

class DeckSetup(BaseModel):
    goal_suit_color: str
    goal_suit: str
    distribution: Dict[str, int]

class PlayerState(BaseModel):
    name: str
    hand: List[str]
    money: int
    beliefs: Dict[str, float]

class TradeEvent(BaseModel):
    trade_index: int
    time: float
    buyer: str
    seller: str
    card: str
    suit: str
    price: float

class GameState(BaseModel):
    metadata: GameMetadata
    deck_setup: DeckSetup
    players: List[PlayerState]
    trades: List[TradeEvent]
    current_turn: int
    pot: int
    actual_goal_suit: str

# --- Global Rich Console ---
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
    at a given price using a Gaussian likelihood.
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

def hand_summary(hand: List[str]) -> str:
    """
    Return a summary string showing the count of cards per suit.
    Black suits (♠, ♣) are shown in blue; red suits (♥, ♦) in red.
    """
    counts = {"Spades": 0, "Clubs": 0, "Hearts": 0, "Diamonds": 0}
    for card in hand:
        suit = unicode_to_name[card[0]]
        counts[suit] += 1
    summary = (
        f"[blue]♠: {counts['Spades']}[/blue]  "
        f"[blue]♣: {counts['Clubs']}[/blue]  "
        f"[red]♥: {counts['Hearts']}[/red]  "
        f"[red]♦: {counts['Diamonds']}[/red]"
    )
    return summary

# --- Player Class ---

class Player:
    def __init__(self, name, hand, money):
        self.name = name
        # Internally store full card labels (e.g. "♣8") but UI shows only counts.
        self.hand = hand[:]
        self.money = money
        self.beliefs = {"Spades": 0.25, "Clubs": 0.25, "Hearts": 0.25, "Diamonds": 0.25}

# --- Global Game Setup ---

config = {
    "FiggieGame": {
        "Title": "Turn-Based Figgie Round",
        "GameID": "G12345",
        "Players": 4,
        "Date": "2025-02-15",
        "Turns": 10,
        "GameVariant": "Standard"
    },
    "DeckSetup": {
        "GoalSuitColor": "Black",
        "GoalSuit": "Spades",
        "Distribution": {
            "Spades": 10,
            "Clubs": 12,
            "Hearts": 10,
            "Diamonds": 8
        }
    }
}

suit_unicode_map = {"Spades": "♠", "Clubs": "♣", "Hearts": "♥", "Diamonds": "♦"}
unicode_to_name = {"♠": "Spades", "♣": "Clubs", "♥": "Hearts", "♦": "Diamonds"}

deck_distribution = config["DeckSetup"]["Distribution"]
deck = []
for suit, count in deck_distribution.items():
    symbol = suit_unicode_map[suit]
    for i in range(1, count + 1):
        # The rank is arbitrary and not shown in the UI.
        deck.append(symbol + str(i))
random.shuffle(deck)

num_players = config["FiggieGame"]["Players"]
initial_deal = {f"P{i+1}": [] for i in range(num_players)}
for idx, card in enumerate(deck):
    key = f"P{(idx % num_players) + 1}"
    initial_deal[key].append(card)

INITIAL_MONEY = 350
players = {}
for pname, cards in initial_deal.items():
    players[pname] = Player(pname, cards, INITIAL_MONEY)

turn_number = 1
pot_amount = 100
actual_goal_suit = config["DeckSetup"]["GoalSuit"]
HUMAN_PLAYER = "P1"

# Global list for trade events.
trade_events_global = []

# --- Function to Build GameState Model ---

def build_game_state(current_turn: int) -> GameState:
    metadata = GameMetadata(
        title=config["FiggieGame"]["Title"],
        game_id=config["FiggieGame"]["GameID"],
        players=config["FiggieGame"]["Players"],
        date=config["FiggieGame"]["Date"],
        turns=config["FiggieGame"]["Turns"],
        game_variant=config["FiggieGame"]["GameVariant"],
    )
    deck_setup = DeckSetup(
        goal_suit_color=config["DeckSetup"]["GoalSuitColor"],
        goal_suit=config["DeckSetup"]["GoalSuit"],
        distribution=deck_distribution,
    )
    player_states = []
    for pname, p in players.items():
        summary = hand_summary(p.hand)
        player_states.append(PlayerState(
            name=pname,
            hand=[summary],
            money=p.money,
            beliefs=p.beliefs
        ))
    trade_events = [TradeEvent(**te) for te in trade_events_global]
    return GameState(
        metadata=metadata,
        deck_setup=deck_setup,
        players=player_states,
        trades=trade_events,
        current_turn=current_turn,
        pot=pot_amount,
        actual_goal_suit=actual_goal_suit
    )

# --- Trade Mechanism Functions ---

def human_propose_trade(opponent):
    human = players[HUMAN_PLAYER]
    console.print(Panel(f"Your current hand:\n{hand_summary(human.hand)}\nMoney: {human.money}",
                          title="Your Hand", style="bold green"))
    console.print(Panel(f"Your turn to propose a trade with {opponent.name}.", style="bold blue"))
    
    action = console.input("[bold]Do you want to [B]uy or [S]ell? (B/S): [/bold]").strip().lower()
    if action not in ("b", "s"):
        return "Invalid action. Trade cancelled."
    
    suit_char = console.input("[bold]Enter the suit ([S]pades, [C]lubs, [H]earts, [D]iamonds): [/bold]").strip().upper()
    suit_map_input = {"S": "Spades", "C": "Clubs", "H": "Hearts", "D": "Diamonds"}
    if suit_char not in suit_map_input:
        return "Invalid suit. Trade cancelled."
    suit = suit_map_input[suit_char]
    
    try:
        price = float(console.input("[bold]Enter your proposed price: [/bold]").strip())
    except ValueError:
        return "Invalid price. Trade cancelled."
    
    if action == "b" and human.money < price:
        return "You do not have enough money to buy at that price."
    if action == "s" and opponent.money < price:
        return f"{opponent.name} does not have enough money to buy at that price."
    
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
            event = {
                "trade_index": len(trade_events_global) + 1,
                "time": round(turn_number * 10.0, 2),
                "buyer": HUMAN_PLAYER,
                "seller": opponent.name,
                "card": f"{suit_unicode_map[suit]} card",
                "suit": suit,
                "price": round(price, 2)
            }
            trade_events_global.append(event)
            msg = f"Trade Executed: You bought one {suit_unicode_map[suit]} card from {opponent.name} at {price:.2f}."
            console.print(Panel(f"Updated Hand:\n{hand_summary(human.hand)}\nMoney: {human.money}", title="Your Updated Hand", style="bold green"))
            return msg
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
            event = {
                "trade_index": len(trade_events_global) + 1,
                "time": round(turn_number * 10.0, 2),
                "buyer": opponent.name,
                "seller": HUMAN_PLAYER,
                "card": f"{suit_unicode_map[suit]} card",
                "suit": suit,
                "price": round(price, 2)
            }
            trade_events_global.append(event)
            msg = f"Trade Executed: You sold one {suit_unicode_map[suit]} card to {opponent.name} at {price:.2f}."
            console.print(Panel(f"Updated Hand:\n{hand_summary(human.hand)}\nMoney: {human.money}", title="Your Updated Hand", style="bold green"))
            return msg
        else:
            return f"Trade Rejected by {opponent.name} (roll {roll:.2f} vs. accept prob {accept_prob:.2f})."

def bot_propose_trade(opponent):
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
        if opponent.money < price:
            return f"Bot {opponent.name} does not have enough money to buy. Trade cancelled."
        response = console.input(f"[bold]Bot {opponent.name} proposes to BUY your {suit_unicode_map[suit]} card for {price:.2f}. Accept? (y/n): [/bold]").strip().lower()
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
            event = {
                "trade_index": len(trade_events_global) + 1,
                "time": round(turn_number * 10.0, 2),
                "buyer": opponent.name,
                "seller": HUMAN_PLAYER,
                "card": f"{suit_unicode_map[suit]} card",
                "suit": suit,
                "price": round(price, 2)
            }
            trade_events_global.append(event)
            msg = f"Trade Executed: You sold one {suit_unicode_map[suit]} card to {opponent.name} at {price:.2f}."
            # Show updated hand for the human.
            human = players[HUMAN_PLAYER]
            console.print(Panel(f"Updated Hand:\n{hand_summary(human.hand)}\nMoney: {human.money}", title="Your Updated Hand", style="bold green"))
            return msg
        else:
            return f"You declined Bot {opponent.name}'s proposal to buy your {suit_unicode_map[suit]} card."
    else:
        bot_cards = [card for card in opponent.hand]
        if bot_cards:
            card = random.choice(bot_cards)
            suit = unicode_to_name[card[0]]
        else:
            suit = random.choice(["Spades", "Clubs", "Hearts", "Diamonds"])
        bot_ev = expected_value(suit, opponent.beliefs)
        price = bot_ev - random.uniform(0, 2)
        if players[HUMAN_PLAYER].money < price:
            return f"You do not have enough money to buy. Trade cancelled."
        response = console.input(f"[bold]Bot {opponent.name} proposes to SELL you a {suit_unicode_map[suit]} card for {price:.2f}. Buy? (y/n): [/bold]").strip().lower()
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
            event = {
                "trade_index": len(trade_events_global) + 1,
                "time": round(turn_number * 10.0, 2),
                "buyer": HUMAN_PLAYER,
                "seller": opponent.name,
                "card": f"{suit_unicode_map[suit]} card",
                "suit": suit,
                "price": round(price, 2)
            }
            trade_events_global.append(event)
            msg = f"Trade Executed: You bought one {suit_unicode_map[suit]} card from {opponent.name} at {price:.2f}."
            human = players[HUMAN_PLAYER]
            console.print(Panel(f"Updated Hand:\n{hand_summary(human.hand)}\nMoney: {human.money}", title="Your Updated Hand", style="bold green"))
            return msg
        else:
            return f"You declined Bot {opponent.name}'s proposal to sell a {suit_unicode_map[suit]} card."

# --- REPL / Turn Loop Functions ---

def show_status(turn):
    status_table = Table(title=f"Status at Turn {turn}", show_edge=True)
    status_table.add_column("Player", style="bold")
    status_table.add_column("Money", justify="right")
    status_table.add_column("Hand")
    status_table.add_column("Beliefs", style="dim")
    for pname, player in players.items():
        hand_str = hand_summary(player.hand)
        if HIDE_OPPONENTS and pname != HUMAN_PLAYER:
            hand_str = "Hidden"
        status_table.add_row(
            pname,
            str(player.money),
            hand_str,
            str(player.beliefs)
        )
    console.print(status_table)
    game_state = build_game_state(turn)
    console.print(Panel(game_state.json(indent=2), title="Game State (JSON)", style="magenta"))

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
        hand_str = hand_summary(player.hand)
        if HIDE_OPPONENTS and pname != HUMAN_PLAYER:
            hand_str = "Hidden"
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
    parser = argparse.ArgumentParser(description="Turn-Based Figgie Game")
    parser.add_argument("--hide-opponents", action="store_true", help="Hide other players' hands from display")
    args = parser.parse_args()
    HIDE_OPPONENTS = args.hide_opponents

    console.rule("[bold blue]Welcome to Turn-Based Figgie![/bold blue]")
    console.print(Panel("In this game, you (P1) will interact each turn with each bot.\n"
                        "You may propose a trade (buy or sell) or pass—in which case the bot will propose a trade to you.\n"
                        "Your current hand (shown as counts per suit) and money are displayed whenever you are prompted.\n"
                        "Try to secure cards in the secret goal suit!\n"
                        "Type your responses when prompted.", style="bold"))
    human = players[HUMAN_PLAYER]
    console.print(Panel(f"Your initial hand:\n{hand_summary(human.hand)}\nMoney: {human.money}", title="Your Hand", style="bold green"))
    turn_loop()
    console.print(Panel("Thank you for playing Figgie!", style="bold green"))

if __name__ == "__main__":
    main()

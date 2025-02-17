#!/usr/bin/env python3
"""
Turn-Based Interactive Simulation of Figgie with Particle Filter Bayesian Updates
and a Rich-Powered Unicode Interface.

In this version:
  - The game state is represented by Pydantic models.
  - Each player's beliefs about the goal suit are tracked via a particle filter.
  - The human player's hand is summarized as counts per suit (using colors for red/black).
  - Real inventories of cards and cash are tracked so no trader can exceed their limits.
  - Single-character responses are used for suit input.
  - Trade executed messages now use Unicode suit symbols.
  - After each executed trade, your updated hand and money are shown.
  - A command-line option (--hide-opponents) hides opponents' hands.
  - A "Score Panel" is displayed after each turn, and the actual goal suit is revealed before final scoring.

  [NEW] After the human interacts with a bot, that bot attempts to trade with the other bots
        in a similar manner (bot-to-bot trades), and these trades are displayed to the console.

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
    money: float
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
    Return the value of a card of suit `card_suit` if candidate_goal is assumed to be the goal suit.
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
    Compute expected value of a card of suit `card_suit` based on a player's belief distribution.
    """
    ev = 0
    for candidate, prob in beliefs.items():
        ev += prob * valuation_given_candidate(card_suit, candidate)
    return ev

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

# --- Particle Filter for Bayesian Updates ---

class Particle:
    def __init__(self, candidate_goal, weight=1.0):
        self.candidate_goal = candidate_goal
        self.weight = weight

class ParticleFilter:
    def __init__(self, n_particles=100):
        self.n_particles = n_particles
        self.particles = []
        self.initialize_particles()

    def initialize_particles(self):
        # Candidate goal suits: "Spades", "Clubs", "Hearts", "Diamonds"
        candidates = ["Spades", "Clubs", "Hearts", "Diamonds"]
        self.particles = []
        for _ in range(self.n_particles):
            candidate = random.choice(candidates)
            self.particles.append(Particle(candidate, weight=1.0))
        self.normalize_weights()

    def normalize_weights(self):
        total_weight = sum(p.weight for p in self.particles)
        if total_weight > 0:
            for p in self.particles:
                p.weight /= total_weight
        else:
            self.initialize_particles()

    def update(self, card_suit, price, sigma=3.0):
        """
        Update each particle's weight based on an observed trade of a card of suit `card_suit`
        at price `price` using a Gaussian likelihood.
        """
        for p in self.particles:
            expected_val = valuation_given_candidate(card_suit, p.candidate_goal)
            likelihood = math.exp(-((price - expected_val) ** 2) / (2 * sigma ** 2))
            p.weight *= likelihood
        self.normalize_weights()
        self.resample_if_needed()

    def effective_sample_size(self):
        return 1.0 / sum(p.weight ** 2 for p in self.particles)

    def resample_if_needed(self, threshold_ratio=0.5):
        if self.effective_sample_size() < self.n_particles * threshold_ratio:
            self.resample_particles()

    def resample_particles(self):
        weights = [p.weight for p in self.particles]
        new_particles = random.choices(self.particles, weights=weights, k=self.n_particles)
        self.particles = [Particle(p.candidate_goal, weight=1.0) for p in new_particles]
        self.normalize_weights()

    def get_belief_distribution(self):
        belief = {"Spades": 0.0, "Clubs": 0.0, "Hearts": 0.0, "Diamonds": 0.0}
        for p in self.particles:
            belief[p.candidate_goal] += p.weight
        return belief

# --- Player Class ---

class Player:
    def __init__(self, name, hand, money):
        self.name = name
        # Store full card labels (e.g. "♣8") while the UI shows summarized counts.
        self.hand = hand[:]
        self.money = money
        # Use a particle filter to track beliefs about the goal suit.
        self.pf = ParticleFilter(n_particles=100)

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
        beliefs = p.pf.get_belief_distribution()
        player_states.append(PlayerState(
            name=pname,
            hand=[summary],
            money=p.money,
            beliefs=beliefs
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

# ---------------------------------------------------------------------
#     NEW or UPDATED SECTIONS FOR BOT-TO-BOT TRADING
# ---------------------------------------------------------------------

def bot_vs_bot_propose_trade(botA, botB):
    """
    Let botA propose a single trade to botB, chosen randomly as buy or sell.
    There is no user input. The acceptance is determined by logistic acceptance
    probabilities, just like the user logic, but automatically.

    Return a message describing the trade result (success/failure).
    """

    # Decide if botA is "buying" or "selling"
    action = random.choice(["buy", "sell"])

    # If buying, pick a card from botB's hand. If none, fail.
    # If selling, pick a card from botA's hand. If none, fail.
    if action == "buy":
        # Choose a random card from botB's hand
        if not botB.hand:
            return f"{botA.name} tried to buy from {botB.name}, but {botB.name} has no cards."
        card = random.choice(botB.hand)
        suit = unicode_to_name[card[0]]
        # The "fair" price is around the EV from botA's perspective, plus noise
        ev = expected_value(suit, botA.pf.get_belief_distribution())
        price = ev + random.uniform(-2, 2)  # random offset
        if price < 0.5:
            price = 0.5  # Set a minimum price so we don't do weird negative or near-zero trades

        # If botA can't afford price, no trade
        if botA.money < price:
            return f"{botA.name} cannot afford to buy a {suit_unicode_map[suit]} card from {botB.name} at ${price:.2f}."

        # Acceptance: from botB's perspective
        # Using logistic(probFactor * (price - expectedValueOfSuitForBotB))
        botB_ev = expected_value(suit, botB.pf.get_belief_distribution())
        probFactor = 0.5
        acceptProb = logistic(probFactor * (price - botB_ev))
        roll = random.random()

        if roll < acceptProb:
            # Trade executes
            botB.hand.remove(card)
            botA.hand.append(card)
            botA.money -= price
            botB.money += price
            # Particle filter update for all
            for p in players.values():
                p.pf.update(suit, price, sigma=3.0)
            event = {
                "trade_index": len(trade_events_global) + 1,
                "time": round(turn_number * 10.0, 2),
                "buyer": botA.name,
                "seller": botB.name,
                "card": f"{suit_unicode_map[suit]} card",
                "suit": suit,
                "price": round(price, 2)
            }
            trade_events_global.append(event)
            return (f"[bold yellow]Bot-to-Bot Trade Executed[/bold yellow]: "
                    f"{botA.name} bought a {suit_unicode_map[suit]} card from {botB.name} at ${price:.2f}.")
        else:
            return (f"{botA.name} offered ${price:.2f} for {suit_unicode_map[suit]}, "
                    f"but {botB.name} refused (roll={roll:.2f}, acceptProb={acceptProb:.2f}).")

    else:
        # action == "sell"
        if not botA.hand:
            return f"{botA.name} tried to sell to {botB.name}, but {botA.name} has no cards."
        card = random.choice(botA.hand)
        suit = unicode_to_name[card[0]]
        ev = expected_value(suit, botA.pf.get_belief_distribution())
        price = ev + random.uniform(-2, 2)
        if price < 0.5:
            price = 0.5

        # If botB can't afford price, no trade
        if botB.money < price:
            return f"{botB.name} cannot afford to buy a {suit_unicode_map[suit]} card from {botA.name} at ${price:.2f}."

        # Acceptance: from botB's perspective, but in a "buy" sense
        # acceptance prob ~ logistic(probFactor * (botB_ev - price))
        botB_ev = expected_value(suit, botB.pf.get_belief_distribution())
        probFactor = 0.5
        acceptProb = logistic(probFactor * (botB_ev - price))
        roll = random.random()

        if roll < acceptProb:
            # Trade executes
            botA.hand.remove(card)
            botB.hand.append(card)
            botB.money -= price
            botA.money += price
            for p in players.values():
                p.pf.update(suit, price, sigma=3.0)
            event = {
                "trade_index": len(trade_events_global) + 1,
                "time": round(turn_number * 10.0, 2),
                "buyer": botB.name,
                "seller": botA.name,
                "card": f"{suit_unicode_map[suit]} card",
                "suit": suit,
                "price": round(price, 2)
            }
            trade_events_global.append(event)
            return (f"[bold yellow]Bot-to-Bot Trade Executed[/bold yellow]: "
                    f"{botA.name} sold a {suit_unicode_map[suit]} card to {botB.name} at ${price:.2f}.")
        else:
            return (f"{botA.name} wanted ${price:.2f} for {suit_unicode_map[suit]}, "
                    f"but {botB.name} refused (roll={roll:.2f}, acceptProb={acceptProb:.2f}).")


def run_bot_to_bot_trades(active_bot, other_bots):
    """
    After the user has interacted with 'active_bot', let 'active_bot' attempt trades
    with each other bot in the list 'other_bots'. We do just one attempt per other bot.
    Display the result.
    """
    for bot in other_bots:
        if bot.name == active_bot.name:
            continue  # skip self
        msg = bot_vs_bot_propose_trade(active_bot, bot)
        console.print(Text(msg, style="cyan"))


# --- Trade Mechanism Functions (Human <-> Bot) ---

def human_propose_trade(opponent):
    human = players[HUMAN_PLAYER]
    console.print(Panel(
        f"Your current hand:\n{hand_summary(human.hand)}\nMoney: {human.money}",
        title="Your Hand", style="bold green")
    )
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
        bot_ev = expected_value(suit, opponent.pf.get_belief_distribution())
        accept_prob = logistic(beta * (price - bot_ev))
        roll = random.random()
        if roll < accept_prob:
            traded_card = opponent_cards[0]
            opponent.hand.remove(traded_card)
            human.hand.append(traded_card)
            human.money -= price
            opponent.money += price
            # Update beliefs using particle filters for all players.
            for player in players.values():
                player.pf.update(suit, price, sigma=3.0)
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
            console.print(Panel(
                f"Updated Hand:\n{hand_summary(human.hand)}\nMoney: {human.money}",
                title="Your Updated Hand", style="bold green")
            )
            return msg
        else:
            return f"Trade Rejected by {opponent.name} (roll {roll:.2f} vs. accept prob {accept_prob:.2f})."
    else:
        # action == "s"
        human_cards = [card for card in human.hand if unicode_to_name[card[0]] == suit]
        if not human_cards:
            return f"You have no {suit} cards to sell. Trade cannot proceed."
        bot_ev = expected_value(suit, opponent.pf.get_belief_distribution())
        accept_prob = logistic(alpha * (bot_ev - price))
        roll = random.random()
        if roll < accept_prob:
            traded_card = human_cards[0]
            human.hand.remove(traded_card)
            opponent.hand.append(traded_card)
            human.money += price
            opponent.money -= price
            for player in players.values():
                player.pf.update(suit, price, sigma=3.0)
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
            console.print(Panel(
                f"Updated Hand:\n{hand_summary(human.hand)}\nMoney: {human.money}",
                title="Your Updated Hand", style="bold green")
            )
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
        bot_ev = expected_value(suit, opponent.pf.get_belief_distribution())
        price = bot_ev + random.uniform(0, 2)
        if opponent.money < price:
            return f"Bot {opponent.name} does not have enough money to buy. Trade cancelled."
        response = console.input(
            f"[bold]Bot {opponent.name} proposes to BUY your {suit_unicode_map[suit]} card for {price:.2f}. Accept? (y/n): [/bold]"
        ).strip().lower()
        if response == "y":
            human_cards = [card for card in players[HUMAN_PLAYER].hand if unicode_to_name[card[0]] == suit]
            if not human_cards:
                return f"You have no {suit} cards to sell. Trade cancelled."
            traded_card = human_cards[0]
            players[HUMAN_PLAYER].hand.remove(traded_card)
            opponent.hand.append(traded_card)
            players[HUMAN_PLAYER].money += price
            opponent.money -= price
            for player in players.values():
                player.pf.update(suit, price, sigma=3.0)
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
            human = players[HUMAN_PLAYER]
            console.print(Panel(
                f"Updated Hand:\n{hand_summary(human.hand)}\nMoney: {human.money}",
                title="Your Updated Hand", style="bold green")
            )
            return msg
        else:
            return f"You declined Bot {opponent.name}'s proposal to buy your {suit_unicode_map[suit]} card."
    else:
        # action == "sell"
        bot_cards = [card for card in opponent.hand]
        if bot_cards:
            card = random.choice(bot_cards)
            suit = unicode_to_name[card[0]]
        else:
            suit = random.choice(["Spades", "Clubs", "Hearts", "Diamonds"])
        bot_ev = expected_value(suit, opponent.pf.get_belief_distribution())
        price = bot_ev - random.uniform(0, 2)
        if players[HUMAN_PLAYER].money < price:
            return f"You do not have enough money to buy. Trade cancelled."
        response = console.input(
            f"[bold]Bot {opponent.name} proposes to SELL you a {suit_unicode_map[suit]} card for {price:.2f}. Buy? (y/n): [/bold]"
        ).strip().lower()
        if response == "y":
            bot_cards = [card for card in opponent.hand if unicode_to_name[card[0]] == suit]
            if not bot_cards:
                return f"Bot {opponent.name} has no {suit} cards to sell. Trade cancelled."
            traded_card = bot_cards[0]
            opponent.hand.remove(traded_card)
            players[HUMAN_PLAYER].hand.append(traded_card)
            players[HUMAN_PLAYER].money -= price
            opponent.money += price
            for player in players.values():
                player.pf.update(suit, price, sigma=3.0)
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
            console.print(Panel(
                f"Updated Hand:\n{hand_summary(human.hand)}\nMoney: {human.money}",
                title="Your Updated Hand", style="bold green")
            )
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
        beliefs = player.pf.get_belief_distribution()
        status_table.add_row(
            pname,
            f"{player.money:.2f}",
            hand_str,
            str({k: round(v,2) for k,v in beliefs.items()})  # rounding beliefs for readability
        )
    console.print(status_table)
    game_state = build_game_state(turn)
    # Use model_dump_json (Pydantic v2) instead of json(indent=2)
    #console.print(Panel(game_state.model_dump_json(indent=2), title="Game State (JSON)", style="magenta"))

def show_score_panel(turn):
    """
    Display each player's money and how many cards of the actual goal suit they hold.
    Called after each turn to provide a 'scoreboard' snapshot.
    """
    console.rule(f"[bold green]Scores After Turn {turn}[/bold green]")
    score_table = Table(title=f"Scoreboard (Goal Suit = {actual_goal_suit})", show_edge=True)
    score_table.add_column("Player", style="bold")
    score_table.add_column("Money", justify="right")
    score_table.add_column(f"{actual_goal_suit} Cards", justify="right")
    for pname, p in players.items():
        goal_count = sum(1 for c in p.hand if unicode_to_name[c[0]] == actual_goal_suit)
        score_table.add_row(pname, f"{p.money:.2f}", str(goal_count))
    console.print(score_table)

def turn_loop():
    global turn_number
    total_turns = config["FiggieGame"]["Turns"]
    all_bots = [players[p] for p in players if p != HUMAN_PLAYER]

    while turn_number <= total_turns:
        console.rule(f"[bold blue]Turn {turn_number}[/bold blue]")
        
        # For each bot: user interacts with the bot, then that bot interacts with other bots
        for opponent in all_bots:
            console.print(Panel(f"Interaction with {opponent.name}", style="bold magenta"))
            choice = console.input(
                f"[bold]Do you want to [T]rade with {opponent.name} or [P]ass? (T/P): [/bold]"
            ).strip().lower()
            
            if choice.startswith("t"):
                result = human_propose_trade(opponent)
                console.print(Text(result, style="green"))
            else:
                result = bot_propose_trade(opponent)
                console.print(Text(result, style="cyan"))

            # [NEW] Now let "opponent" do bot-to-bot trades with other bots
            other_bots = [b for b in all_bots if b.name != opponent.name]
            run_bot_to_bot_trades(opponent, other_bots)

        # Show the status table + game state JSON
        show_status(turn_number)
        # Show scoreboard after each turn
        show_score_panel(turn_number)

        cont = console.input("[bold green]Proceed to next turn? (y/n): [/bold green]").strip().lower()
        if cont != "y":
            break
        turn_number += 1

    console.rule("[bold green]Game Over[/bold green]")
    
    # Reveal the goal suit explicitly before final scoring
    console.print(Panel(
        f"[bold yellow]The Goal Suit is: [bold red]{actual_goal_suit}[/bold red][/bold yellow]",
        style="bold"
    ))
    
    # Tally up final counts for each player
    goal_counts = {}
    for p in players.values():
        count = sum(1 for card in p.hand if unicode_to_name[card[0]] == actual_goal_suit)
        goal_counts[p.name] = count
    
    # Determine winners
    max_count = max(goal_counts.values()) if goal_counts else 0
    winners = [name for name, count in goal_counts.items() if count == max_count]
    
    # Split pot among winners
    if winners:
        share = pot_amount / len(winners)
        for p in players.values():
            if p.name in winners:
                p.money += share
    
    # Display final results
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
            f"{player.money:.2f}",
            str(goal_counts.get(pname, 0)),
            str({k: round(v,2) for k,v in player.pf.get_belief_distribution().items()})
        )
    console.print(final_table)
    
    console.print(f"[bold blue]Winners: {', '.join(winners) if winners else 'No winners'}[/bold blue]")

def main():
    global HIDE_OPPONENTS
    parser = argparse.ArgumentParser(description="Turn-Based Figgie Game with Particle Filter")
    parser.add_argument("--hide-opponents", action="store_true", help="Hide other players' hands from display")
    args = parser.parse_args()
    HIDE_OPPONENTS = args.hide_opponents

    console.rule("[bold blue]Welcome to Turn-Based Figgie![/bold blue]")
    console.print(Panel(
        "In this game, you (P1) will interact each turn with each bot.\n"
        "You may propose a trade (buy or sell) or pass—in which case the bot will propose a trade to you.\n"
        "[NEW] After each bot interaction with you, that bot will attempt trades with other bots.\n"
        "Your current hand (shown as counts per suit) and money are displayed whenever you are prompted.\n"
        "Try to secure cards in the secret goal suit!\n"
        "Type your responses when prompted.", style="bold")
    )
    human = players[HUMAN_PLAYER]
    console.print(Panel(
        f"Your initial hand:\n{hand_summary(human.hand)}\nMoney: {human.money:.2f}",
        title="Your Hand", style="bold green")
    )
    turn_loop()
    console.print(Panel("Thank you for playing Figgie!", style="bold green"))

if __name__ == "__main__":
    main()

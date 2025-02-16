#!/usr/bin/env python3
"""
Monte Carlo Simulation of a Figgie Round Using PFN (Portable Figgie Notation)
Author: [Your Name]
Date: 2025-02-15
"""

import random
import math
import toml  # pip install toml

# --- Helper functions --- #

def logistic(x):
    """Compute the logistic function."""
    return 1.0 / (1.0 + math.exp(-x))

def get_valuation(suit, goal_suit, goal_color):
    """
    Return a baseline valuation for a card of the given suit.
    In this simple model:
      - Cards in the goal suit: high value (30)
      - Cards in the same color as the goal suit but not the goal: medium value (20)
      - Other cards: low value (10)
    """
    black_suits = ["Spades", "Clubs"]
    red_suits   = ["Hearts", "Diamonds"]
    if suit == goal_suit:
        return 30
    elif suit in black_suits and goal_color == "Black":
        return 20
    elif suit in red_suits and goal_color == "Red":
        return 20
    else:
        return 10

# --- Player class --- #

class Player:
    def __init__(self, name, hand, money):
        self.name = name
        self.hand = hand  # list of card labels (e.g., "S1", "C2", etc.)
        self.money = money

# --- Main simulation --- #

def main():
    # For reproducibility in simulation
    random.seed(42)

    # === PFN configuration (as a TOML–compatible dictionary) === #
    config = {
        "FiggieGame": {
            "Title": "Demo Round",
            "GameID": "G12345",
            "Players": 4,
            "Date": "2025-02-15",
            "GameDuration": 60.0,      # total trading time (seconds)
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
        },
        "Deal": {
            "P1": "S1,S2,S3,C2,C7,H5,H6,H7,D8",
            "P2": "S4,S5,S6,C1,C8,H3,H9,D2",
            "P3": "S9,S10,H1,H2,H4,D3,D4,D5",
            "P4": "C3,C4,C5,C6,H8,H10,D6,D7"
        }
    }

    # === Create players from the initial deal === #
    players = {}
    initial_deal = {}  # for output purposes
    INITIAL_MONEY = 350
    for pname, cards_str in config["Deal"].items():
        cards = [card.strip() for card in cards_str.split(",")]
        initial_deal[pname] = cards.copy()
        players[pname] = Player(pname, cards.copy(), INITIAL_MONEY)

    # === Game parameters === #
    game_duration = config["FiggieGame"]["GameDuration"]
    goal_suit = config["DeckSetup"]["GoalSuit"]         # e.g., "Spades"
    goal_color = config["DeckSetup"]["GoalSuitColor"]     # e.g., "Black"
    pot_amount = 100                                    # fixed pot to award the winner

    # For mapping one–letter card prefixes to full suit names
    suit_map = {"S": "Spades", "C": "Clubs", "H": "Hearts", "D": "Diamonds"}

    # === Event–driven simulation of the trading phase === #
    current_time = 0.0
    trade_events = []   # list to store each trade event (for PFN output)
    trade_index = 1

    # We use an exponential inter–arrival time for trade events (mean = 5 sec)
    while current_time < game_duration:
        dt = random.expovariate(1/5.0)
        current_time += dt
        if current_time >= game_duration:
            break

        # Select a seller: must have at least one card
        seller_candidates = [p for p in players.values() if len(p.hand) > 0]
        if not seller_candidates:
            break
        seller = random.choice(seller_candidates)

        # Choose a random card from the seller's hand
        card = random.choice(seller.hand)
        suit = suit_map[card[0]]  # e.g., "S1" -> "Spades"

        # Select a buyer (different from the seller)
        buyer_candidates = [p for p in players.values() if p.name != seller.name]
        if not buyer_candidates:
            continue
        buyer = random.choice(buyer_candidates)

        # Compute the valuations for the card
        buyer_val = get_valuation(suit, goal_suit, goal_color)
        seller_val = get_valuation(suit, goal_suit, goal_color)

        # Determine a candidate price: average valuation plus some noise
        noise = random.uniform(-2, 2)
        candidate_price = (buyer_val + seller_val) / 2 + noise
        candidate_price = max(1, candidate_price)  # enforce a minimum price of 1

        # Ensure the buyer has enough money to pay
        if buyer.money < candidate_price:
            continue

        # Use logistic functions to compute acceptance probabilities
        alpha = 0.5
        beta  = 0.5
        buyer_prob = logistic(alpha * (buyer_val - candidate_price))
        seller_prob = logistic(beta * (candidate_price - seller_val))

        # Decide whether both parties agree to trade
        if random.random() < buyer_prob and random.random() < seller_prob:
            # Execute the trade:
            seller.hand.remove(card)
            buyer.hand.append(card)
            buyer.money -= candidate_price
            seller.money += candidate_price

            # Record the trade event (with a discrete TradeIndex and timestamp)
            trade_event = {
                "TradeIndex": trade_index,
                "T": round(current_time, 2),
                "Buyer": buyer.name,
                "Seller": seller.name,
                "Suit": suit,
                "Card": card,
                "Price": round(candidate_price, 2)
            }
            trade_events.append(trade_event)
            trade_index += 1

    # === End of Trading Phase: Determine Outcome === #
    # Count goal-suit cards for each player.
    goal_counts = {}
    for p in players.values():
        count = sum(1 for card in p.hand if suit_map[card[0]] == goal_suit)
        goal_counts[p.name] = count

    max_count = max(goal_counts.values())
    winners = [name for name, count in goal_counts.items() if count == max_count]

    # Distribute the pot equally among winners.
    share = pot_amount / len(winners)
    for p in players.values():
        if p.name in winners:
            p.money += share

    # Determine which suit in the deck has 12 cards (as revealed at round end).
    revealed_12_suit = None
    for suit_name, count in config["DeckSetup"]["Distribution"].items():
        if count == 12:
            revealed_12_suit = suit_name
            break

    # Build the Result section.
    result = {
        "Revealed12CardSuit": revealed_12_suit,
        "GoalSuit": goal_suit,
    }
    for pname, player in players.items():
        result[f"{pname}_FinalBank"] = int(round(player.money))
    result["Winners"] = winners

    # === Build the complete PFN output === #
    pfn_output = {
        "FiggieGame": config["FiggieGame"],
        "DeckSetup": config["DeckSetup"],
        "Deal": {p: ",".join(cards) for p, cards in initial_deal.items()},
        "Trades": trade_events,
        "Result": result
    }

    # Convert the PFN dictionary to a TOML–formatted string.
    pfn_toml = toml.dumps(pfn_output)
    print(pfn_toml)

if __name__ == "__main__":
    main()


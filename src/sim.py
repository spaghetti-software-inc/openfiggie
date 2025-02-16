#!/usr/bin/env python3
"""
Monte Carlo Simulation of a Figgie Round Using PFN (Portable Figgie Notation)
with Randomized Initial Deal, Simulation Results, Logging, and a Bayesian
Inference Model for each player.
Author: [Your Name]
Date: 2025-02-15
"""

import random
import math
import toml  # pip install toml
import logging

from rich import print # pip install rich

# --- Logging configuration --- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# --- Helper functions --- #

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
    a player's current belief distribution over the candidate goal suits.
    """
    ev = 0
    for candidate, prob in beliefs.items():
        ev += prob * valuation_given_candidate(card_suit, candidate)
    return ev

def update_beliefs(beliefs, card_suit, price, sigma=3.0):
    """
    Updates the belief distribution based on an observed trade of a card of suit
    `card_suit` transacted at price `price`. A Gaussian likelihood function is used:
    
      L(candidate) = exp( -((price - v(candidate))^2 / (2 * sigma^2)) )
    
    where v(candidate) = valuation_given_candidate(card_suit, candidate).
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
        # Avoid division by zero; revert to uniform beliefs.
        return {c: 1.0 / len(beliefs) for c in beliefs}

# --- Player class --- #

class Player:
    def __init__(self, name, hand, money):
        self.name = name
        self.hand = hand  # list of card labels (e.g., "S1", "C2", etc.)
        self.money = money
        # Each player starts with a uniform belief over the four possible goal suits.
        self.beliefs = {"Spades": 0.25, "Clubs": 0.25, "Hearts": 0.25, "Diamonds": 0.25}

# --- Main simulation --- #

def main():
    # For reproducibility in simulation
    # random.seed(42)

    # === PFN configuration (as a TOML–compatible dictionary) === #
    config = {
        "FiggieGame": {
            "Title": "Demo Round with Bayesian Inference",
            "GameID": "G12345",
            "Players": 4,
            "Date": "2025-02-15",
            "GameDuration": 120.0,      # total trading time (seconds)
            "GameVariant": "Standard"
        },
        "DeckSetup": {
            "GoalSuitColor": "Black",  # used in deck design but players don't know true goal suit
            "GoalSuit": "Spades",       # actual goal suit (for simulation outcome)
            "Distribution": {
                "Spades": 10,
                "Clubs": 12,
                "Hearts": 10,
                "Diamonds": 8
            }
        }
    }

    # === Randomize the initial deal === #
    deck_distribution = config["DeckSetup"]["Distribution"]
    suit_letter_map = {"Spades": "S", "Clubs": "C", "Hearts": "H", "Diamonds": "D"}
    
    # Construct the deck based on the distribution
    deck = []
    for suit, count in deck_distribution.items():
        letter = suit_letter_map[suit]
        for i in range(1, count + 1):
            deck.append(letter + str(i))
    
    # Shuffle the deck
    random.shuffle(deck)
    
    # Determine the number of players and deal cards using round-robin distribution.
    num_players = config["FiggieGame"]["Players"]
    initial_deal = {}
    for i in range(num_players):
        initial_deal[f"P{i+1}"] = []
    for idx, card in enumerate(deck):
        player_key = f"P{(idx % num_players) + 1}"
        initial_deal[player_key].append(card)
    
    # Convert each player's hand into a comma-separated string for PFN output.
    deal_str = {p: ",".join(cards) for p, cards in initial_deal.items()}

    # === Create players from the initial deal === #
    players = {}
    INITIAL_MONEY = 350
    for pname, cards in initial_deal.items():
        players[pname] = Player(pname, cards.copy(), INITIAL_MONEY)

    # === Game parameters === #
    game_duration = config["FiggieGame"]["GameDuration"]
    # For simulation outcome, we know the actual goal suit.
    actual_goal_suit = config["DeckSetup"]["GoalSuit"]
    pot_amount = 100  # fixed pot to award the winner

    # Mapping for one-letter prefixes to full suit names.
    suit_map = {"S": "Spades", "C": "Clubs", "H": "Hearts", "D": "Diamonds"}

    # === Event-driven simulation of the trading phase === #
    current_time = 0.0
    trade_events = []   # list to store each trade event (for PFN output)
    trade_index = 1

    # We use an exponential inter-arrival time for trade events (mean = 5 sec)
    while current_time < game_duration:
        dt = random.expovariate(1 / 5.0)
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

        # Each player computes expected value for the card based on their current beliefs.
        buyer_val = expected_value(suit, buyer.beliefs)
        seller_val = expected_value(suit, seller.beliefs)

        # Determine a candidate price: average of buyer and seller valuations plus noise.
        noise = random.uniform(-2, 2)
        candidate_price = (buyer_val + seller_val) / 2 + noise
        candidate_price = max(1, candidate_price)  # enforce a minimum price of 1

        # Ensure the buyer has enough money to pay.
        if buyer.money < candidate_price:
            logging.info(
                f"Trade candidate skipped at T={round(current_time,2)}: Buyer {buyer.name} lacks funds (needs {candidate_price:.2f}, has {buyer.money:.2f})."
            )
            continue

        # Compute acceptance probabilities using logistic functions.
        # (These probabilities are based on the player's own expected valuation.)
        alpha = 0.5
        beta  = 0.5
        buyer_prob = logistic(alpha * (buyer_val - candidate_price))
        seller_prob = logistic(beta * (candidate_price - seller_val))

        # Generate random rolls for each player's decision.
        buyer_roll = random.random()
        seller_roll = random.random()

        logging.info(
            f"Trade Candidate at T={round(current_time,2)}: Seller {seller.name} offers card {card} ({suit}); "
            f"Buyer {buyer.name} considered. Candidate price = {candidate_price:.2f}; "
            f"Buyer EV = {buyer_val:.2f}, Seller EV = {seller_val:.2f}; "
            f"Buyer prob = {buyer_prob:.2f} (roll = {buyer_roll:.2f}), Seller prob = {seller_prob:.2f} (roll = {seller_roll:.2f})."
        )

        # Decide whether both parties agree to trade.
        if buyer_roll < buyer_prob and seller_roll < seller_prob:
            # Execute the trade:
            seller.hand.remove(card)
            buyer.hand.append(card)
            buyer.money -= candidate_price
            seller.money += candidate_price

            # Record the trade event.
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
            logging.info(
                f"Trade Executed: {buyer.name} buys {card} ({suit}) from {seller.name} at {candidate_price:.2f}."
            )
            trade_index += 1

            # --- Bayesian Update for All Players ---
            # After an executed trade, every player updates their belief based on the observed trade.
            for p in players.values():
                old_beliefs = p.beliefs.copy()
                p.beliefs = update_beliefs(p.beliefs, suit, candidate_price, sigma=3.0)
                logging.info(
                    f"Player {p.name} updated beliefs based on trade: {old_beliefs} -> {p.beliefs}"
                )
        else:
            logging.info(
                f"Trade Rejected at T={round(current_time,2)}: Buyer {buyer.name} (roll {buyer_roll:.2f} vs. {buyer_prob:.2f}) "
                f"or Seller {seller.name} (roll {seller_roll:.2f} vs. {seller_prob:.2f}) did not accept."
            )

    # === End of Trading Phase: Determine Outcome === #
    # Count goal-suit cards for each player.
    goal_counts = {}
    for p in players.values():
        count = sum(1 for card in p.hand if suit_map[card[0]] == actual_goal_suit)
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
    for suit_name, count in deck_distribution.items():
        if count == 12:
            revealed_12_suit = suit_name
            break

    # Build the Result section, now with extra simulation results.
    result = {
        "Revealed12CardSuit": revealed_12_suit,
        "GoalSuit": actual_goal_suit,
        "TradeCount": trade_index - 1,
        "FinalTradingTime": round(current_time, 2),
        "GoalCounts": goal_counts,  # goal-suit card count per player
    }
    for pname, player in players.items():
        result[f"{pname}_FinalBank"] = int(round(player.money))
    result["Winners"] = winners

    # === Build the complete PFN output === #
    pfn_output = {
        "FiggieGame": config["FiggieGame"],
        "DeckSetup": config["DeckSetup"],
        "Deal": deal_str,
        "Trades": trade_events,
        "Result": result
    }

    # Convert the PFN dictionary to a TOML–formatted string and print.
    pfn_toml = toml.dumps(pfn_output)
    print(pfn_toml)

if __name__ == "__main__":
    main()

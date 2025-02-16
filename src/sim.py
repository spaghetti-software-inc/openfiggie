#!/usr/bin/env python3
"""
Monte Carlo Simulation of a Figgie Round Using PFN (Portable Figgie Notation)
with Randomized Initial Deal, Simulation Results, Logging, Bayesian Inference,
and Unicode symbols for suits.
Author: [Your Name]
Date: 2025-02-15
"""

import random
import math
import toml  # pip install toml
import logging

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
    `card_suit` transacted at price `price` using a Gaussian likelihood:
    
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
        self.hand = hand  # list of card labels (e.g., "♠1", "♣2", etc.)
        self.money = money
        # Start with a uniform belief over the four candidate goal suits.
        self.beliefs = {"Spades": 0.25, "Clubs": 0.25, "Hearts": 0.25, "Diamonds": 0.25}

# --- Main simulation --- #

def main():
    # For reproducibility in simulation
    random.seed(42)

    # === PFN configuration (as a TOML–compatible dictionary) === #
    config = {
        "FiggieGame": {
            "Title": "Demo Round with Bayesian Inference & Unicode Suits",
            "GameID": "G12345",
            "Players": 4,
            "Date": "2025-02-15",
            "GameDuration": 60.0,      # total trading time (seconds)
            "GameVariant": "Standard"
        },
        "DeckSetup": {
            "GoalSuitColor": "Black",  # used for deck design (hidden to players)
            "GoalSuit": "Spades",       # actual goal suit (for simulation outcome)
            "Distribution": {
                "Spades": 10,
                "Clubs": 12,
                "Hearts": 10,
                "Diamonds": 8
            }
        }
    }

    # === Randomize the initial deal using Unicode symbols === #
    # Use Unicode symbols for suits.
    suit_unicode_map = {"Spades": "♠", "Clubs": "♣", "Hearts": "♥", "Diamonds": "♦"}
    
    deck_distribution = config["DeckSetup"]["Distribution"]
    deck = []
    for suit, count in deck_distribution.items():
        symbol = suit_unicode_map[suit]
        for i in range(1, count + 1):
            deck.append(symbol + str(i))
    
    # Shuffle the deck
    random.shuffle(deck)
    
    # Deal cards to players in round-robin fashion.
    num_players = config["FiggieGame"]["Players"]
    initial_deal = {}
    for i in range(num_players):
        initial_deal[f"P{i+1}"] = []
    for idx, card in enumerate(deck):
        player_key = f"P{(idx % num_players) + 1}"
        initial_deal[player_key].append(card)
    
    # Create a PFN-friendly string for the initial deal.
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

    # Map from Unicode suit symbol to full suit name.
    unicode_to_name = {"♠": "Spades", "♣": "Clubs", "♥": "Hearts", "♦": "Diamonds"}

    # === Event-driven simulation of the trading phase === #
    current_time = 0.0
    trade_events = []   # list to store trade events (for PFN output)
    trade_index = 1

    # Use an exponential inter-arrival time for trade events (mean = 5 sec)
    while current_time < game_duration:
        dt = random.expovariate(1 / 5.0)
        current_time += dt
        if current_time >= game_duration:
            break

        # Select a seller with at least one card.
        seller_candidates = [p for p in players.values() if len(p.hand) > 0]
        if not seller_candidates:
            break
        seller = random.choice(seller_candidates)

        # Choose a random card from the seller's hand.
        card = random.choice(seller.hand)
        # Extract the Unicode symbol from the card label (e.g., "♠" from "♠3").
        symbol = card[0]
        suit = unicode_to_name[symbol]

        # Select a buyer (different from the seller).
        buyer_candidates = [p for p in players.values() if p.name != seller.name]
        if not buyer_candidates:
            continue
        buyer = random.choice(buyer_candidates)

        # Each player computes expected value for the card based on their beliefs.
        buyer_val = expected_value(suit, buyer.beliefs)
        seller_val = expected_value(suit, seller.beliefs)

        # Determine a candidate price (average of EVs plus noise).
        noise = random.uniform(-2, 2)
        candidate_price = (buyer_val + seller_val) / 2 + noise
        candidate_price = max(1, candidate_price)  # minimum price = 1

        if buyer.money < candidate_price:
            logging.info(
                f"Trade candidate skipped at T={round(current_time,2)}: Buyer {buyer.name} lacks funds (needs {candidate_price:.2f}, has {buyer.money:.2f})."
            )
            continue

        # Compute acceptance probabilities.
        alpha = 0.5
        beta  = 0.5
        buyer_prob = logistic(alpha * (buyer_val - candidate_price))
        seller_prob = logistic(beta * (candidate_price - seller_val))

        buyer_roll = random.random()
        seller_roll = random.random()

        logging.info(
            f"Trade Candidate at T={round(current_time,2)}: Seller {seller.name} offers card {card} ({suit}); "
            f"Buyer {buyer.name} considered. Candidate price = {candidate_price:.2f}; "
            f"Buyer EV = {buyer_val:.2f}, Seller EV = {seller_val:.2f}; "
            f"Buyer prob = {buyer_prob:.2f} (roll = {buyer_roll:.2f}), Seller prob = {seller_prob:.2f} (roll = {seller_roll:.2f})."
        )

        if buyer_roll < buyer_prob and seller_roll < seller_prob:
            seller.hand.remove(card)
            buyer.hand.append(card)
            buyer.money -= candidate_price
            seller.money += candidate_price

            trade_event = {
                "TradeIndex": trade_index,
                "T": round(current_time, 2),
                "Buyer": buyer.name,
                "Seller": seller.name,
                "Suit": suit,       # full suit name (e.g., "Spades")
                "Card": card,       # card label with Unicode symbol (e.g., "♠3")
                "Price": round(candidate_price, 2)
            }
            trade_events.append(trade_event)
            logging.info(
                f"Trade Executed: {buyer.name} buys {card} ({suit}) from {seller.name} at {candidate_price:.2f}."
            )
            trade_index += 1

            # Bayesian update for every player.
            for p in players.values():
                old_beliefs = p.beliefs.copy()
                p.beliefs = update_beliefs(p.beliefs, suit, candidate_price, sigma=3.0)
                logging.info(
                    f"Player {p.name} updated beliefs: {old_beliefs} -> {p.beliefs}"
                )
        else:
            logging.info(
                f"Trade Rejected at T={round(current_time,2)}: Buyer {buyer.name} (roll {buyer_roll:.2f} vs. {buyer_prob:.2f}) "
                f"or Seller {seller.name} (roll {seller_roll:.2f} vs. {seller_prob:.2f}) did not accept."
            )

    # === End of Trading Phase: Determine Outcome === #
    goal_counts = {}
    for p in players.values():
        # Count only cards whose Unicode symbol (first character) maps to the actual goal suit.
        count = sum(1 for card in p.hand if unicode_to_name[card[0]] == actual_goal_suit)
        goal_counts[p.name] = count

    max_count = max(goal_counts.values())
    winners = [name for name, count in goal_counts.items() if count == max_count]

    share = pot_amount / len(winners)
    for p in players.values():
        if p.name in winners:
            p.money += share

    # Determine the suit with 12 cards in the deck.
    revealed_12_suit = None
    for suit_name, count in deck_distribution.items():
        if count == 12:
            revealed_12_suit = suit_name
            break

    result = {
        "Revealed12CardSuit": revealed_12_suit,
        "GoalSuit": actual_goal_suit,
        "TradeCount": trade_index - 1,
        "FinalTradingTime": round(current_time, 2),
        "GoalCounts": goal_counts,
    }
    for pname, player in players.items():
        result[f"{pname}_FinalBank"] = int(round(player.money))
    result["Winners"] = winners

    pfn_output = {
        "FiggieGame": config["FiggieGame"],
        "DeckSetup": config["DeckSetup"],
        "Deal": deal_str,
        "Trades": trade_events,
        "Result": result
    }

    pfn_toml = toml.dumps(pfn_output)
    print(pfn_toml)

if __name__ == "__main__":
    main()

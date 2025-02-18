# OpenFiggie
An analysis of the card game Figgie (https://www.figgie.com) using console-based game simulation and analysis.



https://github.com/user-attachments/assets/d43806e3-a57d-4834-bbdc-b39683fdd6e8



## Future Work
- Advanced trading User Interface
- Real-time gameplay 
- Scaling up to large number of players / bots

## Introduction
Figgie is a card game developed by Jane Street in 2013 to emulate the dynamics of open-outcry commodities trading. The game accommodates 4 or 5 players, each starting with $350 in chips. Players trade cards representing four suits—spades (♠), clubs (♣), hearts (♥), and diamonds (♦)—with the objective of amassing the most wealth over multiple rounds. 

**Deck Composition:**

- **Total Cards:** 40

- **Suits Distribution:**
  - One suit with 12 cards
  - One suit with 8 cards
  - Two suits with 10 cards each

The specific distribution of suits is unknown to players at the start of each round.

**Objective:**

The primary goal is to identify and collect cards from the "goal suit," which is the suit that matches the color (red or black) of the 12-card suit. The goal suit contains either 8 or 10 cards and is the only suit with value at the end of the round. 

**Game Setup:**

**Ante:** Each player contributes an equal share to form a communal pot of $200.

**Dealing:** All 40 cards are dealt evenly among the players.

**Trading Phase:**

- **Duration:** 4 minutes

- **Mechanics:** Players engage in open trading, placing bids and offers to buy or sell individual cards. Trading is unstructured, allowing for dynamic negotiation and strategy. 

**Round Conclusion:**

**Reveal:** The goal suit is disclosed.

**Payouts:**
   - **Bonus:** Players receive a $10 bonus from the pot for each goal suit card they possess.
   - **Majority Holder:** The player(s) holding the most goal suit cards claim the remaining pot. If multiple players tie for the majority, they split the remainder evenly. 

**Strategic Considerations:**

- **Information Deduction:** Observing trading behaviors and price movements can provide insights into the goal suit's identity.

- **Risk Management:** Balancing the acquisition of potential goal suit cards against the risk of overpaying is crucial.

- **Market Making:** Actively participating in trades, even without perfect information, can lead to profits through favorable deals.

**Learning Mode:**

For newcomers, Figgie offers a learning mode that extends the trading period to 20 minutes. This mode can display opponents' hands to facilitate understanding of game mechanics. 

For a visual explanation and tutorial on how to play Figgie, you might find the following video helpful:

[How to Play Figgie | Jane Street Card Game Tutorial](https://www.youtube.com/watch?v=s4VN36VYhog) 

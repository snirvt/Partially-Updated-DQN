# Partially-Updated-DQN
Project for creating DQN with a new novel regularization.


# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# https://discuss.pytorch.org/t/update-only-sub-elements-of-weights/29101

1) Select the problems we want to compare on.

2) Implement the vanilla DQN.

3) Find a way to partially update the network.

4) implement the below:

•	Picking completely random subset.
•	Picking the subset based on the gradients magnitude:
1.	Being completely greedy. (per layer because of the vanishing gradient problem).
2.	Choosing the weights randomly but give more chance for gradients with bigger magnitude.
3.	Epsilon greedy.
•	Find scheduler which combines the above methods.

5) Create graphs.












## Project Overview

The aim of this project is to solve a problem of a robotic hand trying to learn how to write alphabetical characters and/or other shapes. We created a physical 2 DOF robotic system that represents a human arm that only operates in x-axis and y-axis. The methodology we used is an adaptation of reinforcement learning and specifically the Q-Learning algorithm as an easier to understand, use and have the desired results with less time consuming and complexity. The robot writing though is not just a physical approach, it has also impact in humanity. Robot writing is a task that could be used for helping purposes in human with disabilities or/and in early education. Further studies about reinforcement learning in robotic systems could also bring even better impact in humanity.

## Reinforcement Learning Methods
- **Q-Learning**: Model-free reinforcement learning, it can also be viewed as a method of asynchronous dynamic programming. It provides agents with the capability of learning to act optimally in Markovian domains by experiencing the consequences of actions, without requiring them to build maps of the domains.

```
Algorithm 1: Q-learning

Input: episodes, α,γ
Output: Q
Initialize : set Q(s,a) arbitrarily, for each s in S and a in A;
set Q(terminalstate,∙)=0;
repeat for each episode
  Initialize S ;
  repeat for each step of episode
    Choose A from S using ε-greedy policy;
    Take action A, observe R,S′ ;
    Q(sₜ, aₜ) ← Q(sₜ, aₜ) + α * (rₜ₊₁ + γ * max(Q(sₜ₊₁, a')) - Q(sₜ, aₜ))
    S⟵S′ ;
  until S is a terminalstate ;
until all episodes are visited ;
```

- **Deep Q-Learning**: An extension of the Q-learning algorithm that combines reinforcement learning with deep neural networks. It was introduced by DeepMind in 2013 and has been widely used for solving complex problems in RL. The key idea behind DQN is to use a deep neural network to approximate the Q-values instead of a traditional Q-table. This allows DQN to handle high-dimensional and continuous state spaces, making it suitable for tasks such as image recognition or game playing.

```
Algorithm 2: Deep Q-learning

Input: episodes, α,γ
Output: Q
Initialize 𝑄−𝑁𝑒𝑡𝑤𝑜𝑟𝑘, 𝑇𝑎𝑟𝑔𝑒𝑡−𝑁𝑒𝑡𝑤𝑜𝑟𝑘, Memory D and ε
repeat for each episode
  Initialize S ;
  repeat for each step of episode
    Choose A from S using ε-greedy policy;
    Take action A, observe R,S′ ;
    Store this experience (𝐴,𝑅,𝑆,𝑆′) in Memory D ;
    Choose a random minibatch 𝐵 from experiences in D ;
    Train 𝑄−𝑁𝑒𝑡𝑤𝑜𝑟𝑘
      Choose random samples from stored experiences
      Update 𝑄−𝑁𝑒𝑡𝑤𝑜𝑟𝑘 based on experiences
    Update 𝑇𝑎𝑟𝑔𝑒𝑡−𝑁𝑒𝑡𝑤𝑜𝑟𝑘 with knowledge from 𝑄−𝑁𝑒𝑡𝑤𝑜𝑟𝑘
    S⟵S′ ;
  until S is a terminalstate ;
until all episodes are visited ;
```

## Diagram
<div align="center">
  <img width="600" alt="11" src="https://github.com/alexkalergis/Robot-writing-via-Reinforcement-Learning/assets/105602973/2380c318-3388-4547-af58-88691ffaef90">
</div>


## Methodology
In our case we want to simulate the physical part of a 2 DOF robotic system. It tries to simulate a human arm but without the accessibility in the z-axis. The architecture of this system is shown in the image below.
<div align="center">
  <img width="600" alt="Robot Architecture 2" src="https://github.com/alexkalergis/Robot-writing-via-Reinforcement-Learning/assets/105602973/2ce33db4-a451-4543-b69d-d2cd1c3dd6a9">
</div>

- **Action Space:** The most tricky and challenging part working with value-based RL algorithms was the discretization of action space. The bigger and finer discretization means difficulty in learning process and bigger time and computational cost. The action space we created to solve this problem is by making it more suitable for our shape in each case and reduce the limits and then with trial and error we tried to fine the best discretization step.

- **State Space:** The most tricky and challenging part working with value-based RL algorithms was the discretization of action space. The bigger and finer discretization means difficulty in learning process and bigger time and computational cost. The action space we created to solve this problem is by making it more suitable for our shape in each case and reduce the limits and then with trial and error we tried to fine the best discretization step.
  <div align="center">
  <img width="400" alt="image" src="https://github.com/alexkalergis/Robot-writing-via-Reinforcement-Learning/assets/105602973/69eb4664-9f43-48a7-9f64-9a2918198cad">
  </div>


- **Reward Function:** The learning process in our problem is getting closer to the desired shape. So we created a reward function that computes the Euclidean distance between the pen position and the desired target point. With the help of Q-Learning that works with steps in the second for loop in its algorithm we managed to break down each shape into number of points and so discretize the shape. So, in every step the rewards function computes the distance between the target point and the pen position and returns as feedback the 1/(1 + 𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒). If the 𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒 > 0.5 reward is subtracted by 10 and if 𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒 < 0.1 reward is increased by 10. Another reward functions could calculate the similarity of the shapes in every step using advanced algorithms.


## Results
<div align="center">
  <video width="200" src="https://github.com/alexkalergis/Robot-writing-via-Reinforcement-Learning/assets/105602973/7fc7ad07-036c-4408-96a4-2ae66f0fa316" />
</div>

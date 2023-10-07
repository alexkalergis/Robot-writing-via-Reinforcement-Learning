#==========VIRTUAL 2DOF ROBOT WRTITING VIA DEEP Q-LEARNING==========
#This is an example for learning the sinewave shape with Deep Q-Learning



import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import itertools
from matplotlib.animation import FuncAnimation

class RobotEnvironment:
    def __init__(self):

        # Initialise the state, action, step parameters
        self.points = 15            # Number of x,y points of the characters
        self.steps = self.points    # Number of discrete steps
        self.num_actions = 24**2    # Number of actions
        self.state_dim = 2          # State Dimension for continuous state space

        # Define the robot model parameters
        self.L1 = 5    # Link 1 length
        self.L2 = 5   # Link 2 length

        # Define the desired trajectory
        self.x, self.y = self.LETTER_C()                            # Choosing the desired trajectory
        self.target_trajectory = np.column_stack((self.x, self.y))  # The trajectory that represents the desired character
        self.current_step = 1                                       # Variable that represents the step for Q-Learning algorithm

        # Initialize the pen position
        self.pen_position_x = self.x[0]
        self.pen_position_y = self.y[0]
        self.pen_position = np.array([self.pen_position_x, self.pen_position_y])

        # Initialize the joint angles
        self.joint1_angle, self.joint2_angle = self.inverse_kinematics(self.pen_position_x, self.pen_position_y, self.L1, self.L2)
        self.joint1_angle_track = []
        self.joint2_angle_track = []

    def reset(self):
        # Reset the environment and return the initial state
        # Reset and track the joint angles
        self.joint1_angle, self.joint2_angle = self.inverse_kinematics(self.x[0], self.y[0], self.L1, self.L2)
        self.joint1_angle_track = []
        self.joint2_angle_track = []
        self.joint1_angle_track.append(self.joint1_angle)
        self.joint2_angle_track.append(self.joint2_angle)
        self.joints_track = np.column_stack((self.joint1_angle_track, self.joint2_angle_track))

        # Reset and track the pen position
        self.pen_position = self.forward_kinematics(self.joint1_angle, self.joint2_angle)
        self.pen_trajectory_x = []
        self.pen_trajectory_y = []
        self.pen_trajectory_x.append(self.pen_position[0])
        self.pen_trajectory_y.append((self.pen_position[1]))
        self.pen_trajectory = np.column_stack((self.pen_trajectory_x, self.pen_trajectory_y))

        # Reset the step of the current state
        self.current_step = 1
        return self.state()

    def step(self, action):
        # Execute the action in the environment and return the next state, reward, and done flag
        delta_x, delta_y = self.action(action)
        self.update_states(delta_x, delta_y)
        next_state = self.state()
        reward = self.reward()
        done = self.current_step >= len(self.target_trajectory) - 1
        self.current_step += 1
        return next_state, reward, done

    def state(self):
        # Return the state
        return np.array([self.joint1_angle, self.joint2_angle])

    def reward(self):
        # Return reward
        target_point = self.target_trajectory[self.current_step]
        pen_point = self.pen_position
        distance = np.linalg.norm(pen_point - target_point)
        reward = 1/(1+10*distance)
        if distance < 0.1: reward += 10
        if distance > 0.8: reward -= 10
        return reward

    def action(self, action):
        # Compute the action space. The discretised action space.
        joint_angle1_values = np.linspace(-0.06, -0.03, 10)
        joint_angle1 = [
            - 0.04045,
            - 0.03669,
            - 0.03919,
            - 0.04637,
            - 0.05271,
            - 0.05358,
            - 0.04869,
            - 0.04166,
            - 0.03763,
            - 0.04002,
            - 0.04752,
            - 0.05417,
            - 0.05383,
            - 0.04576
        ]
        joint_angle1_values = np.append(joint_angle1_values, joint_angle1)
        joint_angle2_values = np.linspace(-0.057, 0.15, 10)
        joint_angle2 = [
            -0.03694,
            -0.02621,
            0.00635,
            0.05207,
            0.09576,
            0.12468,
            0.13396,
            0.12526,
            0.10324,
            0.07283,
            0.03790,
            0.00152,
            - 0.03222,
            - 0.05700,
        ]
        joint_angle2_values = np.append(joint_angle2_values,joint_angle2)
        action_space = list(itertools.product(joint_angle1_values, joint_angle2_values))
        delta_x, delta_y = action_space[action]
        return delta_x, delta_y

    # ================================ROBOT_SYSTEM====================================
    def update_states(self, delta_x, delta_y):
        # Update the robot's states based on the pen updates
        self.joint1_angle += delta_x
        self.joint2_angle += delta_y
        self.joint1_angle_track.append(self.joint1_angle)
        self.joint2_angle_track.append(self.joint2_angle)
        self.joints_track = np.column_stack((self.joint1_angle_track, self.joint2_angle_track))

        # Update the robot's states based on the pen updates
        self.pen_position = self.forward_kinematics(self.joint1_angle, self.joint2_angle)
        self.pen_position_x = self.pen_position[0]
        self.pen_position_y = self.pen_position[1]
        self.pen_trajectory_x.append(self.pen_position_x)
        self.pen_trajectory_y.append(self.pen_position_y)
        self.pen_trajectory = np.column_stack( (self.pen_trajectory_x, self.pen_trajectory_y) )
    # =================================================================================

    def forward_kinematics(self, joint1_angle, joint2_angle):
        # Compute the end effector position given the joint angles
        x = self.L1 * np.cos(joint1_angle) + self.L2 * np.cos(joint1_angle + joint2_angle)
        y = self.L1 * np.sin(joint1_angle) + self.L2 * np.sin(joint1_angle + joint2_angle)
        return np.array([x, y])

    def inverse_kinematics(self, x, y, l1, l2):
        # Compute the joint angles given the end effector position
        theta2 = (x ** 2 + y ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
        q2 = np.arccos(theta2)
        q1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
        return q1, q2

    def generate_sine_wave(self):
        num_points = self.points
        x = np.linspace(-2*np.pi, -np.pi, num_points)  # Create evenly spaced x values
        y = np.sin(2*x) + 5                     # Compute the sine wave
        return x, y

    def plot_robot_links_bullet(self, joint1_angle, joint2_angle, L1, L2):
        x0, y0 = (0, 0)
        x1 = L1 * np.cos(joint1_angle)
        y1 = L1 * np.sin(joint1_angle)
        x2 = x1 + L2 * np.cos(joint1_angle + joint2_angle)
        y2 = y1 + L2 * np.sin(joint1_angle + joint2_angle)
        plt.plot([x0, x1], [y0, y1], 'b-', lw=0.5)  # Link 1
        plt.plot([x1, x2], [y1, y2], 'b-', lw=0.5)  # Link 2
        plt.plot(x0, y0, 'ko', markersize=5)  # Base joint
        plt.plot(x1, y1, 'ko', markersize=5)  # Link 1 joint
        plt.plot(x2, y2, 'ro', markersize=5)  # End effector (pen tip)

    def plot_robot_links_line(self, joint1_angle, joint2_angle, L1, L2):
        x0, y0 = (0, 0)
        x1 = L1 * np.cos(joint1_angle)
        y1 = L1 * np.sin(joint1_angle)
        x2 = x1 + L2 * np.cos(joint1_angle + joint2_angle)
        y2 = y1 + L2 * np.sin(joint1_angle + joint2_angle)
        plt.plot([x0, x1], [y0, y1], 'b-', lw=0.5)  # Link 1
        plt.plot([x1, x2], [y1, y2], 'b-', lw=0.5)  # Link 2
        plt.plot(x0, y0, 'ko', markersize=5)  # Base joint
        plt.plot(x1, y1, 'ko', markersize=5)  # Link 1 joint
        plt.plot(x2, y2, 'r', markersize=5)  # End effector (pen tip)

    def create_gif_for_episode_bullet(self, episode, env):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim([-7, 3])  # Modify as necessary
        ax.set_ylim([-0.5, 8])
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_title("Episode " + str(episode) + " Robot Trajectory")
        line, = ax.plot([], [], 'r', label="Robot Writing")
        targets, = ax.plot(env.target_trajectory[:, 0], env.target_trajectory[:, 1], 'go', label="Target")
        ax.legend()

        def init():
            line.set_data([], [])
            return line,

        def update(i):
            env.plot_robot_links_bullet(env.joint1_angle_track[i], env.joint2_angle_track[i], env.L1, env.L2)
            line.set_data(env.pen_trajectory_x[:i + 1], env.pen_trajectory_y[:i + 1])
            return line,

        ani = FuncAnimation(fig, update, frames=len(env.pen_trajectory_x), init_func=init, blit=True, repeat=False)
        # Save the GIF with episode number in the filename
        file_name = f'BULLET/Robot Trajectory Episode_BULLET_{episode}.gif'
        ani.save(file_name, writer='imagemagick', fps=10)
        plt.close(fig)

    def create_gif_for_episode_line(self, episode, env):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim([-7, 3])  # Modify as necessary
        ax.set_ylim([-0.5, 8])
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_title("Episode " + str(episode) + " Robot Trajectory")
        line, = ax.plot([], [], 'r', label="Robot Writing")
        targets, = ax.plot(env.target_trajectory[:, 0], env.target_trajectory[:, 1], 'g', label="Target")
        ax.legend()

        def init():
            line.set_data([], [])
            return line,

        def update(i):
            env.plot_robot_links_line(env.joint1_angle_track[i], env.joint2_angle_track[i], env.L1, env.L2)
            line.set_data(env.pen_trajectory_x[:i + 1], env.pen_trajectory_y[:i + 1])
            return line,

        ani = FuncAnimation(fig, update, frames=len(env.pen_trajectory_x), init_func=init, blit=True, repeat=False)
        # Save the GIF with episode number in the filename
        file_name = f'LINE/Robot Trajectory Episode_LINE_{episode}.gif'
        ani.save(file_name, writer='imagemagick', fps=10)
        plt.close(fig)


# Define the Neural Network model for our DQNetwork agent
def build_dqn_model(state_dim, action_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(state_dim,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(action_dim, activation='linear')])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.001))
    return model

# Define the DQN Agent
class DQNAgent:
    def __init__(self, environment, state_dim, num_actions, learning_rate, discount_factor, epsilon):
        self.environment = environment
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_network = build_dqn_model(state_dim, num_actions)
        self.target_network = build_dqn_model(state_dim, num_actions)
        self.target_network.set_weights(self.q_network.get_weights())
        self.replay_buffer = []  # Use a replay buffer to store experiences

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values)

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # Calculate target Q-values using the target network
        target_q_values = self.target_network.predict(np.array(next_states), verbose=0)
        max_target_q_values = np.max(target_q_values, axis=1)
        targets = rewards + self.discount_factor * (1 - np.array(dones)) * max_target_q_values
        # Update the Q-network
        with tf.GradientTape() as tape:
            q_values = self.q_network(np.array(states))
            actions = np.array(actions)
            selected_q_values = tf.reduce_sum(q_values * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.reduce_mean(tf.square(targets - selected_q_values))
        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.q_network.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())


# Initialise all the parameters for the loop
env = RobotEnvironment()
state_dim = env.state_dim
num_actions = env.num_actions
learning_rate = 0.01
discount_factor = 0.99
epsilon = 0.1
epsilon_decay = 0.995
min_epsilon = 0.01
steps_per_episodes = env.points
num_episodes = 3000
agent = DQNAgent(env, state_dim, num_actions, learning_rate, discount_factor, epsilon)
batch_size = 64
max_replay_buffer_size = 10000
rewards_per_episode = []
episode = 1


# Qlearning algorithm
for episode in range(num_episodes+1):
    state = env.reset()
    episode_reward = 0
    for step in range(steps_per_episodes):
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.store_experience(state, action, reward, next_state, done)
        if len(agent.replay_buffer) > max_replay_buffer_size:
            agent.replay_buffer.pop(0)  # Remove the oldest experience
        agent.train(batch_size)
        state = next_state
        episode_reward += reward
        agent.epsilon = max(agent.epsilon * epsilon_decay, min_epsilon)
        if done:
            break
    rewards_per_episode.append(episode_reward)

    if (episode) % 100 == 0:
        agent.update_target_network()
        # Plot every 100 episodes robot trajectory
        for i in range(len(env.pen_trajectory_x)):
            plt.figure(1)
            env.plot_robot_links_line(env.joint1_angle_track[i], env.joint2_angle_track[i], env.L1, env.L2)
            plt.plot(env.pen_trajectory_x[:i + 1], env.pen_trajectory_y[:i + 1], 'r',
                     label="Robot Writing" if i == 0 else "")
        plt.plot(env.target_trajectory[:, 0], env.target_trajectory[:, 1], 'g', label="Target")
        plt.title("Episode " + str(episode) + " Robot Trajectory")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.legend()
        # Create GIF
        env.create_gif_for_episode_bullet(episode, env)
        env.create_gif_for_episode_line(episode, env)

        # Plot the rewards per episode
        plt.figure(2)
        plt.plot(rewards_per_episode)
        best_reward_episode = np.argmax(rewards_per_episode)
        best_reward = rewards_per_episode[best_reward_episode]
        plt.plot(best_reward_episode, best_reward, 'ro', label=f"Best Reward: {best_reward}")
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Rewards per Episode')
        plt.legend()
        plt.savefig(f'REWARDS/REWARDS {episode}')
        plt.close()
    print(f"Episode: {episode}/{num_episodes}   ", f"Reward: {episode_reward}")

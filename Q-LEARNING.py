#==========VIRTUAL 2DOF ROBOT WRTITING VIA Q-LEARNING==========
#This is an example for learning the sinewave shape with Q-Learning


import numpy as np
import matplotlib.pyplot as plt
import itertools
from matplotlib.animation import FuncAnimation

class RobotEnvironment:
    def __init__(self):
        self.steps = 15                                     # Number of discrete steps
        self.points = self.steps                            # Number of x,y points of the characters
        self.num_states = self.steps ** 2 + self.steps + 1  # Number of states
        self.num_actions = 15**2                            # Number of actions

        # Define the robot model parameters
        self.L1 = 5     # Link 1 length
        self.L2 = 5    # Link 2 length

        #Define the desired trajectory
        self.x, self.y = self.LETTER_C()                            # Choosing the desired letter
        self.target_trajectory = np.column_stack((self.x, self.y))  # The trajectory that represents the desired character
        self.current_step = 1

        self.pen_position_x = self.x[0]
        self.pen_position_y = self.y[0]
        self.pen_position = np.array([self.pen_position_x, self.pen_position_y])

        self.joint1_angle, self.joint2_angle = self.inverse_kinematics(self.pen_position_x, self.pen_position_y, self.L1, self.L2)
        self.joint1_angle_track = []
        self.joint2_angle_track = []

        # Define the state discretization parameters
        self.pen_bins = np.linspace(0, (self.L1 + self.L2), num=self.points)
        self.joint_bins = np.linspace(0, np.pi, num=self.points, endpoint=False)

    def reset(self):
        # Reset the environment and return the initial state
        self.joint1_angle, self.joint2_angle = self.inverse_kinematics(self.x[0], self.y[0], self.L1, self.L2)
        self.joint1_angle_track = []
        self.joint2_angle_track = []
        self.joint1_angle_track.append(self.joint1_angle)
        self.joint2_angle_track.append(self.joint2_angle)
        self.joints_track = np.column_stack((self.joint1_angle_track, self.joint2_angle_track))

        self.pen_position = self.forward_kinematics(self.joint1_angle, self.joint2_angle)
        self.pen_trajectory_x = []
        self.pen_trajectory_y = []
        self.pen_trajectory_x.append(self.pen_position[0])
        self.pen_trajectory_y.append((self.pen_position[1]))
        self.pen_trajectory = np.column_stack((self.pen_trajectory_x, self.pen_trajectory_y))

        self.current_step = 1

        return self.discretize_state()

    def step(self, action):
        # Execute the action in the environment and return the next state, reward, and done flag
        delta_x, delta_y = self.action(action)
        self.update_states(delta_x, delta_y)
        next_state = self.discretize_state()
        reward = self.reward()
        done = self.current_step >= len(self.target_trajectory) - 1
        self.current_step += 1
        return next_state, reward, done

    def normalize_angle_rad(self, angle):
        normalized_angle = angle % (2 * np.pi)
        if normalized_angle > np.pi:
            normalized_angle -= 2 * np.pi
        return normalized_angle if normalized_angle >= 0 else normalized_angle + np.pi

    def action(self, action):
        # Compute the action space. The discretised action space.
        joint_angle1_values = np.linspace(-0.1, 0.1, 1)
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
        joint_angle1_values = np.append(joint_angle1_values,joint_angle1)
        joint_angle2_values = np.linspace(-0.1, 0.1, 1)
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

    def discretize_state(self):
        # Determine the range of joint angles for each joint
        min_joint1 = 0
        max_joint1 = np.pi
        min_joint2 = 0
        max_joint2 = np.pi
        # Calculate the width of each discretized bin for each joint
        joint1_step = (max_joint1 - min_joint1) / self.points
        joint2_step = (max_joint2 - min_joint2) / self.points
        # Determine the state indices for each joint angle
        joint1_state = int((self.joint1_angle - min_joint1) / joint1_step)
        joint2_state = int((self.joint2_angle - min_joint2) / joint2_step)
        # Ensure the state indices are within bounds
        joint1_state = max(0, min(joint1_state, self.points - 1))
        joint2_state = max(0, min(joint2_state,self.points - 1))
        # Combine joint states into a single state identifier
        state = joint1_state * self.points + joint2_state
        return state

   def reward(self):
        # Return reward
        target_point = self.target_trajectory[self.current_step]
        pen_point = self.pen_position
        distance = np.linalg.norm(pen_point - target_point)
        reward = - distance
        if distance < 0.1: reward += 10
        if distance > 1: reward -= 10
        return reward

    # ================================ROBOT_SYSTEM====================================
    def update_states(self, delta_x, delta_y):
        # Update the robot's states based on the joint updates
        self.joint1_angle += delta_x
        self.joint1_angle = self.normalize_angle_rad(self.joint1_angle)
        self.joint2_angle += delta_y
        self.joint2_angle = self.normalize_angle_rad(self.joint2_angle)
        self.joint1_angle_track.append(self.joint1_angle)
        self.joint2_angle_track.append(self.joint2_angle)
        self.joints_track = np.column_stack((self.joint1_angle_track, self.joint2_angle_track))

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
        theta2 = (x ** 2 + y ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
        q2 = np.arccos(theta2)
        q1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
        return q1, q2

    def LETTER_LINE(self):
        num_points = self.points
        x = np.linspace(-0.6, 0.0, num_points)
        y = np.linspace(-0.2, 0.6, num_points)
        return x, y

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


# Define the Q-learning agent
class QLearningAgent:
    # CORRECT
    def __init__(self, num_states, num_actions, learning_rate, discount_factor, epsilon):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros((num_states, num_actions))

    # CORRECT
    def select_action(self, state):
        # Select an action based on the epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.num_actions)
        else:
            action = np.argmax(self.q_table[state,:])
        return action

    # CORRECT
    def update_q_table(self, state, action, reward, next_state):
        # Update the Q-table based on the Q-learning update rule
        max_q_value = np.max(self.q_table[next_state,:])
        td_error = reward + self.discount_factor * max_q_value - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error


# Create an instance of the RobotEnvironment
env = RobotEnvironment()

# Define the Q-learning agent parameters
num_states = env.num_states
num_actions = env.num_actions

# Hyper Parameters
learning_rate = 0.3
discount_factor = 0.9
epsilon = 0.2
num_episodes = 4000

steps_per_episodes = env.steps

# Create the Q-learning agent
agent = QLearningAgent(num_states, num_actions, learning_rate, discount_factor, epsilon)

# Track the total rewards obtained in each episode
rewards_per_episode = []
states_per_episodes = []
episode = 1

# Run the Q-learning algorithm
for episode in range(num_episodes+1):
    state = env.reset()
    episode_reward = 0

    for step in range(steps_per_episodes):
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.update_q_table(state, action, reward, next_state)
        state = next_state
        episode_reward += reward
        states_per_episodes.append(state)
        epsilon *= 0.995

        if done:
            break

    rewards_per_episode.append(episode_reward)

    if (episode) % 100 == 0:
        # Plot every 100 episodes robot trajectory
        for i in range(len(env.pen_trajectory_x)):
            plt.figure(1)
            env.plot_robot_links_line(env.joint1_angle_track[i], env.joint2_angle_track[i], env.L1, env.L2)
            plt.plot(env.pen_trajectory_x[:i + 1], env.pen_trajectory_y[:i + 1], 'r', label="Robot Writing" if i == 0 else "")
        plt.plot(env.target_trajectory[:, 0], env.target_trajectory[:, 1], 'g', label="Target")
        plt.title("Episode " + str(episode) + " Robot Trajectory")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.legend()
        #plt.show()
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

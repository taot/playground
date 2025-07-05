# Example 4.3 Page 84

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import random

class GamblersProblemSimulator:
    def __init__(self, ph: float = 0.4, goal: int = 100):
        """
        Initialize the Gambler's Problem simulator.
        
        Args:
            ph: Probability of heads (winning a flip)
            goal: Target capital to reach (default 100)
        """
        self.ph = ph
        self.goal = goal
        self.value_function = None
        self.policy = None
        
    def value_iteration(self, theta: float = 1e-9, max_iterations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the gambler's problem using value iteration.
        
        Args:
            theta: Convergence threshold
            max_iterations: Maximum number of iterations
            
        Returns:
            Tuple of (value_function, policy)
        """
        # Initialize value function
        V = np.zeros(self.goal + 1)
        V[self.goal] = 1.0  # Winning state has value 1
        
        policy = np.zeros(self.goal + 1, dtype=int)
        
        for iteration in range(max_iterations):
            delta = 0
            old_V = V.copy()
            
            # Update value for each state (capital level)
            for s in range(1, self.goal):
                v = V[s]
                max_value = 0
                best_action = 0
                
                # Try all possible stakes
                max_stake = min(s, self.goal - s)
                for a in range(0, max_stake + 1):
                    # Expected value for this action
                    win_state = s + a
                    lose_state = s - a
                    
                    expected_value = self.ph * V[win_state] + (1 - self.ph) * V[lose_state]
                    
                    if expected_value > max_value:
                        max_value = expected_value
                        best_action = a
                
                V[s] = max_value
                policy[s] = best_action
                delta = max(delta, abs(v - V[s]))
            
            if delta < theta:
                print(f"Value iteration converged after {iteration + 1} iterations")
                break
        
        self.value_function = V
        self.policy = policy
        return V, policy
    
    def get_optimal_stake(self, capital: int) -> int:
        """Get the optimal stake for a given capital level."""
        if self.policy is None:
            self.value_iteration()
        return self.policy[capital]
    
    def simulate_episode(self, initial_capital: int = 1, use_optimal_policy: bool = True, 
                        custom_policy: Dict[int, int] = None) -> Tuple[bool, int, List[int]]:
        """
        Simulate a single episode of the gambler's problem.
        
        Args:
            initial_capital: Starting capital
            use_optimal_policy: Whether to use the optimal policy
            custom_policy: Custom policy as dict {capital: stake}
            
        Returns:
            Tuple of (won, final_capital, capital_history)
        """
        capital = initial_capital
        capital_history = [capital]
        
        while 0 < capital < self.goal:
            # Determine stake
            if custom_policy:
                stake = custom_policy.get(capital, 0)
            elif use_optimal_policy:
                stake = self.get_optimal_stake(capital)
            else:
                # Random policy for comparison
                max_stake = min(capital, self.goal - capital)
                stake = random.randint(0, max_stake)
            
            # Ensure stake is valid
            stake = max(0, min(stake, capital, self.goal - capital))
            
            # Flip coin
            if random.random() < self.ph:
                # Heads - win
                capital += stake
            else:
                # Tails - lose
                capital -= stake
            
            capital_history.append(capital)
        
        won = capital >= self.goal
        return won, capital, capital_history
    
    def run_simulation(self, n_episodes: int = 1000, initial_capital: int = 1, 
                      use_optimal_policy: bool = True, custom_policy: Dict[int, int] = None,
                      verbose: bool = True) -> Dict:
        """
        Run the simulation for N episodes and collect results.
        
        Args:
            n_episodes: Number of episodes to simulate
            initial_capital: Starting capital for each episode
            use_optimal_policy: Whether to use optimal policy
            custom_policy: Custom policy to use
            verbose: Whether to print progress
            
        Returns:
            Dictionary with simulation results
        """
        if use_optimal_policy and self.policy is None:
            print("Computing optimal policy...")
            self.value_iteration()
        
        results = {
            'episodes': [],
            'wins': 0,
            'losses': 0,
            'win_rate': 0,
            'episode_lengths': [],
            'final_capitals': []
        }
        
        for episode in range(n_episodes):
            won, final_capital, capital_history = self.simulate_episode(
                initial_capital, use_optimal_policy, custom_policy
            )
            
            results['episodes'].append({
                'episode': episode + 1,
                'won': won,
                'final_capital': final_capital,
                'episode_length': len(capital_history) - 1,
                'capital_history': capital_history
            })
            
            if won:
                results['wins'] += 1
            else:
                results['losses'] += 1
            
            results['episode_lengths'].append(len(capital_history) - 1)
            results['final_capitals'].append(final_capital)
            
            if verbose and (episode + 1) % (n_episodes // 10) == 0:
                current_win_rate = results['wins'] / (episode + 1)
                print(f"Episode {episode + 1}/{n_episodes}, Win rate so far: {current_win_rate:.3f}")
        
        results['win_rate'] = results['wins'] / n_episodes
        results['avg_episode_length'] = np.mean(results['episode_lengths'])
        results['std_episode_length'] = np.std(results['episode_lengths'])
        
        return results
    
    def print_results(self, results: Dict):
        """Print a summary of simulation results."""
        print("\n" + "="*50)
        print("GAMBLER'S PROBLEM SIMULATION RESULTS")
        print("="*50)
        print(f"Probability of heads (ph): {self.ph}")
        print(f"Goal: ${self.goal}")
        print(f"Total episodes: {len(results['episodes'])}")
        print(f"Wins: {results['wins']}")
        print(f"Losses: {results['losses']}")
        print(f"Win rate: {results['win_rate']:.4f}")
        print(f"Average episode length: {results['avg_episode_length']:.2f} flips")
        print(f"Std episode length: {results['std_episode_length']:.2f} flips")
        
        if self.value_function is not None:
            theoretical_win_prob = self.value_function[1]
            print(f"Theoretical win probability: {theoretical_win_prob:.4f}")
            print(f"Difference from simulation: {abs(results['win_rate'] - theoretical_win_prob):.4f}")
    
    def plot_results(self, results: Dict):
        """Plot simulation results and optimal policy."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot value function and policy
        if self.value_function is not None and self.policy is not None:
            capitals = range(1, self.goal)
            
            ax1.plot(capitals, self.value_function[1:self.goal])
            ax1.set_xlabel('Capital')
            ax1.set_ylabel('Value (Win Probability)')
            ax1.set_title('Optimal Value Function')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(capitals, self.policy[1:self.goal])
            ax2.set_xlabel('Capital')
            ax2.set_ylabel('Optimal Stake')
            ax2.set_title('Optimal Policy')
            ax2.grid(True, alpha=0.3)
        
        # Plot episode lengths distribution
        ax3.hist(results['episode_lengths'], bins=30, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Episode Length (flips)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Episode Lengths')
        ax3.grid(True, alpha=0.3)
        
        # Plot win rate over time
        cumulative_wins = np.cumsum([ep['won'] for ep in results['episodes']])
        episode_numbers = range(1, len(results['episodes']) + 1)
        win_rates = cumulative_wins / np.array(episode_numbers)
        
        ax4.plot(episode_numbers, win_rates, alpha=0.7)
        if self.value_function is not None:
            ax4.axhline(y=self.value_function[1], color='r', linestyle='--', 
                       label=f'Theoretical: {self.value_function[1]:.4f}')
            ax4.legend()
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Cumulative Win Rate')
        ax4.set_title('Win Rate Convergence')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def main0():
    # Create simulator with ph = 0.4 (as in the example)
    simulator = GamblersProblemSimulator(ph=0.4)
    
    # Run simulation
    results = simulator.run_simulation(n_episodes=10000, initial_capital=1)
    
    # Print results
    simulator.print_results(results)
    
    # Plot results
    simulator.plot_results(results)
    
    # Example of custom policy simulation
    print("\n" + "="*50)
    print("COMPARISON WITH RANDOM POLICY")
    print("="*50)
    
    # Run with random policy
    random_results = simulator.run_simulation(
        n_episodes=1000, 
        initial_capital=1, 
        use_optimal_policy=False,
        verbose=False
    )
    
    print(f"Optimal policy win rate: {results['win_rate']:.4f}")
    print(f"Random policy win rate: {random_results['win_rate']:.4f}")
    print(f"Improvement: {results['win_rate'] - random_results['win_rate']:.4f}")


def main1():
    simulator = GamblersProblemSimulator(ph=0.4, goal=10)
    simulator.simulate_episode(
        initial_capital=1,
        custom_policy=
    )


if __name__ == "__main__":
    main1()

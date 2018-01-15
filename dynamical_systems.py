"""
Define a class representing a dynamical system.

Authors : Arnaud Fanthomme, Thomas Bazeille, Tristan Sterin
"""

import numpy as np
import matplotlib.pyplot as plt

class dynamical_system:
    def __init__(self, n_hidden, n_inputs, n_outputs, flow_function, output_function, driving_function):
        """

        :param n_inputs:
        :param n_hidden:
        :param flow_function: Governs the flow of the hidden state. (n_hidden x 1) -> (n_hidden x 1)
        :param output_function: Governs the observation. Should be linear. (n_hidden + n_inputs) x 1 -> (n_outputs x 1)
        :param driving_function: Influence of inputs on state. Should be linear. (n_inputs x 1) -> (n_hidden x 1)
        """
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.state_flow = flow_function
        self.state_drive = driving_function
        self.observation = output_function

        # Force our system to always remain in [-5, 5]**n_hidden
        self.clipmax = 5
        self.clipmin = -5

        # Initialize state randomly and define the gaussian noise
        self.hidden_state = np.random.uniform(self.clipmin, self.clipmax, size=n_hidden)
        # print('hiddenstate', self.hidden_state)
        self.gaussian_noise = lambda: np.random.normal(scale=0.05)

    # def plot(self):
    #     plt.figure()
    #     X, Y = np.meshgrid(np.linspace(self.clipmin, self.clipmax, 20), np.linspace(self.clipmin, self.clipmax, 20))
    #     U, V = self.state_flow(zip(X, Y))

    def step(self, inputs):
        self.hidden_state = self.state_flow(self.hidden_state) + self.state_drive(inputs) + self.gaussian_noise()
        self.hidden_state = np.clip(self.hidden_state, self.clipmin, self.clipmax)
        print(self.hidden_state, inputs)
        print(np.concatenate((self.hidden_state, inputs)))
        return self.observation(np.concatenate((self.hidden_state, inputs))) + self.gaussian_noise()

    def generate_trajectory(self, n_steps=100, inputs=None):
        if not inputs:
            inputs = np.random.normal(size=(n_steps, self.n_inputs))
            # print(inputs[1:3, :])

        traj_hidden = np.zeros((n_steps, self.n_hidden))
        outputs = np.zeros((n_steps, self.n_outputs))

        for t in range(n_steps):
            # print(inputs[t, :])
            outputs[t, :] = self.step(inputs[t, :])
            traj_hidden[t, :] = self.hidden_state

        return outputs, traj_hidden

def sin_flow(x):
    return np.sin(x)

def no_drive(n_hidden):
    return lambda _: np.zeros(n_hidden)

dummy_obs = lambda x: x # Output is concatenation of hidden_state and input

example_1 = dynamical_system(n_hidden=1, n_inputs=2, n_outputs=3, flow_function=sin_flow, driving_function=no_drive(1),
                             output_function=dummy_obs)

if __name__ == '__main__':
    outputs, true_traj = example_1.generate_trajectory()
    np.savetxt('example1.dat', outputs)
    np.savetxt('example1.traj', true_traj)




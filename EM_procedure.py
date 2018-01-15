import numpy as np
from scipy.stats import multivariate_normal
from pykalman import AdditiveUnscentedKalmanFilter

class EM_container:
    def __init__(self, output_sequence, rbf_centers, rbf_covs, A, b, C, d):
        self.n_rbfs = rbf_centers.shape[0]
        self.n_hidden = A.shape[0]
        self.n_outputs = C.shape[0]

        def yes(i):
            def yesyes(x):
                return multivariate_normal.pdf(x, mean=rbf_centers[i], cov=rbf_covs[i])
            return yesyes

        self.basis_functions = [yes(i) for i
                                in range(self.n_rbfs)]
        self.rho = lambda x: np.array([rho(x) for rho in self.basis_functions])
        self.f_coords = 0.3 * np.random.randn(self.n_rbfs, self.n_hidden) # List of coefs for representing f (self.n_rbfs x self.n_hidden)
        self.linear_f = A  # Matrix of the linear part of f (self.n_hidden x self.n_hidden)
        self.bias_f = b # Bias for the linear part of f (self.n_hidden)

        self.g_coords = 0.3 * np.random.randn(self.n_rbfs, self.n_outputs) # List of coefs for representing g (self.n_rbfs x self.n_outputs)
        self.linear_g = C  # Matrix of the linear part of g (self.n_outputs x self.n_hidden)
        self.bias_g = d # Bias for the linear part of g

        self.v_cov = np.eye(self.n_outputs) # Covariance matrix for v noise
        self.w_cov = np.eye(self.n_hidden) # Covariance matrix for w noise

        self.hidden_sequence = []
        self.output_sequence = output_sequence

        self.g_estimator = lambda x: np.sum(np.dot(self.g_coords.T, self.rho(x))) + np.dot(self.linear_g, x) + self.bias_g
        self.f_estimator = lambda x: np.sum(np.dot(self.f_coords.T, self.rho(x))) + np.dot(self.linear_f, x) + self.bias_f

    def E_step(self):
        # print(self.g_coords.shape)
        # print(self.w_cov,self.v_cov)
        ukf = AdditiveUnscentedKalmanFilter(self.f_estimator, self.g_estimator, transition_covariance=self.w_cov, observation_covariance=self.v_cov)
        self.hidden_sequence = ukf.smooth(self.output_sequence)[0]
        

    def M_step(self):
        # We assume absence of external inputs
        J = self.hidden_sequence.shape[0]

        # First, maximize the update parameters
        f_estimator = lambda x: np.sum(np.dot(self.f_coords.T, self.rho(x))) \
                                + np.dot(self.linear_f, x) + self.bias_f
        z_f = np.array([f_estimator(state) for state in self.hidden_sequence])
        #print(z_f.shape)
        phi_f_T = np.array([np.concatenate(([rho(state) for rho in self.basis_functions],
                                            state, [1.])) for state in self.hidden_sequence])
        #print(phi_f_T.shape)

        tmp = np.sum([np.dot(np.reshape(z_f[j, :], (self.n_hidden, 1)), np.reshape(phi_f_T[j, :], (1, self.n_rbfs +
                                                                    self.n_hidden + 1))) for j in range(J)], axis=0)

        theta_f  = tmp / np.sum([np.dot(phi_f_T[j, :], phi_f_T[j, :]) for j in range(J)], axis=0)
        #print(theta_f.shape)
        Q_f = np.mean([np.dot(np.reshape(z_f[j, :], (self.n_hidden, 1)), np.reshape(z_f[j, :], (1, self.n_hidden)) )
                       for j in range(J)], axis=0)
        Q_f -= theta_f.dot(np.mean([np.dot(np.reshape(phi_f_T[j, :], (self.n_rbfs + self.n_hidden + 1, 1)),
                                           np.reshape(z_f[j, :], (1, self.n_hidden))) for j in range(J)], axis=0))

        # Set the container attributes :
        self.f_coords = theta_f[:, :self.n_rbfs].T
        self.linear_f = theta_f[:, self.n_rbfs:self.n_rbfs+self.n_hidden]
        self.bias_f = theta_f[:, self.n_hidden + self.n_rbfs]
        self.w_cov = Q_f

        # Then, maximize the ouput parameters
        self.n_outputs = self.output_sequence.shape[1]

        # First, maximize the update parameters
        g_estimator = lambda x: np.sum(np.dot(self.g_coords.T, self.rho(x))) + np.dot(self.linear_g, x) + self.bias_g
        z_g = np.array([g_estimator(state) for state in self.hidden_sequence])
        #print(z_g.shape)
        phi_g_T = np.array([np.concatenate(([rho(state) for rho in self.basis_functions],
                                            state, [1.])) for state in self.hidden_sequence])
        #print(phi_g_T.shape)

        tmp = np.sum([np.dot(np.reshape(z_g[j, :], (self.n_outputs, 1)), np.reshape(phi_g_T[j, :], (1, self.n_rbfs +
                                                                    self.n_hidden + 1))) for j in range(J)], axis=0)

        theta_g  = tmp / np.sum([np.dot(phi_g_T[j, :], phi_g_T[j, :]) for j in range(J)], axis=0)
        #print(theta_g.shape)
        Q_g = np.mean([np.dot(np.reshape(z_g[j, :], (self.n_outputs, 1)), np.reshape(z_g[j, :], (1, self.n_outputs)) )
                       for j in range(J)], axis=0)
        Q_g -= theta_g.dot(np.mean([np.dot(np.reshape(phi_g_T[j, :], (self.n_rbfs + self.n_hidden + 1, 1)),
                                           np.reshape(z_g[j, :], (1, self.n_outputs))) for j in range(J)], axis=0))

        #print(Q_f,Q_g)
        # Set the container attributes :
        self.g_coords = theta_g[:, :self.n_rbfs].T
        self.linear_g = theta_g[:, self.n_rbfs:self.n_rbfs+self.n_hidden]
        self.bias_g = theta_g[:, self.n_hidden + self.n_rbfs]
        self.v_cov = Q_g

        self.g_estimator = lambda x: np.sum(np.dot(self.g_coords.T, self.rho(x))) + np.dot(self.linear_g, x) + self.bias_g
        self.f_estimator = lambda x: np.sum(np.dot(self.f_coords.T, self.rho(x))) + np.dot(self.linear_f, x) + self.bias_f
        
        #return f_estimator, g_estimator, self.w_cov, self.v_cov



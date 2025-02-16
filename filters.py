# utils/filters.py
import numpy as np

class KalmanFilter:
    def __init__(self, A, C, Q, R):
        self.A = A
        self.C = C
        self.Q = Q
        self.R = R
        self.reset()

    def reset(self):
        self.Xkf = 0.0
        self.P_k = 1.0

    def predict(self):
        self.X_pre = self.A * self.Xkf
        self.P_pre = self.A * self.P_k * self.A + self.Q

    def update(self, Y):
        K = self.P_pre * self.C / (self.C * self.P_pre * self.C + self.R)
        e = Y - self.C * self.X_pre
        self.Xkf = self.X_pre + K * e
        self.P_k = (1 - K * self.C) * self.P_pre

    def step(self, Y):
        self.predict()
        self.update(Y)

    def get_P(self):
        return self.P_k

class StateEstimator:
    def __init__(self, num_users, A_params, C_params, Q_params, R_params, alpha, sigma1=1e-9, sigma2=1e-9):
        self.num_users = num_users
        self.A_params = A_params
        self.C_params = C_params
        self.Q_params = Q_params
        self.R_params = R_params
        self.alpha = alpha
        self.sigma1 = sigma1
        self.sigma2 = sigma2

        self.kalman_filters = [
            KalmanFilter(self.A_params[i], self.C_params[i], self.Q_params[i], self.R_params[i])
            for i in range(self.num_users)
        ]

        self.P_steadystate = np.zeros(num_users)
        self.overline_P = np.zeros(num_users)
        self.P_k = np.zeros(num_users)

        self.reset()

    def reset(self):
        X = 1
        k_steps = 100  # 卡尔曼滤波步数
        for i in range(self.num_users):
            self.kalman_filters[i].reset()
            for k in range(k_steps):
                if k == 0:
                    Y = 1
                    X = 1
                else:
                    Y = self.C_params[i] * X
                self.kalman_filters[i].step(Y)
            self.P_steadystate[i] = self.kalman_filters[i].get_P()
            self.overline_P[i] = self.P_steadystate[i]
            self.P_k[i] = self.overline_P[i]

    def update(self, gamma_values, h_i_list, user_indices):
        for idx, i in enumerate(user_indices):
            gamma = gamma_values[i]
            h_i = h_i_list[i]
            if gamma == 1:
                self.P_k[i] = self.overline_P[i]
            else:
                self.kalman_filters[i].predict()
                self.P_k[i] = self.kalman_filters[i].get_P()

    def get_current_P(self):
        return self.P_k

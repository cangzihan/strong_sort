import numpy as np
import scipy
import matplotlib.pylab as plt
from .kalman_filter import KalmanFilter


class StrongEKF(KalmanFilter):
    def __init__(self, lamda_max=1.5, weakening_factor=10):
        super().__init__()
        self.fadding_factor_max = lamda_max
        # Rename
        self.A = self._motion_mat
        self.H = self._update_mat

        self.fading_factor = 1
        self.V = None
        self.forgetting_factor = 0.95
        self.weakening_factor = weakening_factor

        self.cal_mode = "default"

    def __cal_fading_factor(self, mean, covariance, z):
        # Calculate M
        M = np.linalg.multi_dot((self.A, covariance, self.A.transpose()))
        M = np.linalg.multi_dot((self.H, M, self.H.transpose()))

        # Calculate Q
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        self.Q = np.diag(np.square(np.r_[std_pos, std_vel]))

        # Calculate R
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        self.R = np.diag(np.square(std))

        # Calculate N
        err = z - np.dot(self._update_mat, mean)
        if self.V is None:
            self.V = np.dot(err, err.transpose())
        else:
            self.V = (self.forgetting_factor * self.V + np.dot(err, err.transpose())) / (1 + self.forgetting_factor)

        N = self.V - np.linalg.multi_dot((self.H, self.Q, self.H.transpose())) - self.weakening_factor * self.R

        # Calculate fading factor
        fading_factor0 = np.trace(N) / np.trace(M)
        self.fading_factor = max(1, fading_factor0)
        a = 2 * (self.fadding_factor_max - 1)
        b = 2 - self.fadding_factor_max
        self.fading_factor = 1 / (1 + np.exp(-self.fading_factor+1)) * a + b
        #self.fading_factor = min(self.fadding_factor_max, self.fading_factor)
        if False:
            print("Fading factor:", self.fading_factor)
        return self.fading_factor

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean_pre = np.dot(self._motion_mat, mean)
        self.__cal_fading_factor(mean_pre, covariance, measurement)
        covariance_stf = self.fading_factor * np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        if self.cal_mode == "fast":
            projected_mean, projected_cov = self.project(mean_pre, covariance_stf)

            chol_factor, lower = scipy.linalg.cho_factor(
                projected_cov, lower=True, check_finite=False)
            kalman_gain = scipy.linalg.cho_solve(
                (chol_factor, lower), np.dot(covariance_stf, self._update_mat.T).T,
                check_finite=False).T
            innovation = measurement - projected_mean

            new_mean = mean_pre + np.dot(innovation, kalman_gain.T)
            new_covariance = covariance_stf - np.linalg.multi_dot((
                kalman_gain, projected_cov, kalman_gain.T))
        else:
            # Correct
            k1 = np.dot(covariance_stf, self.H.transpose())
            k2 = np.dot(self.H, k1) + self.R
            K = np.dot(k1, np.linalg.inv(k2))

            new_mean = mean_pre + np.dot(K, measurement - np.dot(self.H, mean_pre))
            new_covariance = np.dot(np.identity(covariance.shape[0]) - np.dot(K, self.H), covariance_stf)

        return new_mean, new_covariance


def show(samples, pre_list, cor_list, show_priori=True, show_posteriori=True):
    x = [samples[i][0] for i in range(len(samples))]
    y = [samples[i][1] for i in range(len(samples))]
    plt.plot(x, y, label='Actual Track')
    plt.scatter(x, y)

    if show_priori:
        x = [round(pre_list[i][0], 3) for i in range(len(pre_list))]
        y = [round(pre_list[i][1], 3) for i in range(len(pre_list))]
        plt.plot(x, y, 'r', label='Priori state estimate')
        plt.scatter(x, y, c='r')

    if show_posteriori:
        x = [round(cor_list[i][0], 3) for i in range(len(cor_list))]
        y = [round(cor_list[i][1], 3) for i in range(len(cor_list))]
        plt.plot(x, y, 'g', label='Posteriori state estimate')
        plt.scatter(x, y, c='g')

    plt.legend()
    plt.show()


if __name__ == "__main__":
    t = np.linspace(0, 10, 100)
    x = np.sin(t)
    x_pre_list = []
    x_cor_list = []
    x_samples = []
    my_filter = StrongEKF(initial_position=[t[0], x[0]])
    import time

    t0 = time.time()
    for i in range(len(t)):
        a, b = my_filter.run([t[i], x[i]])
        x_samples.append([t[i], x[i]])
        x_pre_list.append(a)
        x_cor_list.append(b)
        # my_filter.predict()
    for i in range(10):
        pass
        # my_filter.run(my_filter.x_cor_list[-1])
    t = time.time() - t0
    print(t)
    show(x_samples, x_pre_list, x_cor_list, show_posteriori=False)

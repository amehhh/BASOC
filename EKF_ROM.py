"""Extended Kalman filter using a reduced-order model (ROM) as the dynamical system
to estimate the state of a full-order model (FOM) using output measurements from the FOM.
Reduced-state estimates can be lifted to the full-state space using the ROM basis.
"""

import numpy as np
from scipy.integrate import solve_ivp


class ExtendedKalmanFilter:
    """Estimate ROM states with linear observations y = C_ROM @ z.

    The caller supplies model functions, the observation matrix, initial
    conditions, noise covariances, sampling interval, and solver tolerances.
    Q is the discrete process-noise covariance added at each prediction.
    R is the measurement-noise covariance.
    P0 is the initial state-estimation error covariance.
    """

    def __init__(self, z0, P0, Q, R, dt, C_ROM, dynamics, jacobian, rtol, atol):

        # Initial state estimate and estimation-error covariance
        self.z0 = np.asarray(z0, dtype=float).copy()
        self.P0 = np.asarray(P0, dtype=float).copy()

        # Process-noise and measurement-noise covariances
        self.Q = np.asarray(Q, dtype=float).copy()
        self.R = np.asarray(R, dtype=float).copy()

        # Observation matrix, ROM dynamics, and state Jacobian
        self.C_ROM = np.asarray(C_ROM, dtype=float).copy()
        self.dynamics = dynamics
        self.jacobian = jacobian

        # Sampling interval and integration tolerances
        self.dt = dt
        self.rtol, self.atol = rtol, atol

        # P0 must have one row and column for each reduced state
        if self.P0.shape != (self.z0.size, self.z0.size):
            raise ValueError("P0 must be a square covariance matrix matching z0.")

        # Current prediction/update step
        self.currentTimeStep = 0

        # Posterior estimates: after incorporating measurements
        # Treat z0 and P0 as the initial posterior at time zero
        self.estimates_aposteriori = [self.z0]
        self.estimationErrorCovarianceMatricesAposterior = [self.P0]

        # Prior estimates: predictions before incorporating measurements
        # The first prior corresponds to time step one
        self.estimates_apriori = []
        self.estimationErrorCovarianceMatricesAprior = []

        # Store Kalman gains and measurement innovations
        self.gainmatrices = []
        self.errors = []

    def ROM_output(self, z):
        """Compute the predicted observation from the reduced state."""

        return self.C_ROM @ z

    def propagateDynamics(self, u):
        """Predict the state and covariance with the control held constant."""

        # Retrieve the current posterior state estimate
        z_plus = self.estimates_aposteriori[self.currentTimeStep]

        # Integrate the ROM over one sampling interval
        sol = solve_ivp(lambda time, z: self.dynamics(z, u), (0, self.dt), z_plus, method="RK45", rtol=self.rtol, atol=self.atol)

        if not sol.success:
            raise RuntimeError(sol.message)

        z_minus = sol.y[:, -1]

        # Linearize the ROM at the posterior state
        J = self.jacobian(z_plus, u)
        F = np.eye(z_plus.size) + self.dt * J

        # Propagate covariance and add discrete process-noise covariance
        P_plus = self.estimationErrorCovarianceMatricesAposterior[self.currentTimeStep]
        P_minus = F @ P_plus @ F.T + self.Q

        # Store the prior state and covariance.
        self.estimates_apriori.append(z_minus)
        self.estimationErrorCovarianceMatricesAprior.append(P_minus)

        self.currentTimeStep += 1

    def computeAposterioriEstimate(self, currentMeasurement):
        """Correct the prior estimate using the current measurement."""

        # Retrieve the predicted state and covariance
        z_minus = self.estimates_apriori[self.currentTimeStep - 1]
        P_minus = self.estimationErrorCovarianceMatricesAprior[self.currentTimeStep - 1]

        # Innovation covariance: uncertainty in the predicted measurement
        S = self.C_ROM @ P_minus @ self.C_ROM.T + self.R

        # Compute the Kalman gain by solving a linear system
        K = np.linalg.solve(S.T, (P_minus @ self.C_ROM.T).T).T

        # Innovation: measured output minus predicted output
        error = np.asarray(currentMeasurement).reshape(-1) - self.ROM_output(z_minus)

        # Correct the state estimate using the innovation
        z_plus = z_minus + K @ error

        # Joseph covariance update improves numerical stability
        I_KC = np.eye(z_minus.size) - K @ self.C_ROM
        P_plus = I_KC @ P_minus @ I_KC.T + K @ self.R @ K.T

        # Store the gain, innovation, and posterior estimates
        self.gainmatrices.append(K)
        self.errors.append(error)

        self.estimates_aposteriori.append(z_plus)
        self.estimationErrorCovarianceMatricesAposterior.append(P_plus)
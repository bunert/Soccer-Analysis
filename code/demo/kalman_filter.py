import sympy
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag, norm
from math import sqrt, atan2


################################################################################
# comments for me:
# Kalman filter for one person with 25 keypoints (body_25 model)
################################################################################


class Kalman:

    def __init__(self):

        #Variables:
        camera_number = 5                    # number ob cameras
        state_dim = 6                        # per keypoint
        measurement_dim = 2 * camera_number  # per keypoint
        keypoint_number = 18                 # COCO model

        R_std=0.025                                   # measurement noise: before 1.0
        Q_var=0.3                                     # process/system noise: before 5.0
        # state dimension: [x,y,z,x',y',z'] for 18 keypoints
        x_dimension = state_dim * keypoint_number           # 108
        # measurement dimension: [x0, y0, x1, y1, x2,..., x4, y4] screen coord. for all 5 cameras for 18 keypoints
        z_dimension = measurement_dim * keypoint_number     # 180
        dt = 0.04                                     # time steps: 1/25 FPS


        #Number of pixels in x and y direction where we still accept the filter output.
        self.threshold = 100
        #Number of frames without update
        self.frames_without_update = 0
        self.frames_without_update_allowed = 5


        #Build State Vector X: [x,y,z,x',y',z']
        X = np.zeros((x_dimension, 1))

        #Build Transition Matrix F or also called A
        block = np.matrix([[1., 0., 0., dt, 0., 0.],
                          [0., 1., 0., 0., dt, 0.],
                          [0., 0., 1., 0., 0., dt],
                          [0., 0., 0., 1., 0., 0.],
                          [0., 0., 0., 0., 1., 0.],
                          [0., 0., 0., 0., 0., 1.]])

        F = block_diag(block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block)


        H = np.zeros((z_dimension, x_dimension))


        #Create the actual filter now:
        kf = ExtendedKalmanFilter(dim_x=x_dimension, dim_z=z_dimension)

        # state vector
        kf.x = X

        # state covariance
        kf.P *= np.zeros((x_dimension,x_dimension))

        # Process Model
        kf.F = F

        # Observation Model
        kf.H = H

        # measurement noise
        kf.R = np.eye(z_dimension) * (R_std)

        # process noise
        kf.Q = np.eye(x_dimension) * (Q_var)

        # discrete white noise for the process noise
        # q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_var)
        #
        # block = block_diag(q,q)
        #
        # kf.Q =  block_diag(block,
        #                 block,
        #                 block,
        #                 block,
        #                 block,
        #                 block,
        #                 block,
        #                 block,
        #                 block,
        #                 block,
        #                 block,
        #                 block,
        #                 block,
        #                 block,
        #                 block,
        #                 block,
        #                 block,
        #                 block,
        #                 block,
        #                 block,
        #                 block,
        #                 block,
        #                 block,
        #                 block,
        #                 block)

        self.filter = kf

    def initialize_state(self, data_dict, keypoints):
        #Number of pixels in x and y direction where we still accept the filter output.
        self.threshold = 200

        #Number of frames without update
        self.keypoints_no_update_count = np.zeros(25)
        self.frames_without_update_allowed = 3



        # Build Measurement Function Matrix H
        # stack H matrix
        H = []

        for i in range(0,50):
            row=[]
            for j in range(0,100):
                if (2*i == j):
                    row.append(1.)
                else:
                    row.append(0.)

            H.append(row)

        H = np.array(H)


        #Build P Matrix
        #Build Transition Matrix
        block = np.array([[500., 0., 0., 0.],
                         [0., 20., 0., 0.],
                         [0., 0., 500., 0.],
                         [0., 0., 0., 20.]])

        P = block_diag(block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,)


        #Set to this state
        new_state = []

        for index,keypoint in enumerate(keypoints[0]):
            x_openpose, y_openpose, confidence = keypoint[0], keypoint[1], keypoint[2]

            if confidence != 0:
                new_state.append(x_openpose)
                new_state.append(0.)
                new_state.append(y_openpose)
                new_state.append(0.)
            else:
                new_state.append(0.)
                new_state.append(0.)
                new_state.append(0.)
                new_state.append(0.)

        kf.x = np.array([new_state]).T

        kf.P = P
        kf.F = F
        kf.H = H

        kf.R *= np.eye(50) * (R_std**2)
        q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_var)
        block = block_diag(q,q)

        kf.Q =  block_diag(block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block,
                        block)

        self.filter = kf

    #Not used anymore
    def update_old(self, keypoints):
        """do some sort of processing on the keypoints and then update the filter"""


        # TODO: idea is this: when we have no openpose measurement for a certain keypoint,
        # then we just go ahead and modify the R matrix accordingly, setting R diagonal very
        # high for the missing values and then passing the previous filter value as measurement
        # that way the filter should not really start overestimating the value of the predictions
        # that should allow us to skip a few frames and still have a reasonable estimate for the filter.

        new_state = []
        no_update = 0

        # Get some information on the "quality of the measurement", so the number
        # of keyposes that we have and their estimation score

        mean_estimation_score = np.array(keypoints[0])[:,2].mean()
        number_of_estimates = np.count_nonzero(np.array(keypoints[0])[:,0])

        #print("Mean Score: {}".format(mean_estimation_score))
        #print("# Of Estimates: {}".format(number_of_estimates))

        for index,keypoint in enumerate(keypoints[0]):
            x_openpose, y_openpose, confidence = keypoint[0], keypoint[1], keypoint[2]
            x_filter, y_filter = self.filter.x[index * 4][0], self.filter.x[index * 4 + 2][0]

            delta_x, delta_y = abs(x_filter-x_openpose), abs(y_filter-y_openpose)

            if confidence == 0:
                #x_openpose = x_filter
                #y_openpose = y_filter

                x_openpose = None
                y_openpose = None
                no_update += 1

            if (delta_x > self.threshold) or (delta_y > self.threshold):

                if self.frames_without_update < self.frames_without_update_allowed:
                    x_openpose = None
                    y_openpose = None
                    no_update += 1
                else:
                    self.frames_without_update = 0

            new_state.append(x_openpose)
            new_state.append(y_openpose)

        if no_update >= 10:
            self.frames_without_update += 1

        self.filter.update(new_state)


    def update(self,keypoints):
        """do some sort of processing on the keypoints and then update the filter"""

        # TODO: idea is this: when we have no openpose measurement for a certain keypoint,
        # then we just go ahead and modify the R matrix accordingly, setting R diagonal very
        # high for the missing values and then passing the previous filter value as measurement
        # that way the filter should not really start overestimating the value of the predictions
        # that should allow us to skip a few frames and still have a reasonable estimate for the filter.

        new_state = []

        #Stores the indexes of the values in the filter that should be reinitialized with the openpose output
        re_initialize_indexes = []

        #Stores the indexes of the keypoints that did not get updated this time around
        no_update = []

        # Get some information on the "quality of the measurement", so the number
        # of keyposes that we have and their estimation score
        mean_estimation_score = np.array(keypoints[0])[:,2].mean()
        number_of_estimates = np.count_nonzero(np.array(keypoints[0])[:,0])

        print("Mean Score: {}".format(mean_estimation_score))
        print("# Of Estimates: {}".format(number_of_estimates))

        #Set R to the initial value
        R_std=3
        self.filter._R = np.eye(50) * (R_std**2)
        new_R = np.eye(50) * (R_std**2)

        #Start looping over the keypoints and update with new measurement if it is good (aka not too far off)
        for index,keypoint in enumerate(keypoints[0]):

            x_openpose, y_openpose, confidence = keypoint[0], keypoint[1], keypoint[2]
            x_filter, y_filter = self.filter.x[index * 4][0], self.filter.x[index * 4 + 2][0]

            delta_x, delta_y = abs(x_filter-x_openpose), abs(y_filter-y_openpose)

            if confidence == 0:
                #no keypoint found by openpose this time
                no_update.append(index)
                x_openpose = x_filter
                y_openpose = y_filter

                #Set R matrix very large for the no-update values to avoid that the filter starts learning
                new_R[index * 2,index * 2] = 10000.
                new_R[index * 2 + 1,index * 2 + 1] = 1000.
                new_state.append(x_openpose)
                new_state.append(y_openpose)
                continue

            if (delta_x > self.threshold) or (delta_y > self.threshold):

                if self.keypoints_no_update_count[index] < self.frames_without_update_allowed:
                    x_openpose = x_filter
                    y_openpose = y_filter
                    no_update.append(index)
                    new_R[index * 2,index * 2] = 10000.
                    new_R[index * 2 + 1,index * 2 + 1] = 10000.
                else:
                    self.keypoints_no_update_count[index] = 0
                    re_initialize_indexes.append(index)

            new_state.append(x_openpose)
            new_state.append(y_openpose)

        if (len(re_initialize_indexes) > 3) and (number_of_estimates > 20):
            self.initialize_state(keypoints)

            print("Reinitializing Kalman Filter due to too much drift!: \n Mean Score: {}".format(mean_estimation_score,))
            return

        #update the list of no_updates
        for index, value in enumerate(no_update):
            self.keypoints_no_update_count[value] += 1


        self.filter._R = np.array(new_R)
        self.filter.update(new_state)


    def predict(self):
        """predicts the next state and returns the predicted next state"""

        self.filter.predict()
        return self.filter.x

    def select_best(self, keypoints):

        # TODO: Potentially input all the datums and then select the highest confidence keypoint
        # based on the 4 inputs. As next step then go and compare the predictions from the kalman filter
        # with what openpose is giving us - reject outliers and take prediction where it is closer.

        """
        [[1.27793396e+03 4.18467346e+02 7.97445953e-01]
          [1.26011047e+03 4.42006012e+02 8.51648331e-01]
          [1.22203149e+03 4.44987122e+02 8.20610523e-01]
          [1.23658801e+03 5.03757385e+02 8.24492157e-01]
          [1.29258618e+03 4.86199158e+02 7.65195727e-01]
          [1.29265417e+03 4.38960541e+02 7.56790042e-01]
          [1.33972583e+03 4.12495758e+02 8.26288760e-01]
          [1.32197717e+03 3.71434937e+02 8.08539987e-01]
          [1.25424915e+03 5.44955933e+02 4.28238153e-01]
          [1.22782422e+03 5.47973999e+02 4.41966891e-01]
          [1.25142285e+03 6.00935242e+02 1.22487575e-01]
          [0.00000000e+00 0.00000000e+00 0.00000000e+00]
          [1.27784692e+03 5.39121033e+02 3.97429943e-01]
          [1.31902795e+03 5.89160034e+02 8.58212411e-02]
          [0.00000000e+00 0.00000000e+00 0.00000000e+00]
          [1.26898621e+03 4.09592468e+02 8.11004996e-01]
          [1.28383752e+03 4.09624817e+02 7.82519400e-01]
          [1.24250146e+03 4.09698364e+02 7.20324695e-01]
          [1.29850439e+03 4.09624237e+02 1.41637519e-01]
          [0.00000000e+00 0.00000000e+00 0.00000000e+00]
          [0.00000000e+00 0.00000000e+00 0.00000000e+00]
          [0.00000000e+00 0.00000000e+00 0.00000000e+00]
          [0.00000000e+00 0.00000000e+00 0.00000000e+00]
          [0.00000000e+00 0.00000000e+00 0.00000000e+00]
          [0.00000000e+00 0.00000000e+00 0.00000000e+00]]]
        """

        returned_keypoints = []
        for index,keypoint in enumerate(keypoints[0]):

            x_openpose, y_openpose = keypoint[0], keypoint[1]
            x_filter, y_filter = self.filter.x[index * 4][0], self.filter.x[index * 4 + 2][0]

            if (x_openpose == 0) and (y_openpose == 0):

                #if we have no estimate from openpose, let's try just using filter estimate
                returned_keypoints.append([x_filter,y_filter,1])

            else:
                delta_x, delta_y = abs(x_filter-x_openpose), abs(y_filter-y_openpose)

                #Check if this delta is bigger than allowed?
                if (delta_x > self.threshold) or (delta_y > self.threshold):
                    returned_keypoints.append([x_openpose,y_openpose,keypoint[2]])

                else:
                    returned_keypoints.append([x_filter,y_filter,1])

        return returned_keypoints

    # HJacobian
    def H_of(x, A, R, T):
        # compute Jacobian of H matrix where h(x) computes
        # the range and bearing to a landmark for state x
        px = A
        py = R
        hyp = (px - x[0, 0])**2 + (py - x[1, 0])**2
        dist = sqrt(hyp)
        H = array([[-(px - x[0, 0]) / dist, -(py - x[1, 0]) / dist, 0],
                   [ (py - x[1, 0]) / hyp, -(px - x[0, 0]) / hyp, -1]])
        return H


    # Hx
    def Hx(x, A, R, T):
        # takes a state variable and returns the measurement
        # that would correspond to that state.
        px = A
        py = R
        dist = sqrt((px - x[0, 0])**2 + (py - x[1, 0])**2)
        Hx = array([[dist],[atan2(py - x[1, 0], px - x[0, 0]) - x[2, 0]]])

    return Hx

# def project(points3d, A, R, T, scale_factor=1.0, dtype=np.int32):
    # if points3d.shape[0] != 3:
    #     points3d = points3d.T
    #
    # assert(T.shape == (3, 1))
    #
    # n_points = points3d.shape[1]
    #
    # projected_points_ = A.dot(R.dot(points3d) + np.tile(T, (1, n_points)))
    # depth = projected_points_[2, :]
    # pixels = projected_points_[0:2, :] / projected_points_[2, :] / scale_factor
    #
    # if issubclass(dtype, np.integer):
    #     pixels = np.round(pixels)
    #
    # pixels = np.array(pixels.T, dtype=dtype)
    #
    # return pixels, depth

    # def hx(x):
    # # example:
    # x, y, z, x_vel, y_vel, z_vel = sympy.symbols('x y z x_vel y_vel z_vel')
    # # example:
    # X = sympy.Matrix([rho*cos(phi), rho*sin(phi), rho**2])
    # Y = sympy.Matrix([x, y, z, x_vel, y_vel, z_vel])
    # X.jacobian(Y)

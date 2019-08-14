import sympy
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag, norm
import utils.camera as cam_utils
import scipy



################################################################################
# comments for me:
# Kalman filter for one person with 25 keypoints (body_25 model)
################################################################################


class Kalman:
    def __init__(self, cameras, keypoints, R_std, Q_var):

        #Variables:
        self.camera_number = cameras                    # number ob cameras
        self.state_dim = 6                        # per keypoint
        self.measurement_dim = 2 * self.camera_number  # per keypoint
        self.keypoint_number = keypoints                # COCO model
        self.frame_number = 0                     # actual interation number of the filter

        self.R_std=R_std                                   # measurement noise: before 1.0
        self.Q_var=Q_var                                   # process/system noise: before 5.0

        # state dimension: [x,y,z,x',y',z'] for 18 keypoints
        self.x_dimension = self.state_dim * self.keypoint_number           # 108
        # measurement dimension: [x0, y0, x1, y1, x2,..., x_j, y_j] screen coord. for all j cameras for 18 keypoints
        self.z_dimension = self.measurement_dim * self.keypoint_number     # 180
        self.dt = 0.04                                     # time steps: 1/25 FPS


        # #Number of pixels in x and y direction where we still accept the filter output.
        # self.threshold = 100
        # Number of frames without update
        # self.frames_without_update = 0

        # Number of frames where a keypoint didn't get an update
        # self.keypoints_no_update_count = np.zeros(18)
        # self.frames_without_update_allowed = 5


        #Build State Vector X: [x,y,z,x',y',z'] for each keypoint
        X = np.zeros((self.x_dimension, 1))

        #Create the actual filter now:
        ekf = ExtendedKalmanFilter(dim_x=self.x_dimension, dim_z=self.z_dimension)

        # state vector
        ekf.x = X

        # state covariance
        ekf.P *= np.zeros((self.x_dimension,self.x_dimension))

        # Process Model
        # Build Transition Matrix F or also called A
        block = np.matrix([[1., 0., 0., self.dt, 0., 0.],
                          [0., 1., 0., 0., self.dt, 0.],
                          [0., 0., 1., 0., 0., self.dt],
                          [0., 0., 0., 1., 0., 0.],
                          [0., 0., 0., 0., 1., 0.],
                          [0., 0., 0., 0., 0., 1.]])

        matrix_list = []
        for i in range (self.keypoint_number):
            matrix_list.append(block)

        F = scipy.linalg.block_diag(*matrix_list)
        ekf.F = F

        # measurement noise
        ekf.R = np.eye(self.z_dimension) * (self.R_std)

        # process noise
        # TODO: which noise model should be used?
        # q = Q_discrete_white_noise(dim=2, dt=self.dt, var=self.Q_var)
        # block = np.matrix([[q[0,0], 0., 0., q[0,1], 0., 0.],
        #                   [0., q[0,0], 0., 0., q[0,1], 0.],
        #                   [0., 0., q[0,0], 0., 0., q[0,1]],
        #                   [q[1,0], 0., 0., q[1,1], 0., 0.],
        #                   [0., q[1,0], 0., 0., q[1,1], 0.],
        #                   [0., 0., q[1,0], 0., 0., q[1,1]]])
        # matrix_list = []
        # for i in range (self.keypoint_number):
        #     matrix_list.append(block)
        # ekf.Q = scipy.linalg.block_diag(*matrix_list)

        ekf.Q = np.eye(self.x_dimension) * (self.Q_var)

        self.filter = ekf

    def initialize_state(self, positions_3D): # keypoints included in arguments
        # #Number of pixels in x and y direction where we still accept the filter output.
        # self.threshold = 200


        # set state vector according to the (18,3) Matrix of the player
        new_state = []
        for i in range (positions_3D.shape[0]):
            # x coordinate
            new_state.append(positions_3D[i,0])
            # y coordinate
            new_state.append(positions_3D[i,1])
            # z coordinate
            new_state.append(positions_3D[i,2])

            # x velocity
            new_state.append(5.)
            # y velocity (vertical up)
            new_state.append(0.)
            # z velocity
            new_state.append(5.)

        self.filter.x = np.array([new_state]).T


    # HJacobian
    # return the Jacobian matrix of the partial derivatives of Hx with respect to x
    def HJacobian_at(self, x, data_dict):
        matrix_list = []
        for k in range(self.keypoint_number):
            # [x,y,z,x',y',z']
            x_state = x[(k*6):((k+1)*6):1] # self.filter.x[(k*6):((k+1)*6):1]
            C_k = np.empty((0,6))
            for i in data_dict:
                # get actual camera matrices from camera i
                frame_name = data_dict[i].frame_basenames[self.frame_number]
                cam = cam_utils.Camera('tmp', data_dict[i].calib[frame_name]['A'], data_dict[i].calib[frame_name]
                                       ['R'], data_dict[i].calib[frame_name]['T'], data_dict[i].shape[0], data_dict[i].shape[1])

                # prepare fx * r_i,k and fy * r_i,k
                fx0_r11 = cam.A[0,0] * cam.R[0,0]
                fx0_r12 = cam.A[0,0] * cam.R[0,1]
                fx0_r13 = cam.A[0,0] * cam.R[0,2]
                fx0_r31 = cam.A[0,0] * cam.R[2,0]
                fx0_r32 = cam.A[0,0] * cam.R[2,1]
                fx0_r33 = cam.A[0,0] * cam.R[2,2]

                fy0_r21 = cam.A[1,1] * cam.R[1,0]
                fy0_r22 = cam.A[1,1] * cam.R[1,1]
                fy0_r23 = cam.A[1,1] * cam.R[1,2]
                fy0_r31 = cam.A[1,1] * cam.R[2,0]
                fy0_r32 = cam.A[1,1] * cam.R[2,1]
                fy0_r33 = cam.A[1,1] * cam.R[2,2]

                # get [x,y,z] as state vector
                x_ = x_state[0:3:1]
                # compute dot products from R matrix with the state vector
                ex0 = (cam.R[0, :].dot(x_) + cam.T[0,0])[0]
                ey0 = (cam.R[1, :].dot(x_) + cam.T[1,0])[0]
                ez0 = (cam.R[2, :].dot(x_) + cam.T[2,0])[0]
                ez0_sq = ez0 * ez0;

                # compute the jacobian
                dxi_dxiw = fx0_r11 / ez0 - fx0_r31 * ex0 / ez0_sq;
                dxi_dyiw = fx0_r12 / ez0 - fx0_r32 * ex0 / ez0_sq;
                dxi_dziw = fx0_r13 / ez0 - fx0_r33 * ex0 / ez0_sq;

                dyi_dxiw = fy0_r21 / ez0 - fy0_r31 * ey0 / ez0_sq;
                dyi_dyiw = fy0_r22 / ez0 - fy0_r32 * ey0 / ez0_sq;
                dyi_dziw = fy0_r23 / ez0 - fy0_r33 * ey0 / ez0_sq;

                # save the computed jacobian in a block matrix
                block = np.matrix([[dxi_dxiw, dxi_dyiw, dxi_dziw, 0, 0, 0],
                                   [dyi_dxiw, dyi_dyiw, dyi_dziw, 0, 0, 0]])
                C_k = np.vstack((C_k,block))


            matrix_list.append(C_k)

        H = scipy.linalg.block_diag(*matrix_list)
        return H


    # Hx
    # computes the corresponding measurement vector z from the given state x
    # [x_0, y_0, x_1, y_1, ... , x_4, y_4] for one keypoint
    # TODO: maybe rewrite with matrix multiplication for speedup
    def Hx(self, x, data_dict):
        matrix_list = []
        Hx = np.empty((0,1))
        for k in range(self.keypoint_number):
            # [x,y,z,x',y',z']
            x_state = x[(k*6):((k+1)*6):1]
            for i in data_dict:
                # get actual camera matrices from camera i
                frame_name = data_dict[i].frame_basenames[self.frame_number]
                cam = cam_utils.Camera('tmp', data_dict[i].calib[frame_name]['A'], data_dict[i].calib[frame_name]
                                       ['R'], data_dict[i].calib[frame_name]['T'], data_dict[i].shape[0], data_dict[i].shape[1])


                # get [x,y,z] as state vector
                x_ = x_state[0:3:1]
                # compute dot products from R matrix with the state vector
                ex0 = (cam.R[0, :].dot(x_) + cam.T[0,0])[0]
                ey0 = (cam.R[1, :].dot(x_) + cam.T[1,0])[0]
                ez0 = (cam.R[2, :].dot(x_) + cam.T[2,0])[0]

                # compute the actual 2D coordinates
                x_k = (cam.A[0,0] * ex0 / ez0) + cam.A[0,2]
                y_k = (cam.A[1,1] * ey0 / ez0) + cam.A[1,2]

                Hx = np.vstack((Hx, x_k))
                Hx = np.vstack((Hx, y_k))


        return Hx




    def update(self, z, data_dict): # keypoints was also included
        """do some sort of processing on the keypoints and then update the filter"""

        # TODO: idea is this: when we have no openpose measurement for a certain keypoint,
        # then we just go ahead and modify the R matrix accordingly, setting R diagonal very
        # high for the missing values and then passing the previous filter value as measurement
        # that way the filter should not really start overestimating the value of the predictions
        # that should allow us to skip a few frames and still have a reasonable estimate for the filter.

        ########################################################################
        # old logic
        ########################################################################
        # new_state = []
        #
        # #Stores the indexes of the values in the filter that should be reinitialized with the openpose output
        # re_initialize_indexes = []
        #
        # #Stores the indexes of the keypoints that did not get updated this time around
        # no_update = []
        #
        # # Get some information on the "quality of the measurement", so the number
        # # of keyposes that we have and their estimation score
        # mean_estimation_score = np.array(keypoints[0])[:,2].mean()
        # number_of_estimates = np.count_nonzero(np.array(keypoints[0])[:,0])
        #
        # print("Mean Score: {}".format(mean_estimation_score))
        # print("# Of Estimates: {}".format(number_of_estimates))
        #
        #
        # #Start looping over the keypoints and update with new measurement if it is good (aka not too far off)
        # for index,keypoint in enumerate(keypoints[0]):
        #
        #     x_openpose, y_openpose, confidence = keypoint[0], keypoint[1], keypoint[2]
        #     x_filter, y_filter = self.filter.x[index * 4][0], self.filter.x[index * 4 + 2][0]
        #
        #     delta_x, delta_y = abs(x_filter-x_openpose), abs(y_filter-y_openpose)
        #
        #     if confidence == 0:
        #         #no keypoint found by openpose this time
        #         no_update.append(index)
        #         x_openpose = x_filter
        #         y_openpose = y_filter
        #
        #         #Set R matrix very large for the no-update values to avoid that the filter starts learning
        #         new_R[index * 2,index * 2] = 10000.
        #         new_R[index * 2 + 1,index * 2 + 1] = 1000.
        #         new_state.append(x_openpose)
        #         new_state.append(y_openpose)
        #         continue
        #
        #     if (delta_x > self.threshold) or (delta_y > self.threshold):
        #
        #         if self.keypoints_no_update_count[index] < self.frames_without_update_allowed:
        #             x_openpose = x_filter
        #             y_openpose = y_filter
        #             no_update.append(index)
        #             new_R[index * 2,index * 2] = 10000.
        #             new_R[index * 2 + 1,index * 2 + 1] = 10000.
        #         else:
        #             self.keypoints_no_update_count[index] = 0
        #             re_initialize_indexes.append(index)
        #
        #     new_state.append(x_openpose)
        #     new_state.append(y_openpose)
        #
        # if (len(re_initialize_indexes) > 3) and (number_of_estimates > 20):
        #     self.initialize_state(keypoints)
        #
        #     print("Reinitializing Kalman Filter due to too much drift!: \n Mean Score: {}".format(mean_estimation_score,))
        #     return
        #
        # #update the list of no_updates
        # for index, value in enumerate(no_update):
        #     self.keypoints_no_update_count[value] += 1
        #
        #
        # self.filter._R = np.array(new_R)
        # self.filter.update(new_state)

        self.filter.update(z, HJacobian=self.HJacobian_at, Hx=self.Hx, args=(data_dict), hx_args=(data_dict))

    # Predict next state (prior) using the Kalman filter state propagation equations.
    # TODO: maybe rewrite predict_x to include discrete white noise
    def predict(self):
        """predicts the next state and returns the predicted next state"""

        self.filter.predict()
        return self.filter.x



    ############################################################################
    # Sympy example
    # maybe include to make sure Jacobian is right
    ############################################################################
    # example:
    # def hx(x):
    # x, y, z, x_vel, y_vel, z_vel = sympy.symbols('x y z x_vel y_vel z_vel')
    # # example:
    # X = sympy.Matrix([rho*cos(phi), rho*sin(phi), rho**2])
    # Y = sympy.Matrix([x, y, z, x_vel, y_vel, z_vel])
    # X.jacobian(Y)

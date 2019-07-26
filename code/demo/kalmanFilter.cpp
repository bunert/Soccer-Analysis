#include "tracking/kalmanFilter.hpp"
#include <iostream>
#include <Eigen/Dense>
#include <openpose/core/array.hpp>
#include "configuration/configuration.hpp"
#include "calibration/calibration.hpp"
#include "utils/io.hpp"
#include "utils/candidate.hpp"

KalmanFilter::KalmanFilter()
    : timestep(1.0 / CAM_FPS)
{
  std::cout << "kalmanFilter constructor()" << std::endl;
  // TODO: check what to init first (because of dependencies)
  startedTracking = false;

  // Set up system matrix A
  // TODO: might introduce sparsity
  A.resize(6 * NUM_BODY_PARTS, 6 * NUM_BODY_PARTS);
  A.setZero();
  A.diagonal().setOnes();
  for (int i = 0; i < A.rows(); i += 6)
  {
    A(i, i + 3) = timestep;
    A(i + 1, i + 4) = timestep;
    A(i + 2, i + 5) = timestep;
  }



  // Calibration Test event baku 2015 (Women C1)
  CameraIntrinsics0.resize(3, 3);
  CameraIntrinsics0 << 2.669617983051140e+03, 0, 959.500000000000,
      0, 2.541796700968023e+03, 539.500000000000,
      0, 0, 1;

  Rt0.resize(3, 4);
  Rt0 << 0.965562071598305, 0.260129997060510, 0.004719165189790, -0.038290929342975,
      -0.004461657028457, 0.034691440788671, -0.999388111572559, 2.184160346276195,
      -0.260134541165432, 0.964950200044181, 0.034657350264337, 10.507282692777736;

  KRt0 = CameraIntrinsics0 * Rt0;

  CameraIntrinsics1.resize(3, 3);
  CameraIntrinsics1 << 2.036746346957060e+03, 0, 959.500000000000,
      0, 1.947673555797361e+03, 539.500000000000,
      0, 0, 1;

  Rt1.resize(3, 4);
  Rt1 << 0.210723108434900, 0.977237273148637, 0.024557759273791, -1.789783453585593,
      0.052151770274695, 0.013847608667532, -0.998543157100086, 1.729055992957124,
      -0.976153658205894, 0.211696848590637, -0.048046642630304, 12.857394546191527;

  KRt1 = CameraIntrinsics1 * Rt1;

  P.resize(6 * NUM_BODY_PARTS, 6 * NUM_BODY_PARTS); //
  P.setZero();

  // TODO: Init Q
  Q.resize(6 * NUM_BODY_PARTS, 6 * NUM_BODY_PARTS);
  Q.setZero();
  Q.diagonal().setOnes();
  Q *= 0.3;

  // TODO: Init R
  R.resize(2 * NUM_CAMERAS * NUM_BODY_PARTS, 2 * NUM_CAMERAS * NUM_BODY_PARTS); //
  R.setZero();
  R.diagonal().setOnes();
  R *= 0.025;

  // Init C (Jacobian of h)
  C.resize(2 * NUM_CAMERAS * NUM_BODY_PARTS, 6 * NUM_BODY_PARTS);
  C.setZero();

  K.resize(6 * NUM_BODY_PARTS, 2 * NUM_CAMERAS * NUM_BODY_PARTS); // Kalman gain
  K.setZero();                                                    // Kalman gain

  x.resize(6 * NUM_BODY_PARTS);

  y.resize(2 * NUM_CAMERAS * NUM_BODY_PARTS);
  hx.resize(2 * NUM_CAMERAS * NUM_BODY_PARTS);

  // Data
  x.setZero();
  y.setZero();
  hx.setZero();

  for (int camId = 0; camId < NUM_CAMERAS; camId++)
  {
    scores.push_back(Eigen::VectorXi());
    scores[camId].resize(NUM_BODY_PARTS);
    scores[camId].setZero();
  }

  state = status::after_initialization;
}

// private functions //

Eigen::VectorXf KalmanFilter::h_x()
{
  Eigen::VectorXf yEstim(2 * NUM_CAMERAS * NUM_BODY_PARTS);
  Eigen::VectorXf xWorld(4);
  xWorld(3) = 1.0;
  for (int i = 0; i < x.rows(); i += 6)
  {
    xWorld.segment(0, 3) = x.segment(i, 3);
    // Project world coordinates on image planes
    Eigen::VectorXf x1 = KRt0 * xWorld;
    Eigen::VectorXf x2 = KRt1 * xWorld;

    int idx = i / 6 * (2 + NUM_CAMERAS);
    // divide by z
    yEstim(idx) = x1(0) / x1(2);
    yEstim(idx + 1) = x1(1) / x1(2);
    yEstim(idx + 2) = x2(0) / x2(2);
    yEstim(idx + 3) = x2(1) / x2(2);
  }
  return yEstim;
}

cv::Point2f KalmanFilter::h(cv::Point3f worldPoint, int camId)
{
  Eigen::Vector4f xWorld(4);
  xWorld << worldPoint.x, worldPoint.y, worldPoint.z, 1.0f;

  Eigen::Vector3f p;
  if (camId == CAM_ID_SIDE)
    p = KRt0 * xWorld;
  else if (camId == CAM_ID_FRONT)
    p = KRt1 * xWorld;

  return cv::Point2f(p(0) / p(2), p(1) / p(2));
}

op::Array<float> KalmanFilter::h_x(int camId)
{
  Eigen::VectorXf yAll = h_x();
  op::Array<float> yCamId({1, NUM_BODY_PARTS, SIZE_ENTRY_2D});
  float score = 0.5f; //SCORE_THRESHOLD + 0.01f;
  for (int part = 0; part < NUM_BODY_PARTS; part++)
  {
    int arrayIndex = part * SIZE_ENTRY_2D;
    int vectorIndex = part * NUM_CAMERAS * 2 + camId * 2;
    if (scores[CAM_ID_SIDE](part) > 0 && scores[CAM_ID_FRONT](part) > 0)
    {
      yCamId[arrayIndex] = yAll(vectorIndex);
      yCamId[arrayIndex + 1] = yAll(vectorIndex + 1);
      yCamId[arrayIndex + 2] = score;
    }
    else
    {
      yCamId[arrayIndex] = 0.0f;
      yCamId[arrayIndex + 1] = 0.0f;
      yCamId[arrayIndex + 2] = 0.0f;
    }
  }
  return yCamId;
}

void KalmanFilter::updateQ()
{
  // TODO or in updateR() consider predScores
  // the error might also come from predicting with invalid values because we dont hanve any depth in the kalman filter and set x to 0, do we?
}

void KalmanFilter::updateR()
{
  // TODO
}

void KalmanFilter::updateC()
{
  Eigen::MatrixXf Ci(2 * NUM_CAMERAS, 3); // contains the nonzero entries

  // prepare factors to be reused
  double fx0_r110 = CameraIntrinsics0(0, 0) * Rt0(0, 0);
  double fx0_r120 = CameraIntrinsics0(0, 0) * Rt0(0, 1);
  double fx0_r130 = CameraIntrinsics0(0, 0) * Rt0(0, 2);
  double fx0_r310 = CameraIntrinsics0(0, 0) * Rt0(2, 0);
  double fx0_r320 = CameraIntrinsics0(0, 0) * Rt0(2, 1);
  double fx0_r330 = CameraIntrinsics0(0, 0) * Rt0(2, 2);

  double fy0_r210 = CameraIntrinsics0(1, 1) * Rt0(1, 0);
  double fy0_r220 = CameraIntrinsics0(1, 1) * Rt0(1, 1);
  double fy0_r230 = CameraIntrinsics0(1, 1) * Rt0(1, 2);
  double fy0_r310 = CameraIntrinsics0(1, 1) * Rt0(2, 0);
  double fy0_r320 = CameraIntrinsics0(1, 1) * Rt0(2, 1);
  double fy0_r330 = CameraIntrinsics0(1, 1) * Rt0(2, 2);

  double fx1_r111 = CameraIntrinsics1(0, 0) * Rt1(0, 0);
  double fx1_r121 = CameraIntrinsics1(0, 0) * Rt1(0, 1);
  double fx1_r131 = CameraIntrinsics1(0, 0) * Rt1(0, 2);
  double fx1_r311 = CameraIntrinsics1(0, 0) * Rt1(2, 0);
  double fx1_r321 = CameraIntrinsics1(0, 0) * Rt1(2, 1);
  double fx1_r331 = CameraIntrinsics1(0, 0) * Rt1(2, 2);

  double fy1_r211 = CameraIntrinsics1(1, 1) * Rt1(1, 0);
  double fy1_r221 = CameraIntrinsics1(1, 1) * Rt1(1, 1);
  double fy1_r231 = CameraIntrinsics1(1, 1) * Rt1(1, 2);
  double fy1_r311 = CameraIntrinsics1(1, 1) * Rt1(2, 0);
  double fy1_r321 = CameraIntrinsics1(1, 1) * Rt1(2, 1);
  double fy1_r331 = CameraIntrinsics1(1, 1) * Rt1(2, 2);

  // compute and fill C
  for (int i = 0; i < NUM_BODY_PARTS; i++)
  {
    // compute indices of Ci
    int ix = i * NUM_CAMERAS * 2;
    int iy = i * 6;

    // reusable terms // TODO: check which x you've derive to
    Eigen::VectorXf x_ = x.segment(iy, 3);
    double ex0 = Rt0.block<1, 3>(0, 0).dot(x_) + Rt0(0, 3);
    double ey0 = Rt0.block<1, 3>(1, 0).dot(x_) + Rt0(1, 3);
    double ez0 = Rt0.block<1, 3>(2, 0).dot(x_) + Rt0(2, 3);
    double ez0_sq = ez0 * ez0;
    double ex1 = Rt1.block<1, 3>(0, 0).dot(x_) + Rt1(0, 3);
    double ey1 = Rt1.block<1, 3>(1, 0).dot(x_) + Rt1(1, 3);
    double ez1 = Rt1.block<1, 3>(2, 0).dot(x_) + Rt1(2, 3);
    double ez1_sq = ez1 * ez1;

    // TODO: might wanna introduce vector operations
    double dxi0_dxiw = fx0_r110 / ez0 - fx0_r310 * ex0 / ez0_sq;
    double dxi0_dyiw = fx0_r120 / ez0 - fx0_r320 * ex0 / ez0_sq;
    double dxi0_dziw = fx0_r130 / ez0 - fx0_r330 * ex0 / ez0_sq;
    double dyi0_dxiw = fy0_r210 / ez0 - fy0_r310 * ey0 / ez0_sq;
    double dyi0_dyiw = fy0_r220 / ez0 - fy0_r320 * ey0 / ez0_sq;
    double dyi0_dziw = fy0_r230 / ez0 - fy0_r330 * ey0 / ez0_sq;
    double dxi1_dxiw = fx1_r111 / ez1 - fx1_r311 * ex1 / ez1_sq;
    double dxi1_dyiw = fx1_r121 / ez1 - fx1_r321 * ex1 / ez1_sq;
    double dxi1_dziw = fx1_r131 / ez1 - fx1_r331 * ex1 / ez1_sq;
    double dyi1_dxiw = fy1_r211 / ez1 - fy1_r311 * ey1 / ez1_sq;
    double dyi1_dyiw = fy1_r221 / ez1 - fy1_r321 * ey1 / ez1_sq;
    double dyi1_dziw = fy1_r231 / ez1 - fy1_r331 * ey1 / ez1_sq;

    Ci << dxi0_dxiw, dxi0_dyiw, dxi0_dziw,
        dyi0_dxiw, dyi0_dyiw, dyi0_dziw,
        dxi1_dxiw, dxi1_dyiw, dxi1_dziw,
        dyi1_dxiw, dyi1_dyiw, dyi1_dziw;
    C.block<2 * NUM_CAMERAS, 3>(ix, iy) = Ci;
  }
}

void KalmanFilter::predict()
{
  if (state == status::after_initialization)
    std::cout << x(1) << std::endl;

  std::cout << "kalmanFilter predict()" << std::endl;
  x = A * x;
  P = A * P * A.transpose() + Q;
  if (state == status::after_initialization)
    std::cout << x(1) << std::endl;

  state = status::after_prediction;
}

void KalmanFilter::innovate()
{
  std::cout << "kalmanFilter innovate()" << std::endl;
  updateQ(); // TODO: check where to update Q
  updateC();
  updateR();
  K = P * C.transpose() * (C * P * C.transpose() + R).inverse();
  x = x + K * (y - h_x());
  P = P - K * C * P;

  state = status::after_innovation;
}

void KalmanFilter::init_X(Candidate2D cSide)
{
  std::cout << "kalmanFilter init_X()" << std::endl;
  bool upsideDown;
  float angle = cSide.getRotationDegree();
  if (angle > 90.0f && angle < 270.0f)
    upsideDown = true;

  // for simplicity init athlete as line with constant forward speed
  float px = -2.5f;
  float py = 1.25f;
  float pz = 0.0f; // init below
  // init with forward speed of 5 m/s
  float vx = 5.0f;
  float vy = 0.0f;
  float vz = 0.0f;
  for (int i = 0; i < NUM_BODY_PARTS; i++)
  {
    int baseIndex = i * 6;
    x(baseIndex) = px;
    x(baseIndex + 1) = py;
    x(baseIndex + 2) = pz;
    x(baseIndex + 3) = vx;
    x(baseIndex + 4) = vy;
    x(baseIndex + 5) = vz;
  }

  // adjust heights of different body parts
  if (upsideDown)
  {
    int idxZ;
    float height;

    // hands
    height = 0.0f;
    idxZ = 4 * 6 + 2;
    x(idxZ) = height;
    idxZ = 7 * 6 + 2;
    x(idxZ) = height;

    // elbows
    height = 0.25f;
    idxZ = 3 * 6 + 2;
    x(idxZ) = height;
    idxZ = 6 * 6 + 2;
    x(idxZ) = height;

    // head
    height = 0.4f;
    idxZ = 0 * 6 + 2;
    x(idxZ) = height;
    idxZ = 15 * 6 + 2;
    x(idxZ) = height;
    idxZ = 16 * 6 + 2;
    x(idxZ) = height;
    idxZ = 17 * 6 + 2;
    x(idxZ) = height;
    idxZ = 18 * 6 + 2;
    x(idxZ) = height;

    // shoulders
    height = 0.6f;
    idxZ = 2 * 6 + 2;
    x(idxZ) = height;
    idxZ = 1 * 6 + 2;
    x(idxZ) = height;
    idxZ = 5 * 6 + 2;
    x(idxZ) = height;

    // hip
    height = 1.10f;
    idxZ = 8 * 6 + 2;
    x(idxZ) = height;
    idxZ = 9 * 6 + 2;
    x(idxZ) = height;
    idxZ = 12 * 6 + 2;
    x(idxZ) = height;

    // knees
    height = 1.40f;
    idxZ = 10 * 6 + 2;
    x(idxZ) = height;
    idxZ = 13 * 6 + 2;
    x(idxZ) = height;

    // feet
    height = 1.80f;
    idxZ = 11 * 6 + 2;
    x(idxZ) = height;
    idxZ = 14 * 6 + 2;
    x(idxZ) = height;
    idxZ = 19 * 6 + 2;
    x(idxZ) = height;
    idxZ = 20 * 6 + 2;
    x(idxZ) = height;
    idxZ = 21 * 6 + 2;
    x(idxZ) = height;
    idxZ = 22 * 6 + 2;
    x(idxZ) = height;
    idxZ = 23 * 6 + 2;
    x(idxZ) = height;
    idxZ = 24 * 6 + 2;
    x(idxZ) = height;
  }
  else
  {
    int idxZ;
    float height;
    // head
    height = 1.5f;
    idxZ = 0 * 6 + 2;
    x(idxZ) = height;
    idxZ = 15 * 6 + 2;
    x(idxZ) = height;
    idxZ = 16 * 6 + 2;
    x(idxZ) = height;
    idxZ = 17 * 6 + 2;
    x(idxZ) = height;
    idxZ = 18 * 6 + 2;
    x(idxZ) = height;

    // shoulders
    height = 1.40f;
    idxZ = 2 * 6 + 2;
    x(idxZ) = height;
    idxZ = 1 * 6 + 2;
    x(idxZ) = height;
    idxZ = 5 * 6 + 2;
    x(idxZ) = height;

    // elbows
    height = 1.20f;
    idxZ = 3 * 6 + 2;
    x(idxZ) = height;
    idxZ = 6 * 6 + 2;
    x(idxZ) = height;

    // hip
    height = 1.00f;
    idxZ = 4 * 6 + 2;
    x(idxZ) = height;
    idxZ = 7 * 6 + 2;
    x(idxZ) = height;
    idxZ = 8 * 6 + 2;
    x(idxZ) = height;
    idxZ = 9 * 6 + 2;
    x(idxZ) = height;
    idxZ = 12 * 6 + 2;
    x(idxZ) = height;

    // knees
    height = 0.50f;
    idxZ = 10 * 6 + 2;
    x(idxZ) = height;
    idxZ = 13 * 6 + 2;
    x(idxZ) = height;

    // feet
    height = 0.00f;
    idxZ = 11 * 6 + 2;
    x(idxZ) = height;
    idxZ = 14 * 6 + 2;
    x(idxZ) = height;
    idxZ = 19 * 6 + 2;
    x(idxZ) = height;
    idxZ = 20 * 6 + 2;
    x(idxZ) = height;
    idxZ = 21 * 6 + 2;
    x(idxZ) = height;
    idxZ = 22 * 6 + 2;
    x(idxZ) = height;
    idxZ = 23 * 6 + 2;
    x(idxZ) = height;
    idxZ = 24 * 6 + 2;
    x(idxZ) = height;
  }

  for (int camId = 0; camId < NUM_CAMERAS; camId++)
  {
    scores[camId].array() = depth;
  }
}

void KalmanFilter::set_y(const float *athletePtr, int camId)
{
  for (int part = 0; part < NUM_BODY_PARTS; part++)
  {
    int baseIndex = part * SIZE_ENTRY_2D;
    // calculate offset of x coordinate of specific joint and specific cam in y
    int offset = part * NUM_CAMERAS * 2 + camId * 2;

    if (athletePtr != nullptr && athletePtr[baseIndex + 2] > SCORE_THRESHOLD)
    {
      // update score
      scores[camId](part) = depth;
      // fill in x coord of specific cam
      y(offset) = athletePtr[baseIndex];
      // fill in y coord of specific cam
      y(offset + 1) = athletePtr[baseIndex + 1];
    }
    else
    {
      // exclude this body part from calculations in this round
      scores[camId](part) -= 1;
      // project predicted x to imageplane
      int idx = part * 6;
      cv::Point3f worldPoint(x(idx), x(idx + 1), x(idx + 2));
      cv::Point2f imagePoint = h(worldPoint, camId);
      y(offset) = imagePoint.x;
      y(offset + 1) = imagePoint.y;
    }
  }
}

void KalmanFilter::set_y(op::Array<float> pose, int camId)
{
  set_y(pose.getConstPtr(), camId);
}

void KalmanFilter::set_y(Candidate2D c)
{
  const float *athletePtr = c.getPosePtr();
  int camId = c.getCamId();
  set_y(athletePtr, camId);
}

op::Array<float> KalmanFilter::get_X()
{
  op::Array<float> poseAthlete({1, NUM_BODY_PARTS, SIZE_ENTRY_3D});
  float score = 0.5f; //SCORE_THRESHOLD + 0.01f;
  for (int part = 0; part < NUM_BODY_PARTS; part++)
  {
    int arrayIndex = part * SIZE_ENTRY_3D;
    int vectorIndex = part * 6;
    if (scores[CAM_ID_SIDE](part) > 0 && scores[CAM_ID_FRONT](part) > 0)
    {
      poseAthlete[arrayIndex] = x(vectorIndex);
      poseAthlete[arrayIndex + 1] = x(vectorIndex + 1);
      poseAthlete[arrayIndex + 2] = x(vectorIndex + 2);
      poseAthlete[arrayIndex + 3] = score;
    }
    else
    {
      poseAthlete[arrayIndex] = 0.0f;
      poseAthlete[arrayIndex + 1] = 0.0f;
      poseAthlete[arrayIndex + 2] = 0.0f;
      poseAthlete[arrayIndex + 3] = 0.0f;
    }
  }
  return poseAthlete;
}

#include "controller_ae483.h"
#include "stabilizer_types.h"
#include "power_distribution.h"
#include "log.h"
#include "param.h"
#include "num.h"
#include "math3d.h"

// Sensor measurements
// - tof (from the z ranger on the flow deck)
static uint16_t tof_count = 0;
static float tof_distance = 0.0f;
// - flow (from the optical flow sensor on the flow deck)
static uint16_t flow_count = 0;
static float flow_dpixelx = 0.0f;
static float flow_dpixely = 0.0f;

// Other sensor measurements
static float n_x = 0.0f;
static float n_y = 0.0f;
static float r = 0.0f;
static float a_z = 0.0f;

static float p_x_mocap = 0.0f;
static float p_y_mocap = 0.0f;
static float p_z_mocap = 0.0f;
static float psi_mocap = 0.0f;
static float theta_mocap = 0.0f;
static float phi_mocap = 0.0f;

// Constants
static float k_flow = 4.09255568f;
static float g = 9.81f;
static float p_z_eq = 0.5f;

// Measurement errors
static float n_x_err = 0.0f;
static float n_y_err = 0.0f;
static float r_err = 0.0f;
static float p_x_mocap_err = 0.0f;
static float p_y_mocap_err = 0.0f;
static float p_z_mocap_err = 0.0f;
static float psi_mocap_err = 0.0f;
static float theta_mocap_err = 0.0f;
static float phi_mocap_err = 0.0f;

// Parameters
static bool use_observer = false;
static bool reset_observer = false;

static float p_x = 0.0f;
static float p_y = 0.0f;
static float p_z = 0.0f;
static float psi = 0.0f;
static float theta = 0.0f;
static float phi = 0.0f;
static float v_x = 0.0f;
static float v_y = 0.0f;
static float v_z = 0.0f;
static float w_x = 0.0f;
static float w_y = 0.0f;
static float w_z = 0.0f;

static float tau_x_cmd = 0.0f;    // tau_x command
static float tau_y_cmd = 0.0f;    // tau_y command
static float w_x_old = 0.0f;      // value of w_x from previous time step
static float w_y_old = 0.0f;      // value of w_y from previous time step
static float J_x = 1.33e-5f;      // principal moment of inertia about x_B axis
static float J_y = 1.80e-5f;      // principal moment of inertia about y_B axis
static float dt = 0.002f;         // time step (corresponds to 500 Hz)

// Setpoint
static float p_x_des = 0.0f;
static float p_y_des = 0.0f;
static float p_z_des = 0.0f;

// Input
static float tau_x = 0.0f;
static float tau_y = 0.0f;
static float tau_z = 0.0f;
static float f_z = 0.0f;

// Motor power command
static uint16_t m_1 = 0;
static uint16_t m_2 = 0;
static uint16_t m_3 = 0;
static uint16_t m_4 = 0;


void ae483UpdateWithTOF(tofMeasurement_t *tof)
{
  tof_distance = tof->distance;
  tof_count++;
}

void ae483UpdateWithFlow(flowMeasurement_t *flow)
{
  flow_dpixelx = flow->dpixelx;
  flow_dpixely = flow->dpixely;
  flow_count++;
}

void ae483UpdateWithDistance(distanceMeasurement_t *meas)
{
  // If you have a loco positioning deck, this function will be called
  // each time a distance measurement is available. You will have to write
  // code to handle these measurements. These data are available:
  //
  //  meas->anchorId  uint8_t   id of anchor with respect to which distance was measured
  //  meas->x         float     x position of this anchor
  //  meas->y         float     y position of this anchor
  //  meas->z         float     z position of this anchor
  //  meas->distance  float     the measured distance
}

void ae483UpdateWithPosition(positionMeasurement_t *meas)
{
  // This function will be called each time you send an external position
  // measurement (x, y, z) from the client, e.g., from a motion capture system.
  // You will have to write code to handle these measurements. These data are
  // available:
  //
  //  meas->x         float     x component of external position measurement
  //  meas->y         float     y component of external position measurement
  //  meas->z         float     z component of external position measurement
}

void ae483UpdateWithPose(poseMeasurement_t *meas)
{
  // This function will be called each time you send an external "pose" measurement
  // (position as x, y, z and orientation as quaternion) from the client, e.g., from
  // a motion capture system. You will have to write code to handle these measurements.
  // These data are available:
  //
  //  meas->x         float     x component of external position measurement
  //  meas->y         float     y component of external position measurement
  //  meas->z         float     z component of external position measurement
  //  meas->quat.x    float     x component of quaternion from external orientation measurement
  //  meas->quat.y    float     y component of quaternion from external orientation measurement
  //  meas->quat.z    float     z component of quaternion from external orientation measurement
  //  meas->quat.w    float     w component of quaternion from external orientation measurement

  // Position
  p_x_mocap = meas->x;
  p_y_mocap = meas->y;
  p_z_mocap = meas->z;

  // Orientation
  // - Create a quaternion from its parts
  struct quat q_mocap = mkquat(meas->quat.x, meas->quat.y, meas->quat.z, meas->quat.w);
  // - Convert the quaternion to a vector with yaw, pitch, and roll angles
  struct vec rpy_mocap = quat2rpy(q_mocap);
  // - Extract the yaw, pitch, and roll angles from the vector
  psi_mocap = rpy_mocap.z;
  theta_mocap = rpy_mocap.y;
  phi_mocap = rpy_mocap.x;
}

void ae483UpdateWithData(const struct AE483Data* data)
{
  // This function will be called each time AE483-specific data are sent
  // from the client to the drone. You will have to write code to handle
  // these data. For the example AE483Data struct, these data are:
  //
  //  data->x         float
  //  data->y         float
  //  data->z         float
  //
  // Exactly what "x", "y", and "z" mean in this context is up to you.
}


void controllerAE483Init(void)
{
  // Do nothing
}

bool controllerAE483Test(void)
{
  // Do nothing (test is always passed)
  return true;
}

void controllerAE483(control_t *control,
                     const setpoint_t *setpoint,
                     const sensorData_t *sensors,
                     const state_t *state,
                     const stabilizerStep_t stabilizerStep)
{
  // Make sure this function only runs at 500 Hz
  if (!RATE_DO_EXECUTE(ATTITUDE_RATE, stabilizerStep)) {
    return;
  }

  // Desired position
  p_x_des = setpoint->position.x;
  p_y_des = setpoint->position.y;
  p_z_des = setpoint->position.z;

  // Measurements
  w_x = radians(sensors->gyro.x);
  w_y = radians(sensors->gyro.y);
  w_z = radians(sensors->gyro.z);
  a_z = g * sensors->acc.z;
  n_x = flow_dpixelx;
  n_y = flow_dpixely;
  r = tof_distance;

  // Torques by finite difference
  tau_x = J_x * (w_x - w_x_old) / dt;
  tau_y = J_y * (w_y - w_y_old) / dt;
  w_x_old = w_x;
  w_y_old = w_y;



if (use_observer) {
  
    // Compute each element of:
    // 
    //   C x + D u - y
    // 
    n_x_err = k_flow * (v_x / p_z_eq - w_y) - n_x;
    n_y_err = k_flow * (v_y / p_z_eq + w_x) - n_y;
    r_err = (p_z-p_z_eq) - (r - p_z_eq);
    p_x_mocap_err = p_x - p_x_mocap;
    p_y_mocap_err = p_y - p_y_mocap;
    p_z_mocap_err = p_z - p_z_mocap;
    psi_mocap_err = psi - psi_mocap;
    theta_mocap_err = theta - theta_mocap;
    phi_mocap_err = phi - phi_mocap;

    // Update estimates
    p_x += dt * (v_x - 0.0009426021959269609f * n_x_err - 16.404197469896395f * p_x_mocap_err - 0.20996734690032678f * theta_mocap_err);
    p_y += dt * (v_y - 0.0004724542189137248f * n_y_err - 19.284667266477435f * p_y_mocap_err + 0.16816603961823223f * phi_mocap_err);
    p_z += dt * (v_z - 28.47504022199136f * r_err - 53.58250076858139f * p_z_mocap_err);
    psi += dt * (w_z - 10.326989767918791f * psi_mocap_err);
    theta += dt * (w_y - 0.002176067045272885f * n_x_err - 9.416918941446607f * p_x_mocap_err - 3.0332006746694167f * theta_mocap_err);
    phi += dt * (w_x + 0.0014187544023920992f * n_y_err + 12.098588349202796f * p_y_mocap_err - 2.450140196937766f * phi_mocap_err);
    v_x += dt * (g * theta - 0.01494505973243907f * n_x_err - 79.9036916310318f * p_x_mocap_err - 4.11295159067612f * theta_mocap_err);
    v_y += dt * (-g * phi - 0.008745650799790212f * n_y_err - 87.90745735903049f * p_y_mocap_err + 3.6692457981658584f * phi_mocap_err);
    v_z += dt * (a_z - g - 139.913014949133f * r_err - 263.279320155502f * p_z_mocap_err);
    
  } else {
    p_x = state->position.x;
    p_y = state->position.y;
    p_z = state->position.z;
    psi = radians(state->attitude.yaw);
    theta = - radians(state->attitude.pitch);
    phi = radians(state->attitude.roll);
    v_x = state->velocity.x*cosf(psi)*cosf(theta) + state->velocity.y*sinf(psi)*cosf(theta) - state->velocity.z*sinf(theta);
    v_y = state->velocity.x*(sinf(phi)*sinf(theta)*cosf(psi) - sinf(psi)*cosf(phi)) + state->velocity.y*(sinf(phi)*sinf(psi)*sinf(theta) + cosf(phi)*cosf(psi)) + state->velocity.z*sinf(phi)*cosf(theta);
    v_z = state->velocity.x*(sinf(phi)*sinf(psi) + sinf(theta)*cosf(phi)*cosf(psi)) + state->velocity.y*(-sinf(phi)*cosf(psi) + sinf(psi)*sinf(theta)*cosf(phi)) + state->velocity.z*cosf(phi)*cosf(theta);
  }
  

  if (setpoint->mode.z == modeDisable) {
    // If there is no desired position, then all
    // motor power commands should be zero

    m_1 = 0;
    m_2 = 0;
    m_3 = 0;
    m_4 = 0;
  } else {
    // Otherwise, motor power commands should be
    // chosen by the controller

    // FIXME (CONTROLLER GOES HERE)
    tau_x_cmd = 0.01148400f * (p_y - p_y_des) -0.00973167f * phi + 0.00504205f * v_y -0.00104432f * w_x -2.57534176f * tau_x;
    tau_y_cmd = -0.01148400f * (p_x - p_x_des) -0.01030994f * theta -0.00517457f * v_x -0.00115652f * w_y -2.26230944f * tau_y;
    tau_z = -0.00259115f * psi -0.00044530f * w_z;
    f_z = -0.42621122f * (p_z - p_z_des) -0.17307770f * v_z + 0.34335000f;

    // FIXME (METHOD OF POWER DISTRIBUTION GOES HERE)
    m_1 = limitUint16( -4353883.7f * tau_x_cmd -4353883.7f * tau_y_cmd -33422459.9f * tau_z + 143678.2f * f_z );
    m_2 = limitUint16( -4353883.7f * tau_x_cmd + 4353883.7f * tau_y_cmd + 33422459.9f * tau_z + 143678.2f * f_z );
    m_3 = limitUint16( 4353883.7f * tau_x_cmd + 4353883.7f * tau_y_cmd -33422459.9f * tau_z + 143678.2f * f_z );
    m_4 = limitUint16( 4353883.7f * tau_x_cmd -4353883.7f * tau_y_cmd + 33422459.9f * tau_z + 143678.2f * f_z );
  }

  // Apply motor power commands
  control->m1 = m_1;
  control->m2 = m_2;
  control->m3 = m_3;
  control->m4 = m_4;
}

//              1234567890123456789012345678 <-- max total length
//              group   .name
LOG_GROUP_START(ae483log)
LOG_ADD(LOG_UINT16,      num_tof,                &tof_count)
LOG_ADD(LOG_UINT16,      num_flow,               &flow_count)
LOG_GROUP_STOP(ae483log)

//                1234567890123456789012345678 <-- max total length
//                group   .name
PARAM_GROUP_START(ae483par)
PARAM_ADD(PARAM_UINT8,     use_observer,            &use_observer)
PARAM_ADD(PARAM_UINT8,     reset_observer,          &reset_observer)
LOG_ADD(LOG_FLOAT,       p_x,                    &p_x)
LOG_ADD(LOG_FLOAT,       p_y,                    &p_y)
LOG_ADD(LOG_FLOAT,       p_z,                    &p_z)
LOG_ADD(LOG_FLOAT,       psi,                    &psi)
LOG_ADD(LOG_FLOAT,       theta,                  &theta)
LOG_ADD(LOG_FLOAT,       phi,                    &phi)
LOG_ADD(LOG_FLOAT,       v_x,                    &v_x)
LOG_ADD(LOG_FLOAT,       v_y,                    &v_y)
LOG_ADD(LOG_FLOAT,       v_z,                    &v_z)
LOG_ADD(LOG_FLOAT,       w_x,                    &w_x)
LOG_ADD(LOG_FLOAT,       w_y,                    &w_y)
LOG_ADD(LOG_FLOAT,       w_z,                    &w_z)
LOG_ADD(LOG_FLOAT,       p_x_des,                &p_x_des)
LOG_ADD(LOG_FLOAT,       p_y_des,                &p_y_des)
LOG_ADD(LOG_FLOAT,       p_z_des,                &p_z_des)
LOG_ADD(LOG_FLOAT,       tau_x,                  &tau_x)
LOG_ADD(LOG_FLOAT,       tau_y,                  &tau_y)
LOG_ADD(LOG_FLOAT,       tau_z,                  &tau_z)
LOG_ADD(LOG_FLOAT,       f_z,                    &f_z)
LOG_ADD(LOG_UINT16,      m_1,                    &m_1)
LOG_ADD(LOG_UINT16,      m_2,                    &m_2)
LOG_ADD(LOG_UINT16,      m_3,                    &m_3)
LOG_ADD(LOG_UINT16,      m_4,                    &m_4)
LOG_ADD(LOG_FLOAT,       r,                      &r)
LOG_ADD(LOG_FLOAT,       n_x,                    &n_x)
LOG_ADD(LOG_FLOAT,       n_y,                    &n_y)
LOG_ADD(LOG_FLOAT,       a_z,                    &a_z)
LOG_ADD(LOG_FLOAT,       p_x_mocap,              &p_x_mocap)
LOG_ADD(LOG_FLOAT,       p_y_mocap,              &p_y_mocap)
LOG_ADD(LOG_FLOAT,       p_z_mocap,              &p_z_mocap)
LOG_ADD(LOG_FLOAT,       psi_mocap,              &psi_mocap)
LOG_ADD(LOG_FLOAT,       theta_mocap,            &theta_mocap)
LOG_ADD(LOG_FLOAT,       phi_mocap,              &phi_mocap)
PARAM_GROUP_STOP(ae483par)

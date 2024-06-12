import math
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
np.set_printoptions(precision=4)
plt.rcParams["savefig.facecolor"] = "0.8"
import os.path

# constants
eps = 1e-10


def xform_rot_dist(x1, x2):
    """[summary]

    Args:
        x1 ([type]): [description]
        x2 ([type]): [description]

    Returns:
        [type]: [description]
    """
    return quat_dist(get_quat(x1), get_quat(x2))


def xform_dist(x1, x2):
    """[summary]

    Args:
        x1 ([type]): [description]
        x2 ([type]): [description]

    Returns:
        [type]: [description]
    """
    return pos_dist(x1.translation(), x2.translation()) + quat_dist(
        get_quat(x1), get_quat(x2)
    )


def quat_dist(q1, q2):
    """[summary]

    Args:
        q1 ([type]): [description]
        q2 ([type]): [description]

    Returns:
        [type]: [description]
    """

    q1_dot_q2 = np.dot(q1.wxyz(), q2.wxyz())

    # https://math.stackexchange.com/a/90098

    # distance w/o trig functions
    # return np.sqrt(np.power(1 - q1_dot_q2 * q1_dot_q2, 2))

    # angular distance
    return np.arccos(2 * np.power(q1_dot_q2, 2) - 1)


def pos_dist(p1, p2):
    """[summary]

    Args:
        p1 ([type]): [description]
        p2 ([type]): [description]

    Returns:
        [type]: [description]
    """
    return np.linalg.norm(p1 - p2)


def get_rpy(xform):
    """[summary]

    Args:
        xform ([type]): [description]

    Returns:
        [type]: [description]
    """
    return RollPitchYaw(xform.rotation()).vector()


def get_quat(xform):
    """[summary]

    Args:
        xform ([type]): [description]

    Returns:
        [type]: [description]
    """
    return xform.rotation().ToQuaternion()


def rotmat_to_unit_spherical(rotmat):
    z_axis = rotmat[:, 2]
    # print(f"z_axis: {z_axis}")
    x, y, z = z_axis.tolist()
    r = np.linalg.norm(z_axis)
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)
    if theta > 0:
        # phi = phi  # do nothing
        pass
    elif theta < 0:
        phi += np.pi
    else:
        phi = 0
    return np.array([theta, phi])


def unitspherical_to_rotmat(theta, phi):
    # from scipy.spatial.transform import Rotation as R
    # RPhi = R.from_euler('xyz', [0,0,phi])
    # RTheta = R.from_euler('xyz', [theta,0,0])
    # RTotal = RTheta * RPhi
    # return RTotal.as_dcm()

    r_phi = RotationMatrix(RollPitchYaw(0, 0, phi))
    r_theta = RotationMatrix(RollPitchYaw(0, theta, 0))
    return (r_phi @ r_theta).matrix()


def xform_to_xyzrpy(xform):
    """[summary]

    Args:
        xform ([type]): [description]

    Returns:
        [type]: [description]
    """
    xyzrpy = np.zeros(6)
    xyzrpy[:3] = xform.translation()
    # xyzrpy[:3] *= 1000  # m to mm
    xyzrpy[3:] = get_rpy(xform)
    xyzrpy[3:] *= 180 / np.pi  # rad to deg
    return xyzrpy


def xform_to_xyzquat(xform):
    xyzq = np.zeros(7)
    xyzq[:3] = xform.translation()
    q_wxyz = get_quat(xform)
    xyzq[3] = q_wxyz.w()
    xyzq[4] = q_wxyz.x()
    xyzq[5] = q_wxyz.y()
    xyzq[6] = q_wxyz.z()
    return xyzq


def xform_to_xyzquat_xyzw(xform, m_to_mm=False):
    xyzq = np.zeros(7)
    xyzq[:3] = xform.translation()
    if m_to_mm:
        xyzq[:3] *= 1000
    q_wxyz = get_quat(xform)
    xyzq[3] = q_wxyz.x()
    xyzq[4] = q_wxyz.y()
    xyzq[5] = q_wxyz.z()
    xyzq[6] = q_wxyz.w()
    return xyzq


def xform_to_pose(X_AB):
    pose = Pose()
    xyz_AB = X_AB.translation()
    pose.position.x = xyz_AB[0]
    pose.position.y = xyz_AB[1]
    pose.position.z = xyz_AB[2]

    q_wxyz_AB = get_quat(X_AB)
    pose.orientation.w = q_wxyz_AB.w()
    pose.orientation.x = q_wxyz_AB.x()
    pose.orientation.y = q_wxyz_AB.y()
    pose.orientation.z = q_wxyz_AB.z()
    return pose


def pose_to_xform(pose):
    xyz = np.array([pose.position.x, pose.position.y, pose.position.z])
    q_wxyz = Quaternion(
        w=pose.orientation.w,
        x=pose.orientation.x,
        y=pose.orientation.y,
        z=pose.orientation.z,
    )
    return RigidTransform(quaternion=q_wxyz, p=xyz)


def xyzquat_to_xform(xyzq, mm_to_m=False):
    xyz = xyzq[:3]
    if mm_to_m:
        xyz *= 1e-3
    q_wxyz = Quaternion(w=xyzq[3], x=xyzq[4], y=xyzq[5], z=xyzq[6])
    return RigidTransform(quaternion=q_wxyz, p=xyz)


def xyzquatxyzw_to_xform(xyzq, mm_to_m=False):
    xyz = xyzq[:3]
    if mm_to_m:
        xyz *= 1e-3
    q_wxyz = Quaternion(w=xyzq[6], x=xyzq[3], y=xyzq[4], z=xyzq[5])
    return RigidTransform(quaternion=q_wxyz, p=xyz)


def pose_df_rowdata_to_xform(rowdata, interfix, suffix):
    fieldnames = [
        f"{val_field}{interfix}_{suffix}"
        for val_field in ["x", "y", "z", "qw", "qx", "qy", "qz"]
    ]
    xyz_np = np.array(rowdata[fieldnames[:3]]).T
    qwxyz_np = np.array(rowdata[fieldnames[3:]]).T
    q_wxyz = Quaternion(qwxyz_np)
    return RigidTransform(quaternion=q_wxyz, p=xyz_np)


def rgb_(value):
    """Converts a value in range [0, 1] to rgb color in [0, 1]^3

    Args:
        value ([float]): [0, 1]

    Returns:
        [numpy.array[float]]: [0, 1]^3
    """
    minimum, maximum = 0.0, 1.0
    ratio = 2 * (value - minimum) / (maximum - minimum)
    b = 1 - ratio
    b = np.clip(b, 0, 1)

    r = ratio - 1
    r = np.clip(r, 0, 1)

    g = 1.0 - b - r
    return np.stack([r, g, b]).transpose()


def sleep(seconds):
    """[summary]

    Args:
        seconds ([type]): [description]
    """
    for ii in range(math.floor(seconds)):
        time.sleep(1)
        # print(".", end="", flush=True)
    time.sleep(seconds % 1)
    # print("-")


def poses_to_angles_array(X_BI, X_IT):
    """[summary]

    Args:
        X_BI ([type]): [description]
        X_IT ([type]): [description]

    Returns:
        [type]: [description]
    """

    theta_phi__BI = rotmat_to_unit_spherical(X_BI.rotation().matrix())
    theta_phi__IT = rotmat_to_unit_spherical((X_IT).rotation().matrix())

    return np.concatenate([theta_phi__BI, theta_phi__IT])


def get_pressures_array(idx=None, pressure=None):
    """[summary]

    Args:
        idx ([type], optional): [description]. Defaults to None.
        pressure ([type], optional): [description]. Defaults to eps.
    """
    pressures_array = np.ones(6) * eps
    if idx is not None:
        pressures_array[idx] = pressure
    return pressures_array


def write_data_to_file(
    angles, pressures, filepath="../../sopra-fem/sensor_test/Temp/AngleData.txt"
):
    """Save angle and pressure data to txt file

    Args:
        angles (4x1 float np.array): 2* 2 angles for each segment
        pressures (6x1 float np.array): 6 chambers
        filepath (str, optional): Defaults to "../../sopra-fem/sensor_test/Temp/AngleData.txt"
    """
    # eps = 1e-10
    save_data_array = np.hstack([angles, pressures])
    np.savetxt(filepath, save_data_array)


def fromBendLabsToSpherical(alpha, beta):

    return alpha, beta

    # # theta is the angle from the vertical (z-axis)
    # # phi is the angle in the xy-plane
    # abs_beta = np.abs(beta)
    # sgn_beta = np.sign(beta)
    # theta = np.sqrt(
    #     alpha ** 2 + beta ** 2
    # )  # the angle is kind of the hypothenuse of a triangle where the sides lengths are both angles
    # phi = (
    #     abs_beta / (np.abs(alpha) + abs_beta + np.finfo(float).eps) * 90
    # )  # if beta==0, we're in the xz plane, if beta==1 where in the yz plane
    # Offset = 0
    # if np.sign(alpha) == -1:
    #     if sgn_beta == -1:
    #         Offset = 180
    #         phi = phi + Offset
    #     else:
    #         Offset = 180
    #         phi = Offset - phi
    # elif sgn_beta == -1:
    #     Offset = 360
    #     phi = 360 - phi

    # return theta, phi


def calcAlphaAndBeta(RotationAngle, Height, RInXY):
    x = RInXY * np.cos(np.deg2rad(RotationAngle))
    y = RInXY * np.sin(np.deg2rad(RotationAngle))
    # print("x, y: " + str(x) + ", " + str(y))
    alpha = np.rad2deg(np.arctan2(x, Height))
    beta = np.rad2deg(np.arctan2(y, Height))
    return (alpha, beta)


def get_datetime_str(time_start=None, include_ms=False):
    if time_start is None:
        time_start = datetime.now()
    if include_ms:
        format_str = "%Y%m%d%H%M%S%f"
    else:
        format_str = "%Y%m%d%H%M%S"
    return time_start.strftime(format_str)


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, 'r')  # return an open file handle


class BendSensorReading(object):
    def __init__(self, beta_alpha_pair):
        """Initialize a sensor reading from alpha, beta values. the sensor
        (or the arduino lib) doesn't follow our convention of the order of
        angles (first angle is in xz plane, second in yz plane)
        """
        self.alpha, self.beta = beta_alpha_pair[1], beta_alpha_pair[0]


class GyroscopeReading(object):
    def __init__(self, phi_theta_pair):
        self.theta, self.phi = phi_theta_pair[1], phi_theta_pair[0]


def yes_or_no(question):
    reply = str(raw_input(question+' (y/n): ')).lower().strip()[:1]
    if reply == 'y':
        return True
    if reply == 'n':
        return False
    else:
        return yes_or_no("Uhhhh... please enter ")


if __name__ == "__main__":

    from tqdm import trange
    for idx in trange(10000):
        theta = np.random.rand() * np.pi/3
        phi = np.random.rand() * np.pi

        rotmat_calc = unitspherical_to_rotmat(theta, phi)
        theta_calc, phi_calc = rotmat_to_unit_spherical(rotmat_calc).tolist()

        np.testing.assert_almost_equal(theta, theta_calc)
        np.testing.assert_almost_equal(phi, phi_calc)

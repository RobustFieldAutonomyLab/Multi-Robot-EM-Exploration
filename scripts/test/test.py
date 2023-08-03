import sys

sys.path.insert(0, "../../")
from nav.virtualmap import pose_2_point_measurement
from marinenav_env.envs.utils.robot import RangeBearingMeasurement
import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve
import torch
from nav.virtualmap import get_range_pose_point, get_range_pose_point_batch
from nav.virtualmap import pose_2_point_measurement_batch, pose_2_point_measurement, predict_virtual_landmark_batch

def rb_exp():
    r_b_m = RangeBearingMeasurement()
    r = r_b_m.sigma_r
    b = r_b_m.sigma_b
    num = 1
    state_min = np.array([-100, 100, -3.14])
    state_max = np.array([100, 100, 3.14])
    range_s = np.array([-5, 5])
    state_list = [[] for _ in range(num)]
    position_list = [[] for _ in range(num)]
    result = []
    time0 = time.time()
    for i in range(num):
        state = np.random.uniform(state_min, state_max)
        state_list[i] = state
        position_list[i] = np.random.uniform(state[0:2] - range_s, state[0:2] + range_s)
        if i == 0:
            position_list[i][0] = state[0]
            position_list[i][1] = state[1]
        result.append(pose_2_point_measurement(state_list[i], position_list[i], b, r, True))
    time1 = time.time()
    r1, r2, r3, r4, r5 = pose_2_point_measurement_batch(torch.tensor(np.array(state_list)),
                                              torch.tensor(np.array(position_list)), b,r)
    time2 = time.time()
    print("batch time:", time2-time1)
    print("single time:", time1-time0)
    for i in range(num):
        assert np.allclose(result[i][0], r1[i], atol=1e-3)
        assert np.allclose(result[i][1], r2[i], atol=1e-3)
        assert np.allclose(result[i][2], r3[i], atol=1e-3)
        assert np.allclose(result[i][3], r4[i], atol=1e-3)
        assert np.allclose(result[i][4], r5[i], atol=1e-3)
def range_pose_point_exp():
    r_b_m = RangeBearingMeasurement()
    r = r_b_m.sigma_r
    b = r_b_m.sigma_b
    num = 300
    state_min = np.array([-100, 100, -3.14])
    state_max = np.array([100, 100, 3.14])
    range_s = np.array([-5, 5])
    state_list = [[] for _ in range(num)]
    position_list = [[] for _ in range(num)]
    for i in range(num):
        state = np.random.uniform(state_min, state_max)
        state_list[i] = state
        position_list[i] = np.random.uniform(state[0:2] - range_s, state[0:2] + range_s)
        if i == 0:
            position_list[i][0] = state[0]
            position_list[i][1] = state[1]
    state_tensor = torch.tensor(np.array(state_list), dtype=torch.float32)
    position_tensor = torch.tensor(np.array(position_list), dtype=torch.float32)
    time0 = time.time()
    b1, b2, b3 = get_range_pose_point_batch(state_tensor, position_tensor)
    time1 = time.time()
    b1 = b1.numpy()
    b2 = b2.numpy()
    b3 = b3.numpy()
    result = []
    for i in range(num):
        result.append(get_range_pose_point(state_list[i], position_list[i]))
    time2 = time.time()
    for i in range(num):
        assert np.allclose(result[i][0], b1[i], atol=1e-5)
        assert np.allclose(result[i][1], b2[i], atol=1e-5)
        assert np.allclose(result[i][2], b3[i], atol=1e-5)

    print("batch time: ", time1 - time0)
    print("single time: ", time2 - time1)

def cho_exp():
    r_b_m = RangeBearingMeasurement()
    r = r_b_m.sigma_r
    b = r_b_m.sigma_b
    num = 2
    num2 = 2
    state_min = np.array([-100, 100, -3.14])
    state_max = np.array([100, 100, 3.14])
    range_s = np.array([-5, 5])
    state_list = [[] for _ in range(num)]
    point_list = [[[]for _ in range(num2)] for _ in range(num)]
    Hx_list = [[[]for _ in range(num2)] for _ in range(num)]
    Hl_list = [[[]for _ in range(num2)] for _ in range(num)]
    info_list = [[[]for _ in range(num2)] for _ in range(num)]
    result = [[[]for _ in range(num2)] for _ in range(num)]
    sigmas = np.array([r_b_m.sigma_b, r_b_m.sigma_r])
    for i in range(num):
        state = np.random.uniform(state_min, state_max)
        state_list[i] = state
        for j in range(num2):
            position = np.random.uniform(state[0:2] - range_s, state[0:2] + range_s)
            point_list[i][j] = position
            bearing, range_1, _, Hx, Hl = pose_2_point_measurement(state, position, b, r, True)
            Hx_list[i][j] = Hx
            Hl_list[i][j] = Hl
            info_this = np.random.uniform(np.array([1000, 1000, 3]), np.array([30000, 30000, 50]))
        info_list[i] = np.diag(info_this)
    time0 = time.time()
    for i in range(num):
        for j in range(num2):
            information_matrix = info_list[i]
            Hx = Hx_list[i][j]
            Hl = Hl_list[i][j]
            R = np.diag(np.squeeze(sigmas)) ** 2
            # Hl_Hl_Hl = (Hl^T * Hl)^{-1}* Hl^T
            # cov = Hl_Hl_Hl * [Hx * Info_Mat^{-1} * Hx^T + R] * Hl_Hl_Hl^T
            Hl_Hl_Hl = np.matmul(np.linalg.pinv(np.matmul(Hl.transpose(), Hl)), Hl.transpose())
            A = np.matmul(Hx, cho_solve(cho_factor(information_matrix), Hx.transpose())) + R
            cov = np.matmul(np.matmul(Hl_Hl_Hl, A), Hl_Hl_Hl.transpose())
            result[i][j] = np.linalg.pinv(cov)
    time1 = time.time()
    point_batch = torch.tensor(np.array(point_list))
    state_batch = torch.tensor(np.array(state_list))
    # Hl = torch.tensor(np.array(Hl_list))
    # Hx = torch.tensor(np.array(Hx_list))

    # Convert data to PyTorch tensors
    sigmas = torch.tensor(np.array(sigmas))
    information_matrix = torch.tensor(np.array(info_list))
    _, _, Hx, Hl = pose_2_point_measurement_batch(state_batch, point_batch)
    info_batch = predict_virtual_landmark_batch(Hx,Hl, sigmas,information_matrix)
    info_numpy = info_batch.numpy()
    time2 = time.time()
    Hx = Hx.numpy()
    Hl = Hl.numpy()
    for i in range(num):
        for j in range(num2):
            if not np.allclose(result[i][j], info_numpy[i][j], atol=1e-5):
                print("Hx,true",  Hx_list[i][j],)
                print("Hx, calculated: ", Hx[i,j, :, :])
                print("Hl,true", Hl_list[i][j])
                print("Hl,calculated: ", Hl[i,j, :, :])
                print("result, info_numpy, sum:", result[i][j],info_numpy[i][j],np.sum(result[i][j]-info_numpy[i][j]) )

    print(time1 - time0)
    print(time2 - time1)

cho_exp()
# range_pose_point_exp()
# rb_exp()
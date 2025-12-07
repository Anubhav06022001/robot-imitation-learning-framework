import numpy as np
import gym
import mujoco as mj
from mujoco import mjtObj

# ---------------------In this module we are creating gym like environment for mujoco---------------------
class BrachiationEnv(gym.Env):
    def __init__(self, xml_path, simend=20):
        super().__init__()
        self.simend = simend
        self.model= mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        low_obs = np.array([0, -np.inf , -np.inf, -np.inf, -np.inf], dtype = np.float32)
        high_obs = np.array([np.inf, np.inf, np.inf, np.inf, np.inf], dtype = np.float32)
        self.observation_space = gym.spaces.Box(low_obs,high_obs, dtype= np.float32)
        self.action_space = gym.spaces.Box(
                    low= np.array([-np.inf],dtype= np.float32), 
                    high= np.array([np.array(np.inf)], dtype = np.float32),
                    dtype = np.float32
                    )
        self.joint_ids = {
            "shoulder": mj.mj_name2id(self.model, mjtObj.mjOBJ_JOINT, "shoulder"),
            "elbow" : mj.mj_name2id(self.model, mjtObj.mjOBJ_JOINT,"elbow")
        }
        self.actuator_ids = {
            'elbow' : mj.mj_name2id(self.model , mjtObj.mjOBJ_ACTUATOR, "elbow_motor")
        }
    
    def reset(self, * , seed = None, options = None):
        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model , self.data)     
        return self._get_obs(), {}
    
    def step(self, action):
        self.data.ctrl[self.actuator_ids['elbow']] = float(action[0])
        mj.mj_step(self.model, self.data)
        obs = self._get_obs()
        done = (self.data.time >= self.simend)
        return obs, 0.0 , done, False, {}
    
    def _get_obs(self):
        t = self.data.time
        th1 = self.data.qpos[self.joint_ids['shoulder']]
        th2 = self.data.qpos[self.joint_ids['elbow']]
        d1 = self.data.qvel[self.joint_ids['shoulder']]
        d2 = self.data.qvel[self.joint_ids['elbow']]
        return np.array([t, th1,th2,d1,d2], dtype= np.float32)
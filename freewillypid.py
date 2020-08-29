import numpy as np
import time

class freewillyPID():
    def __init__(self, f2x, f2y, pathtime,pathlength):
        # Angle gains
        self._Kpa = 40
        self._Kda = 10
        self._Kia = 0.1
        #Distance gains
        self._Kpd = 5
        self._Kdd = 0.1
        self._Kid = 0
        #Trajectory
        self.f2x = f2x
        self.f2y = f2y
        self.pathtime = pathtime
        self.pathlength = pathlength
        #Integral controller initialize
        self.int_a = 0
        self.int_d = 0
        # Error for the vehicle pose
        self._error_a = 0
        self.error_d= 0
        self.d_errlist = [0]
        self.a_errlist = [0]
        self.dt = 0.1

        self._is_init = True
   
    def generate_time_vec(self):
        timevec = np.linspace(0, self.pathtime, self.pathlength*2)
        return timevec
    
    def _reset_controller(self):
        self._error_pose = np.zeros(4)
        self._int = np.zeros(4)
    
    def calc_thrust(self,xpos, ypos,ang,i):
        timevec = generate_time_vec()
        ydes = self.f2y(timevec[i])
        xdes = self.f2x(timevec[i])
        yerr = ydes - ypos
        xerr = xdes - xpos
        d_err = np.sqrt(yerr**2+xerr**2)
        self.int_d += d_err
        self.d_errlist.append(d_err)
        #Calc u3
        u3 = d_err*self._Kpd+self._Kid*self.int_d+self.Kdd*(self.d_errlist[i]-self.d_errlist[i-1])
        #Calc u1, u2
        ang_des = np.arctan2(yerr, xerr)
        ang_err = ang_des - ang
        self.int_a += ang_err
        self.a_errlist.append(ang_err)

        u1 = -1*(ang_err*self._Kpa+self._Kia*self.int_a+self.Kda*(self.a_errlist[i]-self.a_errlist[i-1]))
        u2 = 1*(ang_err*self._Kpa+self._Kia*self.int_a+self.Kda*(self.a_errlist[i]-self.a_errlist[i-1]))

    return u1, u2, u3
    
    def follow_traj(xpos, ypos, ang):
        timevec = generate_time_vec()
        for i in range(1, len(timevec)+1):
            u1,u2,u3 = calc_thrust(xpos, ypos, ang, i)
            #Command motors from here
            time.sleep(self.dt)



import numpy as np
import matplotlib.pyplot as plt
from true import true_measurement
import matplotlib

matplotlib.rcParams['font.sans-serif'] = "Kiona"

##IMU outputs at 100Hz
##GPS outputs at 10 hz
class sim_measurements():
    def __init__(self, sp, sa):
        self.var = 1
        self.gpsfreq = 10
        self.imufreq = 100
        self.scale_freq = self.imufreq/self.gpsfreq
        self.sp = sp
        self.sa = sa
        self.mpxlist = []
        self.mpylist = []
        self.mpzlist = []
        self.sqw = 0.01
        self.sqx = 0.01
        self.sqy = 0.01
        self.sqz = 0.01
    def generate_GPS_measurements(self,x,y,z, i):
        if i%self.scale_freq == 0:
            GPS = True
            mpx, mpy, mpz = self.update_GPS_measurements(x, y, z)
            self.mpxlist.append(mpx)
            self.mpylist.append(mpy)
            self.mpzlist.append(mpz)

        else:
            self.mpxlist = self.mpxlist
            self.mpylist = self.mpylist
            self.mpzlist = self.mpzlist
            GPS = False
            mpx = self.mpxlist[-1]
            mpy = self.mpylist[-1]
            mpz = self.mpzlist[-1]
        return mpx, mpy, mpz, GPS

    def update_GPS_measurements(self,x, y, z):
        mpx = np.array(x+self.sp*np.random.randn())
        mpy = np.array(y+self.sp*np.random.randn())
        mpz = np.array(z+self.sp*np.random.randn())
        return mpx, mpy, mpz

    def estimate_orientation(self, true_quat): #https://cdn-learn.adafruit.com/downloads/pdf/adafruit-bno055-absolute-orientation-sensor.pdf?timestamp=1585614037
    #IMU has internal sensor fusion algorithm to get good estimates for orientation
        quat_var = np.array([self.sqw, self.sqx, self.sqy, self.sqz])
        true_orientation = true_quat+quat_var*(np.random.randn(1,4).flatten())
        return true_orientation
    
    def quat_to_rot_mat(self, quat): #https://cdn-learn.adafruit.com/downloads/pdf/adafruit-bno055-absolute-orientation-sensor.pdf?timestamp=1585614037
    #IMU has internal sensor fusion algorithm to get good estimates for orientation
        rot_mat = np.matrix([[1-2*quat[2]**2-2*quat[3]**2, 2*quat[1]*quat[2]-2*quat[3]*quat[0], 2*quat[1]*quat[3]+2*quat[2]*quat[0]],
                            [2*quat[1]*quat[2], 1-2*quat[1]**2-2*quat[3]**2, 2*quat[2]*quat[3]-2*quat[1]*quat[0]],
                            [2*quat[1]*quat[3]-2*quat[2]*quat[0], 2*quat[2]*quat[3]+2*quat[1]*quat[0], 1-2*quat[1]**2-2*quat[2]**2]])
        return rot_mat
    def generate_accel_measurements(self, ax, ay, az, true_quat): #need to rotate acceleration from IMU via gyroscope
        mx = np.array(ax+self.sa*np.random.randn())
        my = np.array(ay+self.sa*np.random.randn())
        mz = np.array(az+self.sa*np.random.randn())
        a_body = np.array([mx, my, mz])
        transform_quat = self.estimate_orientation(true_quat)
        rot_mat = self.quat_to_rot_mat(transform_quat)
        # print(rot_mat.shape)

        a_world = (a_body*rot_mat.T).flatten()
        return a_world
    
    def collate_measurements(self,  x, y, z, ax, ay, az, true_quat, i):
        mpx, mpy, mpz, GPS= self.generate_GPS_measurements(x, y, z,i)
        accel = self.generate_accel_measurements(ax, ay, az, true_quat)
        mx = accel[0,0]
        my = accel[0,1]
        mz = accel[0,2]
        measurements = np.vstack((mpx,mpy,mpz, mx,my, mz))
        return measurements, GPS



class kalman_filter():
    def __init__(self, dt, sp, sa):
        self.dt = dt
        self.sp = sp
        self.sa = sa
        
    def A(self):
        A = np.matrix([[1.0, 0.0, 0.0, self.dt, 0.0, 0.0, 1/2.0*self.dt**2, 0.0,0.0],
                    [0.0,1.0,0.0, 0.0, self.dt, 0.0,0.0,1/2.0*self.dt**2,0.0],
                    [0.0, 0.0,1.0, 0.0, 0.0, self.dt, 0.0,0.0, 1/2.0*self.dt**2],
                    [0.0,0.0,0.0,1.0,0.0,0.0,self.dt,0.0,0.0],
                    [0.0,0.0,0.0,0.0,1.0,0.0,0.0,self.dt,0.0],
                    [0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,self.dt],
                    [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0],
                    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0],
                    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]])
        return A
    def Q(self):
        G = np.matrix([[1/2.0*self.dt**2],
                       [1/2.0*self.dt**2],
                       [1/2.0*self.dt**2],
                       [self.dt],
                       [self.dt],
                       [self.dt],
                       [1.0],
                       [1.0],
                       [1.0]])
        Q = G*G.T*self.sa**2  
        return Q  
    def initialize_x(self):
        x = np.matrix([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
        return x

    def initialize_P(self):
        P = np.diag([200.0, 200.0, 200.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0]) #Iniital covariance matrix (a function of uncertainty of initial states. The 100's are for x, the 10's are for x', and 1's are for x'')
        return P
    def initialize_H(self):
        H = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]])
        return H
    def R(self):
        ra = 10.0**2   # Noise of Acceleration Measurement
        rp = 100.0**2  # Noise of Position Measurement
        R = np.matrix([[rp, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, rp, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, rp, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, ra, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, ra, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, ra]])
        return R
    def I(self):
        I = np.eye(9)
        return I
    def kalman(self, data, GPS, x, P, H, A, Q, R, I):        
        x = A*x    
        P = A*P*A.T + Q  
        if GPS:        
            S = H*P*H.T + R
            K = (P*H.T) * np.linalg.pinv(S)        
            Z = data.reshape(H.shape[0],1)
            y = Z - (H*x)                            # Innovation or Residual
            x = x + (K*y)              
            P = (I - (K*H))*P
        return x, P


def savestates(x, P, measurements):
    xk.append(float(x[0]))
    yk.append(float(x[1]))
    zk.append(float(x[2]))
    dxk.append(float(x[3]))
    dyk.append(float(x[4]))
    dzk.append(float(x[5]))
    ddxk.append(float(x[6]))
    ddyk.append(float(x[7]))
    ddzk.append(float(x[8]))

    xsim.append(measurements[0])
    ysim.append(measurements[1])
    zsim.append(measurements[2])


    xtl.append(xt)
    ytl.append(yt)
    ztl.append(zt)

    Px.append(float(P[0,0]))
    Py.append(float(P[1,1]))
    Pz.append(float(P[2,2]))

    Pdx.append(float(P[3,3]))
    Pdy.append(float(P[4,4]))
    Pdz.append(float(P[5,5]))

    Pddx.append(float(P[4,4]))
    Pddy.append(float(P[5,5]))
    Pddz.append(float(P[6,6]))

def plot_P():
    fig = plt.figure(0)
    plt.subplot(211)
    plt.plot(range(len(Px)),Px, label='x')
    plt.plot(range(len(Py)),Py, label='y')
    plt.plot(range(len(Pz)),Pz, label='z')

    plt.title('Uncertainty (Elements from Matrix $P$)')
    plt.legend(loc='best',prop={'size':22})
    plt.subplot(212)
    plt.plot(range(len(Pddx)),Pddx, label='xdot')
    plt.plot(range(len(Pddy)),Pddy, label='ydot')
    plt.plot(range(len(Pddz)),Pddz, label='zdot')


    plt.xlabel('Filter Step')
    plt.ylabel('')
    plt.legend(loc='best',prop={'size':22})
    plt.show()


def plot_loc():
    print('printing loc')
    fig = plt.figure(1, figsize = (9,9))
    plt.subplot(211)
    plt.scatter(xk, yk, color = "red", label = "Kalman Filter Output", s = 2)
    plt.scatter(xtl, ytl, color = "green", label="Ground Truth", s = 2)
    plt.scatter(xsim, ysim, color = "blue", label="Simulated Sensor Readout", s = 2)
    plt.ylabel("Distance (m)")
    plt.xlabel("Distance (m)")

    plt.legend(loc='best',prop={'size':10})
    plt.show()

if __name__ == '__main__':
    dt = 0.1
    sa = 0.5
    sp = 1.4

    xk = []
    yk = []
    zk = []

    xtl = []
    ytl = []
    ztl = []

    xsim = []
    ysim = []
    zsim = []

    dxk= []
    dyk= []
    dzk= []
    ddxk=[]
    ddyk=[]
    ddzk=[]
    Zx = []
    Zy = []
    Zz = []
    Px = []
    Py = []
    Pz = []
    Pdx= []
    Pdy= []
    Pdz= []
    Pddx=[]
    Pddy=[]
    Pddz=[]
    Kx = []
    Ky = []
    Kz = []
    Kdx= []
    Kdy= []
    Kdz =[]
    Kddx=[]
    Kddy=[]
    Kddz=[]

    m = 700
    sim = sim_measurements(sp, sa)
    Kalman = kalman_filter(dt, sp, sa)
    x= Kalman.initialize_x()
    P = Kalman.initialize_P()
    H = Kalman.initialize_H()
    A = Kalman.A()
    Q = Kalman.Q()
    R = Kalman.R()
    I = Kalman.I()

    for i in range(m):
        truemeas = true_measurement()
        xt = truemeas.xt(i)
        yt = truemeas.yt(i)
        zt = truemeas.zt(i)
        axt = truemeas.axt()
        ayt = truemeas.ayt()
        azt = truemeas.azt()
        true_quat = truemeas.quaternion_orientation()
        measurements, GPS = sim.collate_measurements(xt,yt, zt, axt, ayt, azt, true_quat, i)
        kalman_x, kalman_P = Kalman.kalman(measurements, GPS, x, P, H, A, Q, R, I)
        P = kalman_P
        x = kalman_x
        savestates(x, P, measurements)
    plot_P()
    plot_loc()
    print('done!')


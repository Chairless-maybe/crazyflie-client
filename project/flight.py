###################################
# IMPORTS

# Imports for crazyflie (the drone)
import logging
import time
import json
import numpy as np
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig

# Imports for qualisys (the motion capture system)
import asyncio
import xml.etree.cElementTree as ET
from threading import Thread
import qtm_rt as qtm
from scipy.spatial.transform import Rotation
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 
# 








logging.basicConfig(level=logging.ERROR)


###################################
# PARAMETERS

# Specify the uri of the drone to which you want to connect (if your radio
# channel is X, the uri should be 'radio://0/X/2M/E7E7E7E7E7')
uri = 'radio://0/78/2M/E7E7E7E7E7'

# Specify the variables you want to log at 100 Hz from the drone
variables = [
    # State estimates (custom observer)
    'ae483log.p_x',
    'ae483log.p_y',
    'ae483log.p_z',
    'ae483log.psi',
    'ae483log.theta',
    'ae483log.phi',
    'ae483log.v_x',
    'ae483log.v_y',
    'ae483log.v_z',
    # 'ae483log.r_s',
    'ae483log.d_dot',
    'ae483log.speed_cos',
    # State estimates (default observer)
    'stateEstimate.x',
    'stateEstimate.y',
    'stateEstimate.z',
    'stateEstimate.yaw',
    'stateEstimate.pitch',
    'stateEstimate.roll',
    'stateEstimate.vx',
    'stateEstimate.vy',
    'stateEstimate.vz',
    # Measurements
    'ae483log.w_x',
    'ae483log.w_y',
    'ae483log.w_z',
    'ae483log.n_x',
    'ae483log.n_y',
    'ae483log.r',
    'ae483log.a_z',
    'ae483log.d',
    # Desired position (custom controller)
    'ae483log.p_x_des',
    'ae483log.p_y_des',
    'ae483log.p_z_des',
    # Desired position (default controller)
    'ctrltarget.x',
    'ctrltarget.y',
    'ctrltarget.z',
    # Motor power commands
    # 'ae483log.m_1',
    # 'ae483log.m_2',
    # 'ae483log.m_3',
    # 'ae483log.m_4',
    # 'ae483par.reset_observer'
]

# Specify the IP address of the motion capture system
ip_address = '128.174.245.190'

# Specify the name of the rigid body that corresponds to your active marker
# deck in the motion capture system. If your marker deck number is X, this name
# should be 'marker_deck_X'.
marker_deck_name = 'marker_deck_30'

# Specify the marker IDs that correspond to your active marker deck in the
# motion capture system. If your marker deck number is X, these IDs should be
# [X + 1, X + 2, X + 3, X + 4]. They are listed in clockwise order (viewed
# top-down), starting from the front.
marker_deck_ids = [31, 32, 33, 34]


###################################
# CLIENT FOR CRAZYFLIE

class CrazyflieClient:
    def __init__(self, uri, use_controller=False, use_observer=False, marker_deck_ids=None):
        self.use_controller = use_controller
        self.use_observer = use_observer
        self.marker_deck_ids = marker_deck_ids
        self.cf = Crazyflie(rw_cache='./cache')
        self.cf.connected.add_callback(self._connected)
        self.cf.fully_connected.add_callback(self._fully_connected)
        self.cf.connection_failed.add_callback(self._connection_failed)
        self.cf.connection_lost.add_callback(self._connection_lost)
        self.cf.disconnected.add_callback(self._disconnected)
        print(f'CrazyflieClient: Connecting to {uri}')
        self.cf.open_link(uri)
        self.is_fully_connected = False
        self.data = {}

    def _connected(self, uri):
        print(f'CrazyflieClient: Connected to {uri}')
    
    def _fully_connected(self, uri):
        if self.marker_deck_ids is not None:
            print(f'CrazyflieClient: Using active marker deck with IDs {marker_deck_ids}')

            # Set the marker mode (3: qualisys)
            self.cf.param.set_value('activeMarker.mode', 3)

            # Set the marker IDs
            self.cf.param.set_value('activeMarker.front', marker_deck_ids[0])
            self.cf.param.set_value('activeMarker.right', marker_deck_ids[1])
            self.cf.param.set_value('activeMarker.back', marker_deck_ids[2])
            self.cf.param.set_value('activeMarker.left', marker_deck_ids[3])

        # Reset the default observer
        self.cf.param.set_value('kalman.resetEstimation', 1)
        time.sleep(0.1)
        self.cf.param.set_value('kalman.resetEstimation', 0)
        
        # Reset the ae483 observer
        self.cf.param.set_value('ae483par.reset_observer', 1)

        # Enable the controller (1 for default, 6 for ae483)
        if self.use_controller:
            self.cf.param.set_value('stabilizer.controller', 6)
        else:
            self.cf.param.set_value('stabilizer.controller', 1)

        # Enable the observer (0 for disable, 1 for enable)
        if self.use_observer:
            self.cf.param.set_value('ae483par.use_observer', 1)
        else:
            self.cf.param.set_value('ae483par.use_observer', 0)

        # Start logging
        self.logconfs = []
        self.logconfs.append(LogConfig(name=f'LogConf0', period_in_ms=10))
        num_variables = 0
        for v in variables:
            num_variables += 1
            if num_variables > 5: # <-- could increase if you paid attention to types / sizes (max 30 bytes per packet)
                num_variables = 0
                self.logconfs.append(LogConfig(name=f'LogConf{len(self.logconfs)}', period_in_ms=10))
            self.data[v] = {'time': [], 'data': []}
            self.logconfs[-1].add_variable(v)
        self.data['speed_cos'] = {'time': [], 'data': []}
        for logconf in self.logconfs:
            try:
                self.cf.log.add_config(logconf)
                logconf.data_received_cb.add_callback(self._log_data)
                logconf.error_cb.add_callback(self._log_error)
                logconf.start()
            except KeyError as e:
                print(f'CrazyflieClient: Could not start {logconf.name} because {e}')
                for v in logconf.variables:
                    print(f' - {v.name}')
            except AttributeError:
                print(f'CrazyflieClient: Could not start {logconf.name} because of bad configuration')
                for v in logconf.variables:
                    print(f' - {v.name}')
        
        print(f'CrazyflieClient: Fully connected to {uri}')
        self.is_fully_connected = True

    def _connection_failed(self, uri, msg):
        print(f'CrazyflieClient: Connection to {uri} failed: {msg}')

    def _connection_lost(self, uri, msg):
        print(f'CrazyflieClient: Connection to {uri} lost: {msg}')

    def _disconnected(self, uri):
        print(f'CrazyflieClient: Disconnected from {uri}')
        self.is_fully_connected = False
    
    def _log_data(self, timestamp, data, logconf):
        # Get magnitude of velocity. By the dot product, we have:
        # r_s_dot / speed = cos(angle between velocity and P_inT_ofB)
        # 
        # Knowing this gives us a circle of all possible target locations. How can we use this?
        # 
        # Alternatively, and probably more simply, we can minimize that cosine if we are far away, 
        # and we can maximize it if we are close to the target.

        for v in logconf.variables:
            self.data[v.name]['time'].append(timestamp / 1e3)
            self.data[v.name]['data'].append(data[v.name])
        
        # speed_cos = r_s_dot / speed
        # self.data['speed_cos']['time'].append(timestamp / 1e3)
        # self.data['speed_cos']['data'].append(self.data['ae483log.d_dot']['data'][-1] / np.sqrt(np.sum([self.data[name]['data'][-1] ** 2 for name in ['ae483log.v_x', 'ae483log.v_y', 'ae483log.v_z']])))

    def _log_error(self, logconf, msg):
        print(f'CrazyflieClient: Error when logging {logconf}: {msg}')
    

    def move(self, x, y, z, yaw, dt):
        print(f'CrazyflieClient: Move to {x}, {y}, {z} with yaw {yaw} degrees for {dt} seconds')
        start_time = time.time()
        while time.time() - start_time < dt:
            self.cf.commander.send_position_setpoint(x, y, z, yaw)
            time.sleep(0.1)
    
    def move_smooth(self, p_inW_1, p_inW_2, yaw, v):
        print(f'Move smoothly from {p_inW_1} to {p_inW_2} with yaw {yaw} degrees at {v} meters / second')

        # Make sure p_inW_1 and p_inW_2 are numpy arrays
        p_inW_1 = np.array(p_inW_1)
        p_inW_2 = np.array(p_inW_2)
        
        # Compute distance from p_inW_1 to p_inW_2
        d = np.linalg.norm (p_inW_2-p_inW_1)             # <-- FIXME (A)
        
        # Compute time it takes to move from p_inW_1 to p_inW_2 at desired speed
        dt = d/v           # <-- FIXME (B)
        
        # Get start time
        start_time = time.time()

        # Repeat until the current time is dt seconds later than the start time
        while True:
            # Get the current time
            t = time.time()
            
            # Compute what fraction of the distance from p_inW_1 to p_inW_2
            # should have been travelled by the current time
            s = 0
            if start_time <= t and t <=start_time+dt:
                s=(t-start_time)/dt
            elif t >= start_time+dt:
                s=1
            
            # Compute where the drone should be at the current time, in the
            # coordinates of the world frame
            p_inW_des = (1-s)*p_inW_1+s*p_inW_2  
            
            # Send the desired position (and yaw angle) to the drone
            self.cf.commander.send_position_setpoint(p_inW_des[0], p_inW_des[1], p_inW_des[2], yaw)

            # Stop if the move is complete (i.e., if the desired position is at p_inW_2)
            # otherwise pause for 0.1 seconds before sending another desired position
            if s >= 1:
                return
            else:
                time.sleep(0.1)
            
    def stop(self, dt):
        print(f'CrazyflieClient: Stop for {dt} seconds')
        self.cf.commander.send_stop_setpoint()
        self.cf.commander.send_notify_setpoint_stop()
        start_time = time.time()
        while time.time() - start_time < dt:
            time.sleep(0.1)

    def disconnect(self):
        self.cf.close_link()

    def get_pos(self):
        return np.array([self.data[u]['data'][-1] for u in ['ae483log.p_x', 'ae483log.p_y', 'ae483log.p_z']])

    def hover(self, yaw, dt):
        self.move(*self.get_pos(), yaw, dt)

    def move_relative(self, dP_inW, yaw, dt):
        self.move(*(self.get_pos() + dP_inW), yaw, dt)

    def move_relative_smooth(self, dP_inW, yaw, v):
        # current_pos = np.array([self.data[u]['data'][-1] for u in ['stateEstimate.x', 'stateEstimate.y', 'stateEstimate.z']])
        current_pos = self.get_pos()
        desired_pos = current_pos + dP_inW

        self.move_smooth(current_pos, desired_pos, yaw, v)

    def follow_gradient(self, dt, yaw=0., speed=0.5, travel_dist=0.5, learning_rate=0.3, min_d=1., height=0.5):
        start_time = time.time()

        # Set a heading on the XY plane. Assume fixed height for now.
        target_heading = 0.
        target_heading_old = 0.
        maneuver_time = travel_dist / speed
        

        heading_change_dir = 1. # This is +1 or -1 depending on whether we need to 
        d_hat = 0.
        vd_hat = 0.
        t = 0.
        d_hat_old = 0.
        vd_hat_old = 0.
        t_old = 0.
        gradient = 0.
        P_inW_eventual = self.get_pos()
        P_inW_eventual[2] = height
        
        while time.time() - start_time < dt:
            print(f"Time: {time.time()}")
            current_v = np.array([self.data[u]['data'][-1] for u in ['ae483log.v_x', 'ae483log.v_y', 'ae483log.v_z']])

            # dir_des = np.array([np.cos(target_heading), np.sin(target_heading), 0])
            d_hat_old = d_hat
            t_old = t
            
            d_hat = self.data['ae483log.d']['data'][-1] - min_d
            t = self.data['ae483log.d']['time'][-1] - start_time

            vd_hat_old = vd_hat
            if t != t_old:
                vd_hat = (d_hat - d_hat_old) / (t - t_old)
            else:
                vd_hat = 0

            if np.abs(d_hat - d_hat_old) > speed * 0.1 * 2 or d_hat_old == d_hat: # Sometimes, measurements don't come in, and issues arise.
                print("Bad measurement. Skipping...")
                time.sleep(0.01) # Wait for next measurement and restart the loop
                continue

            print(f"d_hat = {d_hat}")

            self.move(*P_inW_eventual, yaw, 0.75)
            
            # gradient = 0. if np.isclose(target_heading, target_heading_old) else (speed_cos - speed_cos_old) / (target_heading - target_heading_old)
            # speed_cos may be better when doing this in 3D, but 2D heading seems like it should use r_s...or maybe I'm full of shit

            # gradient = 0. if np.isclose(target_heading, target_heading_old, atol=0.01) else dd / (target_heading - target_heading_old)
            gradient = 0. if np.isclose(target_heading, target_heading_old, atol=0.01) else (vd_hat - vd_hat_old) / (target_heading - target_heading_old)

            # gradient = 0. if np.isclose(target_heading, target_heading_old, atol=0.01) else (speed_cos - speed_cos_old) / (target_heading - target_heading_old)


            # If gradient is zero
            #     We are at a maximum or minimum distance.
            #     Our heading is constant.
            #     We may be going the right way.

            # If gradient is not zero
            #     We have changed our direction recently (duh)
            #     We want to change heading by learning_rate * gradient


            target_heading_old = target_heading
            if gradient == 0:
                print("Gradient is zero")
                # if dd > 0:
                #     target_heading = -target_heading
                # if data['drone']['speed_cos'] > 0:
                target_heading += 0.1
            else:
                print(f"Gradient is {gradient}")
                d_heading = np.clip(-gradient * learning_rate, a_min=-1., a_max=1.)
                # if np.abs(d_heading) > 1.:
                #     d_heading /= np.abs(d_heading)
                target_heading += d_heading
            print(f"Heading = {target_heading}")
            
            # 
            dir_des = np.array([np.cos(target_heading), np.sin(target_heading), 0])
            current_pos = self.get_pos()

            P_inW_eventual = current_pos + travel_dist * target_heading
            P_inW_eventual[2] = height

            move_start = time.time()
            while True:
                current_t = time.time()
                
                s = 0

                if move_start <= current_t and current_t <= move_start + maneuver_time:
                    s=(current_t-move_start)/maneuver_time
                elif current_t >= move_start + maneuver_time:
                    s=1
                
                # Compute where the drone should be at the current time, in the
                # coordinates of the world frame
                p_inW_des = (1-s) * current_pos+s * P_inW_eventual
                
                # p_inW_des = (i + 1) / 5 * (speed * 0.1 * dir_des * np.min([d_hat, 1])) + current_pos
                self.cf.commander.send_position_setpoint(*p_inW_des[:2], height, yaw)

                if s >= 1:
                    break
                else:
                    time.sleep(0.1)
            
                # dir_des = np.array([np.cos(target_heading), np.sin(target_heading), 0])
                # p_inW_des = 0.1 * speed * dir_des * learning_rate * np.min([r_s_hat, 1]) + self.get_pos()


                # We want to minimize heading_cos...
            # If we are not moving fast enough, we need to get moving in order to know how to adjust our heading.
            # So, start moving in target heading direction.
            # else:
            

            # speed_cos_old = speed_cos

            # Move in direction of heading

            # Find change in r_s

            # Using change in r_s, adjust heading
            # Use two-step backward differentiation to 
            

            # Calculate gradient
            # ds = 0.05 # step size in each direction for gradient calculation
            # grad_steps = ds * np.eye(3)
            
            # r_initial = self.data['ae483log.d']['data'][-1]
            
            # dr = np.array([0., 0., 0.])

            # for i in range(2):
            #     self.move_relative_smooth(grad_steps[i, :], yaw, speed)
            #     self.hover(yaw, 0.5)
            #     dr[i] = self.data['ae483log.r_s']['data'][-1] - r_initial
            #     self.move_relative_smooth(-grad_steps[i, :], yaw, speed)
            #     self.hover(yaw, 0.5)
            #     r_initial = self.data['ae483log.r_s']['data'][-1]

            # self.gradient = dr / ds
            # self.gradient /= np.linalg.norm(self.gradient)
        
            # # Follow gradient
            # print("Following gradient")
            # print(f"Gradient = {self.gradient}")
            # if r_initial > 1.:
            #     self.move_relative_smooth(-travel_dist * self.gradient, yaw, speed)
            # else:
            #     self.move_relative_smooth(travel_dist * self.gradient, yaw, speed)

        return
###################################
# CLIENT FOR QUALISYS

class QualisysClient(Thread):

    def __init__(self, ip_address, marker_deck_name):
        Thread.__init__(self)
        self.ip_address = ip_address
        self.marker_deck_name = marker_deck_name
        self.connection = None
        self.qtm_6DoF_labels = []
        self._stay_open = True
        self.data = {
            'time': [],
            'x': [],
            'y': [],
            'z': [],
            'yaw': [],
            'pitch': [],
            'roll': [],
        }
        self.start()

    def close(self):
        self._stay_open = False
        self.join()

    def run(self):
        asyncio.run(self._life_cycle())

    async def _life_cycle(self):
        await self._connect()
        while (self._stay_open):
            await asyncio.sleep(1)
        await self._close()

    async def _connect(self):
        print('QualisysClient: Connect to motion capture system')
        self.connection = await qtm.connect(self.ip_address, version='1.24')
        params = await self.connection.get_parameters(parameters=['6d'])
        xml = ET.fromstring(params)
        self.qtm_6DoF_labels = [label.text.strip() for index, label in enumerate(xml.findall('*/Body/Name'))]
        await self.connection.stream_frames(
            components=['6d'],
            on_packet=self._on_packet,
        )

    def _on_packet(self, packet):
        header, bodies = packet.get_6d()
        
        if bodies is None:
            print(f'QualisysClient: No rigid bodies found')
            return
        
        if self.marker_deck_name not in self.qtm_6DoF_labels:
            print(f'QualisysClient: Marker deck {self.marker_deck_name} not found')
            return
         
        index = self.qtm_6DoF_labels.index(self.marker_deck_name)
        position, orientation = bodies[index]

        # Get time in seconds, with respect to the qualisys clock
        t = packet.timestamp / 1e6

        # Get position of marker deck (x, y, z in meters)
        x, y, z = np.array(position) / 1e3
        
        # Get orientation of marker deck (yaw, pitch, roll in radians)
        R = Rotation.from_matrix(np.reshape(orientation.matrix, (3, -1), order='F'))
        yaw, pitch, roll = R.as_euler('ZYX', degrees=False)

        # Store time, position, and orientation
        self.data['time'].append(t)
        self.data['x'].append(x)
        self.data['y'].append(y)
        self.data['z'].append(z)
        self.data['yaw'].append(yaw)
        self.data['pitch'].append(pitch)
        self.data['roll'].append(roll)

    async def _close(self):
        await self.connection.stream_frames_stop()
        self.connection.disconnect()


###################################
# FLIGHT CODE

if __name__ == '__main__':
    # Specify whether or not to use the motion capture system
    use_mocap = False

    # Initialize radio
    cflib.crtp.init_drivers()

    # Create and start the client that will connect to the drone
    drone_client = CrazyflieClient(
        uri,
        use_controller=True,
        use_observer=True,
        marker_deck_ids=marker_deck_ids if use_mocap else None,
    )

    # Wait until the client is fully connected to the drone
    while not drone_client.is_fully_connected:
        time.sleep(0.1)
    
    # Create and start the client that will connect to the motion capture system
    if use_mocap:
        mocap_client = QualisysClient(ip_address, marker_deck_name)


    # Pause before takeoff
    drone_client.stop(1.0)
    drone_client.cf.param.set_value('ae483par.reset_observer', 1)
    # drone_client.stop(6.0)
    # print(f"Starting position: {drone_client.get_pos()}")

    # Graceful takeoff
    drone_client.move(0.0, 0.0, 0.2, 0.0, 1.0)
    drone_client.move_smooth([0., 0., 0.2], [0., 0., 0.5], 0.0, 0.20)
    drone_client.move(0.0, 0.0, 0.5, 0.0, 1.0)

    # drone_client.move_relative(np.array([-0.5, 0.0, 0.2]), 0.0, 1.0)
    # drone_client.hover(0., 5.)
    # drone_client.move_smooth([0., 0., 0.2], [0., 0., 0.5], 0.0, 0.20)
    # drone_client.move(0.0, 0.0, 0.5, 0.0, 1.0)

    print(f"Estimated position now: {drone_client.get_pos()}")

    drone_client.follow_gradient(30., learning_rate=1., speed=0.5, min_d=0.5)
    
    # Move in a square five times (with a pause at each corner)
    # num_squares = 5
    # for i in range(num_squares):
    #     drone_client.move_smooth([0.0, 0.0, 0.5], [0.5, 0.0, 0.5], 0.0, 0.20)
    #     drone_client.move(0.5, 0.0, 0.5, 0.0, 1.0)
    #     drone_client.move_smooth([0.5, 0.0, 0.5], [0.5, 0.5, 0.5], 0.0, 0.20)
    #     drone_client.move(0.5, 0.5, 0.5, 0.0, 1.0)
    #     drone_client.move_smooth([0.5, 0.5, 0.5], [0.0, 0.5, 0.5], 0.0, 0.20)
    #     drone_client.move(0.0, 0.5, 0.5, 0.0, 1.0)
    #     drone_client.move_smooth([0.0, 0.5, 0.5], [0.0, 0.0, 0.5], 0.0, 0.20)
    #     drone_client.move(0.0, 0.0, 0.5, 0.0, 1.0)

    # Graceful landing
    # drone_client.move_smooth(drone_client.get_pos(), [0., 0., 0.20], 0.0, 0.20)
    # drone_client.move(0.0, 0.0, 0.20, 0.0, 1.0)

    # Disconnect from the drone
    drone_client.disconnect()

    # Disconnect from the motion capture system
    if use_mocap:
        mocap_client.close()

    # Assemble flight data from both clients
    data = {}
    data['drone'] = drone_client.data
    data['mocap'] = mocap_client.data if use_mocap else {}

    # Write flight data to a file
    with open('follower_test.json', 'w') as outfile:
        json.dump(data, outfile, sort_keys=False)
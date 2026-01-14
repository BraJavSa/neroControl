#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
import matplotlib.pyplot as plt
import tf_transformations
import threading
import time
import numpy as np

class RealTimePlot(Node):
    def __init__(self):
        super().__init__('realtime_plot_node')

        # --- Subscribers ---
        self.sub_odom = self.create_subscription(
            Odometry, '/bebop/odom', self.odom_callback, 10)
        self.sub_ref = self.create_subscription(
            Float64MultiArray, '/bebop/ref_vec', self.ref_callback, 10)

        self.lock = threading.Lock()
        self.t_data = []
        self.x_data, self.y_data, self.z_data, self.yaw_data = [], [], [], []

        self.t_ref = []
        self.x_ref, self.y_ref, self.z_ref, self.yaw_ref = [], [], [], []

        self.start_time = time.time()

        # --- Plot setup ---
        plt.ion()
        self.fig, self.axs = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
        self.fig.suptitle("Position and Orientation States", fontsize=14, fontweight='bold')

        labels = ['x [m]', 'y [m]', 'z [m]', 'yaw [rad]']
        self.lines_pos, self.lines_ref = [], []

        for ax, label in zip(self.axs, labels):
            ax.set_ylabel(label)
            ax.grid(True)
            line_pos, = ax.plot([], [], lw=2, color='tab:blue', label='Measured')
            line_ref, = ax.plot([], [], lw=2, color='tab:red', linestyle='--', label='Reference')
            ax.legend(loc='upper right')
            self.lines_pos.append(line_pos)
            self.lines_ref.append(line_ref)

        self.axs[-1].set_xlabel('Time [s]')
        self.window = 4.0  # seconds
        self.timer = self.create_timer(0.05, self.update_plot)  # 20 Hz

    def odom_callback(self, msg):
        with self.lock:
            t = time.time() - self.start_time
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            z = msg.pose.pose.position.z
            q = msg.pose.pose.orientation
            quat = [q.x, q.y, q.z, q.w]
            _, _, yaw = tf_transformations.euler_from_quaternion(quat)
            yaw = np.arctan2(np.sin(yaw), np.cos(yaw))  # normalize to [-pi, pi]

            self.t_data.append(t)
            self.x_data.append(x)
            self.y_data.append(y)
            self.z_data.append(z)
            self.yaw_data.append(yaw)

            # keep 4s window
            while self.t_data and (self.t_data[-1] - self.t_data[0] > self.window):
                self.t_data.pop(0)
                self.x_data.pop(0)
                self.y_data.pop(0)
                self.z_data.pop(0)
                self.yaw_data.pop(0)

    def ref_callback(self, msg):
        with self.lock:
            if len(msg.data) >= 4:
                t = time.time() - self.start_time
                x, y, z, yaw = msg.data[0], msg.data[1], msg.data[2], msg.data[3]
                yaw = np.arctan2(np.sin(yaw), np.cos(yaw))

                self.t_ref.append(t)
                self.x_ref.append(x)
                self.y_ref.append(y)
                self.z_ref.append(z)
                self.yaw_ref.append(yaw)

                # keep same time window
                while self.t_ref and (self.t_ref[-1] - self.t_ref[0] > self.window):
                    self.t_ref.pop(0)
                    self.x_ref.pop(0)
                    self.y_ref.pop(0)
                    self.z_ref.pop(0)
                    self.yaw_ref.pop(0)

    def update_plot(self):
        with self.lock:
            if not self.t_data:
                return

            # measured data
            data_pos = [self.x_data, self.y_data, self.z_data, self.yaw_data]
            # reference data
            data_ref = [self.x_ref, self.y_ref, self.z_ref, self.yaw_ref]

            for line_p, y_p in zip(self.lines_pos, data_pos):
                line_p.set_data(self.t_data, y_p)
            for line_r, y_r in zip(self.lines_ref, data_ref):
                line_r.set_data(self.t_ref, y_r)

            t_max = max(self.t_data[-1], self.t_ref[-1] if self.t_ref else self.t_data[-1])
            t_min = t_max - self.window

            for i, ax in enumerate(self.axs):
                ax.set_xlim(t_min, t_max)

                if i == 0 or i == 1:  # x, y
                    ax.set_ylim(-2.0, 2.0)
                elif i == 2:  # z
                   
                    ax.set_ylim(0.0, 2.5)
                else:  # yaw
                    ax.set_ylim(-np.pi*1.1, np.pi*1.1)

            plt.pause(0.001)

def main(args=None):
    rclpy.init(args=args)
    node = RealTimePlot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

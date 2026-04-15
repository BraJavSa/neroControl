#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
import matplotlib.pyplot as plt
import tf_transformations
import threading
import numpy as np
from math import cos, sin

class RealTimePlot(Node):
    """
    High-performance telemetry visualizer for Bebop 2.
    Translates body-fixed velocities to the global frame for consistent 
    comparison with inertial references.
    """
    def __init__(self):
        super().__init__('realtime_plot_node')

        # --- Communications ---
        self.sub_odom = self.create_subscription(Odometry, '/odometry/filtered', self.odom_callback, 10)
        self.sub_ref = self.create_subscription(Float64MultiArray, '/bebop/ref_vec', self.ref_callback, 10)

        # --- Data Buffers (Fixed Size) ---
        self.points = 150  # Increased for better trajectory history
        self.x_axis = np.linspace(0, 1, self.points)
        self.lock = threading.Lock()
        
        self.data_buf = np.zeros((8, self.points))
        self.ref_buf = np.zeros((8, self.points))

        # --- Plotting Setup ---
        self.fig, self.axs = plt.subplots(4, 2, figsize=(11, 9))
        self.fig.canvas.manager.set_window_title('Bebop 2: Global State Tracking (Inertial Frame)')
        
        labels = ['x [m]', 'y [m]', 'z [m]', 'yaw [rad]', 
                  'vx_global [m/s]', 'vy_global [m/s]', 'vz [m/s]', 'v_yaw [rad/s]']
        
        self.lines_pos, self.lines_ref = [], []

        for i in range(8):
            ax = self.axs[i % 4, i // 4]
            ax.set_ylabel(labels[i])
            ax.set_xlim(0, 1)
            
            # Academic Standard Limits
            if i < 2: ax.set_ylim(-2.0, 2.0)      # x, y
            elif i == 2: ax.set_ylim(0, 2.5)      # z
            elif i == 3: ax.set_ylim(-3.15, 3.15) # yaw
            else: ax.set_ylim(-1.5, 1.5)          # velocities
            
            lp, = ax.plot(self.x_axis, self.data_buf[i], lw=1.6, color='#1f77b4', animated=True, label='Global Measured')
            lr, = ax.plot(self.x_axis, self.ref_buf[i], lw=1.3, color='#d62728', ls='--', animated=True, label='Global Ref')
            self.lines_pos.append(lp)
            self.lines_ref.append(lr)
            if i == 0: ax.legend(loc='upper right', fontsize='x-small')

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        
        self.create_timer(1.0/30.0, self.update_plot)

    def odom_callback(self, msg: Odometry):
        # 1. Extraction of Orientation (Yaw)
        q = msg.pose.pose.orientation
        _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        yaw_norm = np.arctan2(np.sin(yaw), np.cos(yaw))

        # 2. Body-to-Inertial Velocity Transformation
        # v_global = F(yaw) * v_body
        v_body = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
            msg.twist.twist.angular.z
        ])

        # Rotation Matrix F(psi) for 4-DOF
        F = np.array([
            [cos(yaw), -sin(yaw), 0, 0],
            [sin(yaw),  cos(yaw), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        v_global = F @ v_body

        with self.lock:
            # 3. State Vector Construction (All in Global Frame)
            curr = [
                msg.pose.pose.position.x, 
                msg.pose.pose.position.y, 
                msg.pose.pose.position.z,
                yaw_norm,
                v_global[0], 
                v_global[1], 
                v_global[2], 
                v_global[3]
            ]
            
            self.data_buf = np.roll(self.data_buf, -1, axis=1)
            self.data_buf[:, -1] = curr

    def ref_callback(self, msg: Float64MultiArray):
        if len(msg.data) < 8: return
        with self.lock:
            # The reference is already in global frame from the MPC
            self.ref_buf = np.roll(self.ref_buf, -1, axis=1)
            self.ref_buf[:, -1] = msg.data

    def update_plot(self):
        self.fig.canvas.restore_region(self.bg)
        with self.lock:
            for i in range(8):
                self.lines_pos[i].set_ydata(self.data_buf[i])
                self.lines_ref[i].set_ydata(self.ref_buf[i])
                self.axs[i % 4, i // 4].draw_artist(self.lines_pos[i])
                self.axs[i % 4, i // 4].draw_artist(self.lines_ref[i])

        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()

def main(args=None):
    rclpy.init(args=args)
    node = RealTimePlot()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    plt.show(block=True) 

if __name__ == '__main__':
    main()
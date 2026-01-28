import time
from gz.msgs10.entity_factory_pb2 import EntityFactory
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.twist_pb2 import Twist
from gz.msgs10.odometry_pb2 import Odometry
from gz.transport13 import Node

ultima_velocidad = Twist()

def odom_callback(msg):
    global ultima_velocidad
    # Copiamos el Twist de la odometría del Bebop
    ultima_velocidad.CopyFrom(msg.twist)

def mover():
    node = Node()
    world_name = "empty"
    
    request = EntityFactory()
    # MODIFICACIÓN AQUÍ: Añadimos <gravity>0</gravity>
    request.sdf = """
    <sdf version='1.8'>
      <model name='mi_bola'>
        <link name='link'>
          <gravity>0</gravity> 
          <inertial>
            <mass>0.01</mass>
            <inertia><ixx>0.0001</ixx><iyy>0.0001</iyy><izz>0.0001</izz></inertia>
          </inertial>
          <visual name='v1'>
            <geometry><sphere><radius>0.1</radius></sphere></geometry>
            <material><ambient>0 0 1 1</ambient><diffuse>0 0 1 1</diffuse></material>
          </visual>
          <collision name='c1'>
            <geometry><sphere><radius>0.1</radius></sphere></geometry>
          </collision>
        </link>
        <plugin filename="gz-sim-velocity-control-system" name="gz::sim::systems::VelocityControl" />
      </model>
    </sdf>
    """
    request.pose.position.z = 1.5 # Empezará flotando a 1.5 metros

    print("Creando esfera flotante...")
    factory_topic = f"/world/{world_name}/create"
    node.request(factory_topic, request, EntityFactory, Boolean, 5000)

    odom_topic = "/bebop/odom"
    node.subscribe(Odometry, odom_topic, odom_callback)
    
    vel_topic = "/model/mi_bola/cmd_vel"
    vel_pub = node.advertise(vel_topic, Twist)
    
    print(f"Sincronizado: La esfera flotará y seguirá al Bebop.")
    
    try:
        while True:
            vel_pub.publish(ultima_velocidad)
            time.sleep(0.05) 
            
    except KeyboardInterrupt:
        vel_pub.publish(Twist())
        print("\nDetenido.")

if __name__ == "__main__":
    mover()
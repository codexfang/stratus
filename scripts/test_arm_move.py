import sys, time
sys.path.insert(0, "/Users/seraphicflare/stratus/src")
from stratus.drivers.vectorbh6_arm import VectorBH6ArmDriver

arm = VectorBH6ArmDriver()
print("Connecting...")
arm.connect()
print("Arm enabled. Moving to pickup pose...")
arm.move_to_pose(0.25, 0.0, 0.15, pitch=0.4)
print("Waiting 3s...")
time.sleep(3)
print("Moving to drop pose...")
arm.move_to_pose(0.15, -0.2, 0.15, pitch=0.4)
time.sleep(3)
print("Disconnecting...")
arm.disconnect()
print("Done.")

from KUKA import KUKA
from GUI_pygame import GuiControl

robot = KUKA('192.168.88.21', ros=False, camera_enable=True)
#robot = ['192.168.88.21', '192.168.88.22', '192.168.88.23', '192.168.88.24', '192.168.88.25']
sim = GuiControl(1200, 900, robot)
sim.run()

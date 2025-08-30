Core Architecture
1. Multi-threading Design
MyThread: The main processing thread, responsible for camera data acquisition, YOLO target detection, and spatial coordinate calculation 
SerialReceiver: Serial communication thread, controlling the robotic arm to perform actions 
TcpReceiver: TCP client thread, responsible for receiving instructions from the server 
2. Functional Modules
Visual Processing Module: 
Obtain RGB and depth images using the DepthAI/OAKD camera 
The YOLO model is used for detecting various tools (such as circuit breakers, contactors, hammers, etc.) 
The three-dimensional spatial coordinates of the calculation tool (X, Y, Z) 
Mechanical Arm Control Module: 
Send control instructions via serial port 
Support multiple operations: Grab (Q), Place (F), Set (Set), Reset (Reset) 
Emergency stop function (such as Zscram and other safety instructions) 
Communication module: 
The TCP client connects to the server to receive instructions. 
Serial port communication controls the mechanical arm hardware 
3. Work Flow
The camera conducts real-time detection and calculates 3D coordinates 
After the user selects the target tool, they click the operation button. 
Convert the coordinates into mechanical arm control instructions (in MOV format) 
Send instructions via serial port to control the movements of the robotic arm 
Receive the feedback status from the mechanical arm and display it 
4. UI Interface
Real-time display of detection images and recognition results 
Tool Selection Drop-down Menu 
Various control buttons (start, stop, grab, place, etc.) 
Status display text box 
5. Data Flow
Camera → YOLO Detection → Coordinate Calculation → Instruction Conversion → Serial Port Transmission → Mechanical Arm Execution → Status Feedback → UI Display 
The system has achieved a complete automated process from visual detection to mechanical arm control, and is applicable to scenarios involving tool grasping and placement.

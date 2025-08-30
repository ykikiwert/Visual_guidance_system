import sys
import cv2
import depthai as dai
import numpy as np
import time
import serial
from pathlib import Path
from PyQt5.QtWidgets import QApplication,QWidget
from PyQt5 import uic
from PyQt5.Qt import QThread, pyqtSignal,QImage,QPixmap
import PyQt5.QtCore as QtCore
import socket
import threading
import struct
import pandas as pd

class TcpReceiver(QThread):
    tcp_data_received = pyqtSignal(str)

    def __init__(self, host, port):
        super().__init__()
        self.host = host
        self.port = port
        self.tcp_socket = None

    def run(self):
         while True:
            try:
                if self.tcp_socket is None:
                    self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.tcp_socket.connect((self.host, self.port))
                    print(f"Connected to TCP server at {self.host}:{self.port}")
                data = self.tcp_socket.recv(1024)
                if data:
                    self.tcp_data_received.emit(data.decode('utf-8'))
                    print("Received data:", data.decode('utf-8'))
                else:
                    break
                    self.tcp_socket.close()
            except socket.error as e:
                print(f"Error receiving data from TCP server: {e}")
                time.sleep(5)  # 等待5秒后重试连接
                self.tcp_socket = None  # 重置套接字


class MyThread(QThread):
    btQ_clicked = False
    btF_clicked = False
    restatus_text_changed = pyqtSignal(str)
    combo1_text_changed = pyqtSignal(str)
    send_frame = pyqtSignal(str)
    data_ready = pyqtSignal(list)  # 信号将发送一个列表，这里是 MOV 数据
    data_ready1 = pyqtSignal(list)   
    def __init__(self, my_window_instance):
        super().__init__()
        self.my_window_instance = my_window_instance
        self.f_pixmap=None
        self.frame=None
        self.label=None
        self.tcp_socket = None  # 初始化 TCP 套接字

        # 初始化串口连接
        self.serial_port= None
        transmit_N_size=20
        self.transmit_N=[None for _ in range(transmit_N_size)]
        self.transmit_XYZ=[[0,0,0] for _ in range(transmit_N_size)]

        self.X = 0
        self.Y = 0
        self.Z = 0
        self.MOV = [0X00AA, 0X0001, 0X0003, 0X0014, 0X0001, 0X0000, 0X0002, 0X0032, 0XFFFF, 0X00B4, 0X0000, 0X005A, 0X0000, 0X00B4, 0X003B]
        
    def emit_data(self, data):
        print("Received data:", data)
        # 将整型数组转换为列表
        data_list = list(data)
        print("Data List:", data_list)
        # 发送信号
        self.data_ready.emit(data_list)
        self.data_ready1.emit(data_list)
    def convert_to_hex(mov, x, y, z):
        # 检查输入参数是否为数字
        if not all(isinstance(i, (int, float)) for i in [x, y, z]):
            raise ValueError("输入参数必须是数字")

        x = int(x)
        y = int(y)
        z = int(z)
        # 计算 X、Y、Z 的绝对值
        abs_x = abs(x)
        abs_y = abs(y)
        abs_z = abs(z)

        # 将 X、Y、Z 的绝对值转换为 4 位 HEX 格式
        hex_x = format(abs_x, '04X')
        hex_y_flag = '0000' if y >= 0 else 'FFFF'
        hex_y = format(abs_y, '04X')
        hex_z = format(abs_z, '04X')

        # 将 X、Y、Z 的 HEX 值存放到 Mov 数组中
        mov = [0X00AA, 0X0001, 0X0003, 0X0014, 0X0001, 0X0000, 0X0002, 0X0032, 0XFFFF, 0X00B4, 0X0000, 0X005A, 0X0000, 0X00B4, 0X003B]
        
        mov[4] = int(hex_x, 16)
        mov[5] = int(hex_y_flag, 16) 
        mov[6] = int(hex_y, 16)
        mov[7] = int(hex_z, 16)

        return mov





    
    def connect_to_tcp_server(self, host, port):
        try:
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.connect((host, port))
            print(f"Connected to TCP server at {host}:{port}")
        except socket.error as e:
            print(f"Error connecting to TCP server: {e}")
    def send_tool_to_tcp(self, tool):
        if self.tcp_socket is not None:
            try:
                self.tcp_socket.sendall(tool.encode())
                print(f"Sent tool '{tool}' to TCP server.")
            except socket.error as e:
                print(f"Error sending tool to TCP server: {e}")
        else:
            print("TCP socket is not connected.")
    def on_btQ_clicked(self):
         MyThread.btQ_clicked = True
    def on_btF_clicked(self):
        MyThread.btF_clicked = True
    def run(self):
        # 获取restatus的当前文本
        restatus_text = self.my_window_instance.restatus.toPlainText()
        self.restatus_text_changed.emit(restatus_text)
        time.sleep(0.5)  # 避免过度占用 CPU
        labelMap = ["air_switch", "contactor", "hammer", "spanner", "pliers", "wire_tripper"]
        syncNN = True
        pipeline = dai.Pipeline()
        # Define sources and outputs
        camRgb = pipeline.create(dai.node.ColorCamera)
        spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)
        nnNetworkOut = pipeline.create(dai.node.XLinkOut)

        xoutRgb = pipeline.create(dai.node.XLinkOut)
        xoutNN = pipeline.create(dai.node.XLinkOut)
        xoutDepth = pipeline.create(dai.node.XLinkOut)

        xoutRgb.setStreamName("rgb")
        xoutNN.setStreamName("detections")
        xoutDepth.setStreamName("depth")
        nnNetworkOut.setStreamName("nnNetwork")

        # Properties
        camRgb.setPreviewSize(640, 640)
        #camRgb.setPreviewSize(640, 640)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setCamera("left")
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setCamera("right")

        # setting node configs
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        # Align depth map to the perspective of RGB camera, on which inference is done
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())
        stereo.setSubpixel(True)

        spatialDetectionNetwork.setBlobPath(nnBlobPath)
        spatialDetectionNetwork.setConfidenceThreshold(0.2)
        # spatialDetectionNetwork.setConfidenceThreshold(0.1)#置信度阈值
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(5000)

        #Yolo specific parameters
        spatialDetectionNetwork.setNumClasses(6)#物品种类数
        #spatialDetectionNetwork.setIouThreshold(0.5)
        spatialDetectionNetwork.setIouThreshold(0.1)
        spatialDetectionNetwork.setCoordinateSize(4)
        #spatialDetectionNetwork.setAnchors([10,14, 23,27, 37,58, 81,82, 135,169, 344,319])
        spatialDetectionNetwork.setAnchors([10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326])
        spatialDetectionNetwork.setAnchors([10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326, 373, 326, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326, 30,61, 62,45, 59,119, 116,90])
        #spatialDetectionNetwork.setAnchorMasks({ "side26": [1,2,3], "side13": [3,4,5] })
        # spatialDetectionNetwork.setAnchorMasks({ "side52": [0,1,2], "side26": [3,4,5] , "side13": [6,7,8] })
        spatialDetectionNetwork.setAnchorMasks({"side3328": [0,1,2], "side1664": [3,4,5] , "side832": [6,7,8],"side416": [9,10,11], "side208": [12,13,14] , "side104": [15,16,17],"side52": [18,19,20], "side26": [21,22,23] , "side13": [24,25,26]
 })
        # Linking.0
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        camRgb.preview.link(spatialDetectionNetwork.input)
        if syncNN:
            spatialDetectionNetwork.passthrough.link(xoutRgb.input)
        else:
            camRgb.preview.link(xoutRgb.input)

        spatialDetectionNetwork.out.link(xoutNN.input)

        stereo.depth.link(spatialDetectionNetwork.inputDepth)
        spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)
        spatialDetectionNetwork.outNetwork.link(nnNetworkOut.input)

        # Connect to device and start pipeline
        with dai.Device(pipeline) as device:

            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
            depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            networkQueue = device.getOutputQueue(name="nnNetwork", maxSize=4, blocking=False)

            startTime = time.monotonic()
            counter = 0
            fps = 0
            color = (255, 255, 255)
            printOutputLayersOnce = True

            while True:
                inPreview = previewQueue.get()
                inDet = detectionNNQueue.get()
                depth = depthQueue.get()
                inNN = networkQueue.get()

                if printOutputLayersOnce:
                    toPrint = 'Output layer names:'
                    for ten in inNN.getAllLayerNames():
                        toPrint = f'{toPrint} {ten},'
                    print(toPrint)
                    printOutputLayersOnce = False

                self.frame = inPreview.getCvFrame()
                depthFrame = depth.getFrame() # depthFrame values are in millimeters

                depth_downscaled = depthFrame[::4]
                min_depth = np.percentile(depth_downscaled[depth_downscaled != 0], 1)
                max_depth = np.percentile(depth_downscaled, 99)
                depthFrameColor = np.interp(depthFrame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
                depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

                counter+=1
                current_time = time.monotonic()
                if (current_time - startTime) > 1 :
                    fps = counter / (current_time - startTime)
                    counter = 0
                    startTime = current_time

                detections = inDet.detections

                # If the frame is available, draw bounding boxes on it and show the frame
                height = self.frame.shape[0]
                width  = self.frame.shape[1]
                channel = self.frame.shape[2]
                bytes_line=channel * width
                f_image = QImage(self.frame.data, width, height, bytes_line, QImage.Format_RGB888)
                self.f_pixmap = QPixmap.fromImage(f_image)

                i=0
                for detection in detections:
                    roiData = detection.boundingBoxMapping
                    roi = roiData.roi
                    roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                    topLeft = roi.topLeft()
                    bottomRight = roi.bottomRight()
                    xmin = int(topLeft.x)
                    ymin = int(topLeft.y)
                    xmax = int(bottomRight.x)
                    ymax = int(bottomRight.y)
                    cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, 1)

                    # Denormalize bounding box
                    x1 = int(detection.xmin * width)
                    x2 = int(detection.xmax * width)
                    y1 = int(detection.ymin * height)
                    y2 = int(detection.ymax * height)
                    try:
                        self.label = labelMap[detection.label]
                    except:
                        self.label = detection.label
                    cv2.putText(self.frame, str(self.label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(self.frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(self.frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(self.frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(self.frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    self.X = int(detection.spatialCoordinates.x)
                    self.Y = int(detection.spatialCoordinates.y)
                    self.Z = int(detection.spatialCoordinates.z)
                    # 填充物品名称和坐标
                    self.transmit_N[i] = self.label
                    self.transmit_XYZ[i][0] = self.X
                    self.transmit_XYZ[i][1] = self.Y
                    self.transmit_XYZ[i][2] = self.Z
                    i = i+1

                    cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)
                cv2.putText(self.frame, "NN fps: {:.2f}".format(fps), (2, self.frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
                #cv2.imshow("rgb", frame)
                self.send_frame.emit("transmit")#发送信号让触发槽函数
                # 检测restatus文本框中是否包含特定字符串
                restatus_text = self.my_window_instance.restatus.toPlainText()
                if "数据库存在该工具，可进行抓取！" in restatus_text:
                    if self.label == self.my_window_instance.combo1.currentText():
                            if  MyThread.btQ_clicked:
                                print(f"Coordinates for {self.label}: X={self.X}, Y={self.Y}, Z={self.Z}")
                                self.MOV= self.convert_to_hex(self.X, self.Y, self.Z)
                                # 发送数据准备好的信号
                                self.data_ready.emit(self.MOV) 
                                MyThread.btQ_clicked = False
                                print(self.MOV)
                                time.sleep(0.3)  # 避免过度占用 CPU
                            elif  MyThread.btF_clicked:
                                print(f"Coordinates for {self.label}: X={self.X}, Y={self.Y}, Z={self.Z}")
                                self.MOV= self.convert_to_hex(self.X, self.Y, self.Z)
                                # 发送数据准备好的信号
                                self.data_ready1.emit(self.MOV)
                                MyThread.btF_clicked = False
                                print(self.MOV)
                                time.sleep(0.3)  # 避免过度占用 CPU

class SerialReceiver(QThread):
    serial_data_received = pyqtSignal(bytes)
    def __init__(self, my_window_instance):
        super().__init__()
        self.my_window_instance =my_window_instance
        self.my_window_instance.serial_data_sendQ.connect(self.Q)
        self.my_window_instance.serial_data_sendF.connect(self.F)
        self.my_window_instance.serial_data_sendZs.connect(self.Zscram)
        self.my_window_instance.serial_data_sendrelZ.connect(self.relZscram)
        self.my_window_instance.serial_data_sendQs.connect(self.Qscram)
        self.my_window_instance.serial_data_sendFs.connect(self.Fscram)
        self.my_window_instance.serial_data_sendS.connect(self.Set)
        self.my_window_instance.serial_data_sendSs.connect(self.Setscram)
        self.my_window_instance.serial_data_sendR.connect(self.Reset)
        self.my_window_instance.serial_data_sendRs.connect(self.Resetscram)
      
        # try:
        #     self.serial_port = serial.Serial(
        #         port="COM4",
        #         baudrate=115200,
        #         bytesize=serial.EIGHTBITS,
        #         parity=serial.PARITY_EVEN,
        #         stopbits=serial.STOPBITS_ONE,
        #     )
        # except serial.SerialException as e:
        #     print(f"Error initializing serial port: {e}")
    def Zscram(self):
        self.serial_port.write(Allscram)
        time.sleep(2) 
        self.serial_port.write(Allscram)
 
    def Qscram(self):
        self.serial_port.write(Access_tools_scram)
        time.sleep(2)
        self.serial_port.write(Access_tools_scram)

    def Fscram(self):
        self.serial_port.write(Put_tools_scram)
        time.sleep(2)
        self.serial_port.write(Put_tools_scram)
    def Setscram(self):
        self.serial_port.write(Set_scram)
        time.sleep(2)
        self.serial_port.write(Set_scram)

    def Resetscram(self):
        self.serial_port.write(Reset_scram)
        time.sleep(2)
        self.serial_port.write(Reset_scram)

    def relZscram(self):
        self.serial_port.write(relAllscram)
        time.sleep(2)
        self.serial_port.write(relAllscram)
    def reverse_bytes(self, data):
        reversed_data = []
        for i in range(0, len(data), 2):
            byte_slice = data[i:i+2]
            reversed_data.append(byte_slice[::-1])
        return bytes(bytearray().join(reversed_data))


    def bytes_to_hex_string(byte_data):
    # 使用 hex() 方法将字节数据转换为十六进制字符串
        hex_string = byte_data.hex()
        return hex_string  

    def Q(self,mov_data):
        right_data = self.reverse_bytes(mov_data)
        self.serial_port.write(Set)
        time.sleep(0.1)
        self.serial_port.write(Access_tools)
        time.sleep(0.1)
        self.serial_port.write(right_data)
        time.sleep(2)

        self.serial_port.write(Set)
        time.sleep(0.1)
        self.serial_port.write(Access_tools)
        time.sleep(0.1)
        self.serial_port.write(right_data)
    def F(self,mov_data):
        right_data = self.reverse_bytes(mov_data)
        self.serial_port.write(right_data)
        time.sleep(0.1)
        self.serial_port.write(Put_tools)
        time.sleep(0.5)
        self.serial_port.write(Reset)
        time.sleep(2)
        self.serial_port.write(right_data)
        time.sleep(0.1)
        self.serial_port.write(Put_tools)
        time.sleep(0.5)
        self.serial_port.write(Reset)
    def Set(self):
        self.serial_port.write(Set)
        time.sleep(2)
        self.serial_port.write(Set)
    def Reset(self):
        self.serial_port.write(Reset)
        time.sleep(2)
        self.serial_port.write(Reset)
    def run(self):
        while self.isRunning:
            if self.serial_port.in_waiting > 0:
                data = self.serial_port.read(4)
                self.serial_data_received.emit(data)
                time.sleep(0.01)
                                                             

    def stop(self):
        self.isRunning = False


class MyWindow(QWidget):
    clicked = pyqtSignal()
    serial_data_sendQ= pyqtSignal(bytes)
    serial_data_sendF= pyqtSignal(bytes)
    serial_data_sendZs= pyqtSignal(bytes)
    serial_data_sendrelZ= pyqtSignal(bytes)
    serial_data_sendQs= pyqtSignal(bytes)
    serial_data_sendFs= pyqtSignal(bytes)
    serial_data_sendS= pyqtSignal(bytes)
    serial_data_sendSs= pyqtSignal(bytes)
    serial_data_sendR= pyqtSignal(bytes)
    serial_data_sendRs= pyqtSignal(bytes)
    def __init__(self):
        super().__init__()
        self.thread = MyThread(self)   
        
        self.init_ui()
        #ui界面初始化
    def init_ui(self):
        #载入ui
        self.ui = uic.loadUi("./01ui.ui")
        # 控件映射
        bt_start = self.ui.bt_start
        bt_close = self.ui.bt_close
        self.combo1 = self.ui.comboBox
        self.lab_show = self.ui.lab_show
        self.connect=self.ui.connect
        self.disconnect=self.ui.disconnect
        self.query=self.ui.query
        self.restatus = self.ui.restatus  
        btZscram=self.ui.btZscram    
        btQscram=self.ui.btQscram
        btFscram=self.ui.btFscram
        btSetscram=self.ui.btSetscram
        btResetscram=self.ui.btResetscram
        btrelZscram=self.ui.btrelZscram
        self.btQ=self.ui.btQ
        self.btF=self.ui.btF
        self.btSet=self.ui.btSet
        self.btReset=self.ui.btReset

        # 绑定
        bt_start.clicked.connect(self.run)
        bt_close.clicked.connect(self.close)
        btZscram.clicked.connect(self.Zscram)
        btQscram.clicked.connect(self.Qscram)
        btFscram.clicked.connect(self.Fscram)
        btSetscram.clicked.connect(self.Setscram)
        btResetscram.clicked.connect(self.Resetscram)
        btrelZscram.clicked.connect(self.relZscram)
        self.btQ.clicked.connect(self.thread.on_btQ_clicked) 
        self.btF.clicked.connect(self.thread.on_btF_clicked)
        self.btSet.clicked.connect(self.Set)
        self.btReset.clicked.connect(self.Reset)

        # self.combo1.activated.connect(self.com1)
        self.combo1.addItem(" ")
        self.combo1.addItem("crimping_tool")
        self.combo1.addItem("wire_tripper")
        self.combo1.addItem("spanner")
        self.combo1.addItem("hammer")
        self.combo1.addItem("air_switch")
        self.combo1.addItem("contactor")
        self.combo1.addItem("pliers")

       
        self.connect.clicked.connect(self.on_connect_button_clicked)
        self.disconnect.clicked.connect(self.disconnect_from_tcp_server)
        self.query.clicked.connect(self.on_query_button_clicked)
        

        self.mythread = MyThread(self)#创建线程对象
        self.mythread.send_frame.connect(self.updata_cv_frame)#信号绑定槽函数
        self.combo1.currentTextChanged.connect(self.mythread.combo1_text_changed) 
        self.mythread.restatus_text_changed.connect(self.update_restatus)
        self.mythread.data_ready.connect(self.Q)
        self.mythread.data_ready1.connect(self.F)
        self.serial_receiver = SerialReceiver(self) 
        self.serial_receiver.serial_data_received.connect(self.update_serial_data)
        

        self.tcp_receiver = TcpReceiver("192.168.205.37", 9093)#创建线程对象
        self.tcp_receiver.tcp_data_received.connect(self.update_restatus)
        print("Signal connected")  # 打印信号连接状态
        

    def run(self):
        self.mythread.start()
        self.tcp_receiver.start()  # 启动TCP接收线程
        # self.serial_receiver.start()  # 启动串口接收线程
    def on_connect_button_clicked(self):
        host = "127.0.0.1"  # 替换为实际的 TCP 服务器主机
        port = 9093  # 替换为实际的 TCP 服务器端口
        self.mythread.connect_to_tcp_server(host, port)

    def disconnect_from_tcp_server(self):
        if self.mythread.tcp_socket:
            self.mythread.tcp_socket.close()
            self.mythread.tcp_socket = None
            print("TCP连接已断开")
    def closeEvent(self, event):
        # 调用线程的断开连接方法
        self.disconnect_from_tcp_server()
        # 阻止窗口关闭事件的传播，只关闭TCP连接
        event.ignore()

    def on_query_button_clicked(self):
        tool = self.combo1.currentText()
        if tool!= " ":
            # 在这里添加 TCP 发送逻辑
            self.mythread.send_tool_to_tcp(tool)
            print("tool:", tool)
        else:
            print("请选择一个工具。")


    def update_restatus(self, data):
        self.restatus.setText(data)
        
    def close(self):
        print("close")
        self.mythread.close()
        self.serial_receiver.stop()
        self.serial_receiver.wait()
    def updata_cv_frame(self):#子线程传递数据给主线程槽函数
        self.lab_show.setPixmap(self.mythread.f_pixmap)
        cv2.imshow("rgb",self.mythread.frame)
    
    def combo1_text_changed(self, text):
        self.mythread.label = text

    
    def update_serial_data(self,data):
                serial_restatus_text=data
                if len(serial_restatus_text) >= 4:
                    if serial_restatus_text[0]==0x41 and serial_restatus_text[1]==0x32 and serial_restatus_text[2]==0x30 and serial_restatus_text[3]==0x31:
                        self.restatus.setText("已取到工具！") 
                        data_hex = ' '.join(['{:02x}'.format(byte) for byte in data])
                        # 打印十六进制字符串
                        print(data_hex)
                    elif serial_restatus_text[0]==0x41 and serial_restatus_text[1]==0x33 and serial_restatus_text[2]==0x30 and serial_restatus_text[3]==0x31:
                        self.restatus.setText("工具已放至指定位置！") 
                        data_hex = ' '.join(['{:02x}'.format(byte) for byte in data])
                        # 打印十六进制字符串
                        print(data_hex)
    def Zscram(self):
        self.serial_data_sendZs.emit(b'your_bytes_data_here')

    def Qscram(self):
        self.serial_data_sendQs.emit(b'your_bytes_data_here')

    def Fscram(self):
        self.serial_data_sendFs.emit(b'your_bytes_data_here')

    def Setscram(self):
        self.serial_data_sendSs.emit(b'your_bytes_data_here')

    def Resetscram(self):
        self.serial_data_sendRs.emit(b'your_bytes_data_here')
    def relZscram(self):
        self.serial_data_sendrelZ.emit(b'your_bytes_data_here')
    def on_btQ_clicked(self):
        self.btQ_clicked = True
        self.clicked.emit()
        print("Q clicked")
    def list_to_bytes(self, lst):
    # 初始化一个字节数组来存储结果
        result = bytearray()
        for i in lst:
            result.extend(struct.pack('<H', i))
        return bytes(result)


    def Q(self, mov_data):
        # 确保 mov_data 是列表类型
        if not isinstance(mov_data, list):
            raise ValueError("mov_data must be a list")
        bytes_data = self.list_to_bytes(mov_data)
        self.serial_data_sendQ.emit(bytes_data)
 
    def on_btF_clicked(self):
        self.btF_clicked = True
        self.clicked.emit()
        print("F clicked")

    def F(self,mov_data):
        # 确保 mov_data 是列表类型
        if not isinstance(mov_data, list):
            raise ValueError("mov_data must be a list")
        bytes_data = self.list_to_bytes(mov_data)
        self.serial_data_sendF.emit(bytes_data)

    def Set(self):
        self.serial_data_sendS.emit(b'your_bytes_data_here')

    def Reset(self):
        self.serial_data_sendR.emit(b'your_bytes_data_here')


if __name__ =="__main__":

    relAllscram=[0x32,0x31,0x30,0x31]
    Allscram=[0x32,0x31,0x30,0x30]
    Access_tools=[0x32,0x32,0x30,0x31]
    Access_tools_scram=[0x32,0x32,0x30,0x30]
    Put_tools=[0x32,0x33,0x30,0x31]
    Put_tools_scram=[0x32,0x33,0x30,0x30]
    Set=[0x32,0x34,0x30,0x31]
    Set_scram=[0x32,0x34,0x30,0x30]
    Reset=[0x32,0x35,0x30,0x31]
    Reset_scram=[0x32,0x35,0x30,0x30]


    nnBlobPath = str((Path(__file__).parent / Path('best_openvino_2022.1_6shave.blob')).resolve().absolute())
    if 1 < len(sys.argv):
        arg = sys.argv[1]
        if arg == "yolo3":
            nnBlobPath = str((Path(__file__).parent / Path('best_openvino_2022.1_6shave.blob')).resolve().absolute())
        elif arg == "yolo4":
            nnBlobPath = str((Path(__file__).parent / Path('best_openvino_2022.1_6shave.blob')).resolve().absolute())
        else:
            nnBlobPath = arg
    else:
        print("Using Tiny YoloV4 model. If you wish to use Tiny YOLOv3, call 'tiny_yolo.py yolo3'")

    if not Path(nnBlobPath).exists():
        import sys

        raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')
    # QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling) #解决了Qtdesigner设计的界面与实际运行界面不一致的问题
    app = QApplication(sys.argv)#创建程序对象                                     
    w = MyWindow()  # 创建窗口对象
    w.ui.show()
    app.exec()#执行程序

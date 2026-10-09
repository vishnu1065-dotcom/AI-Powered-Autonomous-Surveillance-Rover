# AI-Powered Autonomous Surveillance Rover

## Project Overview

The AI-Powered Autonomous Surveillance Rover is an intelligent robotic system designed for real-time surveillance, person detection, face recognition, and autonomous obstacle avoidance. The project integrates Artificial Intelligence, Embedded Systems, IoT, and Robotics to create a smart mobile surveillance platform.

The system uses a Raspberry Pi 4 for AI processing and web-based surveillance, while an ESP32 manages motor control and autonomous navigation. Both devices communicate wirelessly over Wi-Fi.

## Key Features

- **AI-Based Person Detection:** Detects people in real time using YOLO and OpenCV.
- **Face Recognition:** Identifies known and unknown individuals.
- **Unknown Person Detection:** Captures and displays images of unknown individuals on the web dashboard.
- **Autonomous Navigation:** Detects obstacles using an ultrasonic sensor and avoids collisions.
- **Manual Control:** Controls rover movement through a web interface.
- **Live Surveillance:** Provides a real-time camera feed through a Flask dashboard.
- **Wireless Communication:** Connects Raspberry Pi and ESP32 through Wi-Fi.
- **Real-Time Monitoring:** Displays surveillance information and person counts on the dashboard.

## Hardware Requirements

- Raspberry Pi 4 Model B
- ESP32 development board
- USB Webcam
- Ultrasonic Sensor (HC-SR04)
- L298N Motor Driver
- DC Geared Motors
- Servo Motor
- Rover Chassis
- Battery and Power Supply

## Software and Technologies

- Python
- OpenCV
- YOLO
- Flask
- Arduino IDE
- Embedded C/C++
- Wi-Fi and TCP/IP Communication
- HTML, CSS, and JavaScript

## System Architecture

**Raspberry Pi 4**
- AI-based person detection
- Face recognition
- Live video processing
- Web surveillance dashboard
- Communication with ESP32

**ESP32**
- Motor control
- Ultrasonic distance measurement
- Autonomous obstacle avoidance
- Manual movement commands received from Raspberry Pi

## Working Principle

1. The webcam captures live video.
2. The Raspberry Pi processes the video to detect people and recognize known or unknown individuals.
3. The Flask dashboard displays live footage, detection results, and captured unknown-person images.
4. The ESP32 measures obstacle distances using the ultrasonic sensor.
5. The rover navigates autonomously or responds to manual movement commands from the web dashboard.
6. The Raspberry Pi and ESP32 exchange control information over Wi-Fi.

## Applications

- Security and surveillance
- Industrial monitoring
- Smart buildings
- Restricted-area monitoring
- Robotics and embedded AI research

## Learning Outcomes

This project provided practical experience in embedded system integration, computer vision, AI-based detection, wireless communication, web development, motor control, and autonomous robotics.

## Future Enhancements

- Thermal imaging for low-light surveillance
- GPS-based outdoor navigation
- Improved person tracking and re-identification
- Predictive maintenance using motor and encoder data
- Cloud-based monitoring and notifications

## Conclusion

The AI-Powered Autonomous Surveillance Rover demonstrates how AI, embedded systems, and wireless communication can be integrated to develop an intelligent mobile surveillance platform with autonomous navigation and remote monitoring capabilities.

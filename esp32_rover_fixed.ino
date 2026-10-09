#include <WiFi.h>
#include <WebServer.h>
#include <ESP32Servo.h>

// ======================================================
// WIFI  (ESP32 and Raspberry Pi MUST be on the same WiFi)
// ======================================================

const char* WIFI_SSID     = "INKER_ROBOTICS.....!!!!!";
const char* WIFI_PASSWORD = "Inker.Robotics@2022";

// HTTP server on port 80  (the Raspberry Pi code uses
// http://ESP32_IP:80/  and  http://ESP32_IP:80/command?cmd=XXXX)
WebServer server(80);


// ======================================================
// MDD20 MOTOR DRIVER
// ======================================================

// Motor 1 = LEFT MOTOR
#define M1_PWM 25
#define M1_DIR 26

// Motor 2 = RIGHT MOTOR
#define M2_PWM 27
#define M2_DIR 14


// ======================================================
// HC-SR04
// ======================================================

#define TRIG_PIN 5
#define ECHO_PIN 18


// ======================================================
// SERVO
// ======================================================

#define SERVO_PIN 19

Servo scanner;

#define CENTER_ANGLE 90
#define LEFT_ANGLE   150
#define RIGHT_ANGLE  30


// ======================================================
// MOTOR SETTINGS
// ======================================================

const int FORWARD_SPEED  = 180;
const int BACKWARD_SPEED = 170;
const int TURN_SPEED     = 180;


// ======================================================
// OBSTACLE SETTINGS
// ======================================================

const int OBSTACLE_DISTANCE = 25;


// ======================================================
// MODE
// ======================================================

volatile bool autoMode = true;


// ======================================================
// SMART DELAY
// ======================================================
// A normal delay() freezes the ESP32, so the Raspberry Pi
// gets no reply and shows OFFLINE.  smartDelay() keeps
// answering HTTP requests while waiting.
// It returns false if the mode was changed to MANUAL,
// so the auto routine can stop immediately.

bool smartDelay(unsigned long ms)
{
  unsigned long start = millis();

  while (millis() - start < ms)
  {
    server.handleClient();

    if (!autoMode)
    {
      return false;
    }

    delay(2);
  }

  return autoMode;
}


// ======================================================
// MOTOR FUNCTIONS
// ======================================================

void stopMotors()
{
  analogWrite(M1_PWM, 0);
  analogWrite(M2_PWM, 0);
}


void forward()
{
  digitalWrite(M1_DIR, LOW);
  digitalWrite(M2_DIR, LOW);

  analogWrite(M1_PWM, FORWARD_SPEED);
  analogWrite(M2_PWM, FORWARD_SPEED);
}


void backward()
{
  digitalWrite(M1_DIR, HIGH);
  digitalWrite(M2_DIR, HIGH);

  analogWrite(M1_PWM, BACKWARD_SPEED);
  analogWrite(M2_PWM, BACKWARD_SPEED);
}


void turnLeft()
{
  // LEFT MOTOR backward, RIGHT MOTOR forward
  digitalWrite(M1_DIR, HIGH);
  digitalWrite(M2_DIR, LOW);

  analogWrite(M1_PWM, TURN_SPEED);
  analogWrite(M2_PWM, TURN_SPEED);
}


void turnRight()
{
  // LEFT MOTOR forward, RIGHT MOTOR backward
  digitalWrite(M1_DIR, LOW);
  digitalWrite(M2_DIR, HIGH);

  analogWrite(M1_PWM, TURN_SPEED);
  analogWrite(M2_PWM, TURN_SPEED);
}


// ======================================================
// ULTRASONIC DISTANCE
// ======================================================

long getDistance()
{
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(3);

  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);

  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000);

  // No echo
  if (duration == 0)
  {
    return 400;
  }

  return duration * 0.0343 / 2;
}


// ======================================================
// SERVO SCAN
// ======================================================

long checkLeft()
{
  scanner.write(LEFT_ANGLE);
  smartDelay(500);

  long distance = getDistance();

  Serial.print("LEFT = ");
  Serial.print(distance);
  Serial.println(" cm");

  scanner.write(CENTER_ANGLE);
  smartDelay(200);

  return distance;
}


long checkRight()
{
  scanner.write(RIGHT_ANGLE);
  smartDelay(500);

  long distance = getDistance();

  Serial.print("RIGHT = ");
  Serial.print(distance);
  Serial.println(" cm");

  scanner.write(CENTER_ANGLE);
  smartDelay(200);

  return distance;
}


// ======================================================
// OBSTACLE AVOIDANCE (AUTO MODE)
// ======================================================

void obstacleAvoidance()
{
  long frontDistance = getDistance();

  // ---------------- NO OBSTACLE ----------------
  if (frontDistance > OBSTACLE_DISTANCE)
  {
    forward();
    return;
  }

  // ---------------- OBSTACLE FOUND ----------------
  Serial.println();
  Serial.println("==============================");
  Serial.print("OBSTACLE DETECTED  FRONT = ");
  Serial.print(frontDistance);
  Serial.println(" cm");
  Serial.println("==============================");

  stopMotors();
  if (!smartDelay(200)) { stopMotors(); return; }

  // ---------------- MOVE BACKWARD ----------------
  Serial.println("MOVING BACKWARD");

  backward();
  if (!smartDelay(450)) { stopMotors(); return; }

  stopMotors();
  if (!smartDelay(250)) { stopMotors(); return; }

  // ---------------- SCAN ----------------
  Serial.println("SCANNING LEFT");
  long leftDistance = checkLeft();
  if (!autoMode) { stopMotors(); return; }

  Serial.println("SCANNING RIGHT");
  long rightDistance = checkRight();
  if (!autoMode) { stopMotors(); return; }

  Serial.print("LEFT  : ");
  Serial.print(leftDistance);
  Serial.println(" cm");

  Serial.print("RIGHT : ");
  Serial.print(rightDistance);
  Serial.println(" cm");

  // ---------------- LEFT IS CLEARER ----------------
  if (leftDistance > rightDistance &&
      leftDistance > OBSTACLE_DISTANCE)
  {
    Serial.println("DECISION: TURN LEFT");

    turnLeft();
    if (!smartDelay(550)) { stopMotors(); return; }

    stopMotors();
    smartDelay(200);
    return;
  }

  // ---------------- RIGHT IS CLEARER ----------------
  if (rightDistance > leftDistance &&
      rightDistance > OBSTACLE_DISTANCE)
  {
    Serial.println("DECISION: TURN RIGHT");

    turnRight();
    if (!smartDelay(550)) { stopMotors(); return; }

    stopMotors();
    smartDelay(200);
    return;
  }

  // ---------------- BOTH SIDES BLOCKED ----------------
  Serial.println("BOTH SIDES BLOCKED - REVERSE AND TURN RIGHT");

  backward();
  if (!smartDelay(600)) { stopMotors(); return; }

  stopMotors();
  if (!smartDelay(200)) { return; }

  turnRight();
  if (!smartDelay(750)) { stopMotors(); return; }

  stopMotors();
  smartDelay(200);
}


// ======================================================
// COMMAND HANDLING
// ======================================================
// Returns the text sent back to the Raspberry Pi.
// Anything other than a 2xx HTTP status makes the Pi
// treat the command as failed.

String processCommand(String command)
{
  command.trim();
  command.toUpperCase();

  Serial.print("PI COMMAND: ");
  Serial.println(command);

  // ---------------- AUTO ----------------
  if (command == "AUTO")
  {
    stopMotors();
    autoMode = true;

    Serial.println("AUTO MODE ENABLED");
    return "AUTO_OK";
  }

  // ---------------- MANUAL ----------------
  if (command == "MANUAL")
  {
    autoMode = false;
    stopMotors();

    Serial.println("MANUAL MODE ENABLED");
    return "MANUAL_OK";
  }

  // ---------------- STOP ----------------
  if (command == "STOP")
  {
    stopMotors();
    return "STOP_OK";
  }

  // ---------------- MOVEMENT (MANUAL ONLY) ----------------
  if (command == "FORWARD" ||
      command == "BACKWARD" ||
      command == "LEFT" ||
      command == "RIGHT")
  {
    if (autoMode)
    {
      return "IGNORED_IN_AUTO";
    }

    if (command == "FORWARD")       forward();
    else if (command == "BACKWARD") backward();
    else if (command == "LEFT")     turnLeft();
    else if (command == "RIGHT")    turnRight();

    return command + "_OK";
  }

  return "UNKNOWN_COMMAND";
}


// ======================================================
// HTTP HANDLERS
// ======================================================

// GET /   ->  ESP32_OK   (used by the Pi online check)
void handleRoot()
{
  server.send(200, "text/plain", "ESP32_OK");
}


// GET /command?cmd=FORWARD
void handleCommand()
{
  if (!server.hasArg("cmd"))
  {
    server.send(400, "text/plain", "MISSING_CMD");
    return;
  }

  String result = processCommand(server.arg("cmd"));

  server.send(200, "text/plain", result);
}


void handleNotFound()
{
  server.send(404, "text/plain", "NOT_FOUND");
}


// ======================================================
// WIFI
// ======================================================

void connectWiFi()
{
  Serial.println();
  Serial.print("Connecting to WiFi: ");
  Serial.println(WIFI_SSID);

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);            // faster, more reliable replies
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long start = millis();

  while (WiFi.status() != WL_CONNECTED)
  {
    delay(500);
    Serial.print(".");

    // retry every 20 seconds if it cannot connect
    if (millis() - start > 20000)
    {
      Serial.println();
      Serial.println("Retrying WiFi...");
      WiFi.disconnect();
      delay(200);
      WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
      start = millis();
    }
  }

  Serial.println();
  Serial.println("WiFi connected");

  Serial.print("ESP32 IP ADDRESS: ");
  Serial.println(WiFi.localIP());

  Serial.println(">>> Put this IP in the Raspberry Pi code:");
  Serial.print(">>> ESP32_IP = \"");
  Serial.print(WiFi.localIP());
  Serial.println("\"");
}


// ======================================================
// SETUP
// ======================================================

void setup()
{
  Serial.begin(115200);

  // ---------------- MOTOR PINS ----------------
  pinMode(M1_PWM, OUTPUT);
  pinMode(M1_DIR, OUTPUT);
  pinMode(M2_PWM, OUTPUT);
  pinMode(M2_DIR, OUTPUT);

  // ---------------- ULTRASONIC ----------------
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  // ---------------- SERVO ----------------
  scanner.setPeriodHertz(50);
  scanner.attach(SERVO_PIN, 500, 2400);
  scanner.write(CENTER_ANGLE);

  stopMotors();

  // ---------------- WIFI ----------------
  connectWiFi();

  // ---------------- HTTP SERVER ----------------
  server.on("/", handleRoot);
  server.on("/command", handleCommand);
  server.onNotFound(handleNotFound);

  server.begin();

  Serial.println("HTTP SERVER STARTED  (port 80)");

  // ---------------- DEFAULT MODE ----------------
  autoMode = true;

  Serial.println();
  Serial.println("==============================");
  Serial.println("ROVER READY - AUTO MODE");
  Serial.println("==============================");
}


// ======================================================
// LOOP
// ======================================================

void loop()
{
  // Reconnect WiFi if it dropped
  if (WiFi.status() != WL_CONNECTED)
  {
    stopMotors();
    Serial.println("WiFi lost - reconnecting...");
    connectWiFi();
  }

  // Answer the Raspberry Pi
  server.handleClient();

  // AUTO mode: obstacle avoidance
  // MANUAL mode: the Pi controls the motors
  if (autoMode)
  {
    obstacleAvoidance();
  }

  delay(10);
}

#include <Arduino.h>
#include "BoardCheck.h"
#include "driver/gpio.h"

// פיני מנועים (CNC Shield V3 מול ESP32-S3)
const int MOTOR_A_STEP = 1;   // X-STEP
const int MOTOR_A_DIR  = 2;   // X-DIR

const int MOTOR_B_STEP = 41;  // Y-STEP
const int MOTOR_B_DIR  = 42;  // Y-DIR

const int EN_PIN       = 21;  // D8 / Enable
const int VIO_PIN      = 38;  // 3.3V לוגיקה לדרייברים (Shield 5V/VIO), ~1mA

// פיני מפסקי גבול (חבר בין הפין ל-GND)
const int LIMIT_X_PIN  = 39;  // שנה לפי ה-GPIO שבחרת
const int LIMIT_Y_PIN  = 40;  // שנה לפי ה-GPIO שבחרת

// הגדרות מהירות ומרחק נסיגה (Back-off)
const int HOMING_DELAY_US = 500;  // מהירות תנועה בטוחה לאיפוס
const int BACKOFF_STEPS   = 300;  // צעדים להתרחקות מהמתג לאחר לחיצה
const long MAX_TRAVEL     = 25000; // הגנת תוכנה מפני תקיעה אם מפסק לא מגיב

void setup() {
  // ראשון: מתח לוגיקה לדרייברים לפני STEP/DIR/EN, כדי שלא ידלוף דרך דיודות ההגנה
  pinMode(VIO_PIN, OUTPUT);
  gpio_set_drive_capability((gpio_num_t)VIO_PIN, GPIO_DRIVE_CAP_3);
  digitalWrite(VIO_PIN, HIGH);
  delay(5);

  Serial.begin(115200);
  boardCheck();
  Serial.println("--- CoreXY Homing Sequence Started ---");

  pinMode(MOTOR_A_STEP, OUTPUT);
  pinMode(MOTOR_A_DIR, OUTPUT);
  pinMode(MOTOR_B_STEP, OUTPUT);
  pinMode(MOTOR_B_DIR, OUTPUT);
  pinMode(EN_PIN, OUTPUT);

  // הגדרת מפסקים עם Pull-up פנימי
  pinMode(LIMIT_X_PIN, INPUT_PULLUP);
  pinMode(LIMIT_Y_PIN, INPUT_PULLUP);

  // הפעלת דרייברים
  digitalWrite(EN_PIN, LOW);
  delay(150);

  // הרצת רוטינת האיפוס פעם אחת בעלייה
  runHoming();
}

// פונקציית צעד בודד קרטזי במנגנון CoreXY
void stepCartesian(int dirX, int dirY) {
  long stepA = dirX + dirY;
  long stepB = dirX - dirY;

  digitalWrite(MOTOR_A_DIR, stepA >= 0 ? HIGH : LOW);
  digitalWrite(MOTOR_B_DIR, stepB >= 0 ? HIGH : LOW);

  bool pulseA = (stepA != 0);
  bool pulseB = (stepB != 0);

  if (pulseA) digitalWrite(MOTOR_A_STEP, HIGH);
  if (pulseB) digitalWrite(MOTOR_B_STEP, HIGH);

  delayMicroseconds(HOMING_DELAY_US);

  if (pulseA) digitalWrite(MOTOR_A_STEP, LOW);
  if (pulseB) digitalWrite(MOTOR_B_STEP, LOW);

  delayMicroseconds(HOMING_DELAY_US);
}

void runHoming() {
  Serial.println("Homing Axis Y...");
  long stepsCount = 0;

  // 1. תנועה לכיוון מתג Y עד לחיצה (LOW)
  while (digitalRead(LIMIT_Y_PIN) == HIGH) {
    stepCartesian(0, +1); // אם העגלה נוסעת לכיוון ההפוך מהמתג, החלף ל- (0, 1)
    stepsCount++;
    if (stepsCount > MAX_TRAVEL) {
      Serial.println("Error: Y Limit Switch not reached!");
      return;
    }
  }

  // נסיגה קלה לאחור לשחרור המתג
  delay(100);
  for (int i = 0; i < BACKOFF_STEPS; i++) {
    stepCartesian(0, -1);
  }
  delay(200);

  Serial.println("Homing Axis X...");
  stepsCount = 0;

  // 2. תנועה לכיוון מתג X עד לחיצה (LOW)
  while (digitalRead(LIMIT_X_PIN) == HIGH) {
    stepCartesian(+1, 0); // אם העגלה נוסעת לכיוון ההפוך מהמתג, החלף ל- (1, 0)
    stepsCount++;
    if (stepsCount > MAX_TRAVEL) {
      Serial.println("Error: X Limit Switch not reached!");
      return;
    }
  }

  // נסיגה קלה לאחור לשחרור המתג
  delay(100);
  for (int i = 0; i < BACKOFF_STEPS; i++) {
    stepCartesian(-1, 0);
  }
  delay(200);

  Serial.println("Homing Completed Successfully! Coordinate system set to (0,0).");
}

void loop() {
  // המערכת מאופסת ונמצאת במצב המתנה
}
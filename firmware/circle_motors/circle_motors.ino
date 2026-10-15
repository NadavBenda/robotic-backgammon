#include <Arduino.h>

// הגדרת פינים (CNC Shield V3 מול ESP32-S3)
const int MOTOR_A_STEP = 1;   // X-STEP
const int MOTOR_A_DIR  = 2;   // X-DIR

const int MOTOR_B_STEP = 41;  // Y-STEP
const int MOTOR_B_DIR  = 42;  // Y-DIR

const int EN_PIN       = 21;  // D8 / Enable

// פרמטרים מעודכנים: רדיוס פי 3 ומהירות גבוהה בהרבה
const float RADIUS_STEPS = 4400.0; // רדיוס מוגדל פי 3 (היה 800)
const int TOTAL_SEGMENTS = 360;    // 360 קטעים לשמירה על עיגול מושלם
const int STEP_DELAY_US  = 120;    // מהירות גבוהה (היה 900 מיקרו-שניות)

void setup() {
  Serial.begin(115200);
  Serial.println("--- CoreXY Large & Fast Circle Started ---");

  pinMode(MOTOR_A_STEP, OUTPUT);
  pinMode(MOTOR_A_DIR, OUTPUT);
  pinMode(MOTOR_B_STEP, OUTPUT);
  pinMode(MOTOR_B_DIR, OUTPUT);
  pinMode(EN_PIN, OUTPUT);

  // הפעלת הדרייברים (Active LOW)
  digitalWrite(EN_PIN, LOW);
  delay(150);
}

// פונקציית תנועת CoreXY לפי הפרשי צעדים (dX, dY)
void moveCoreXY(long dX, long dY) {
  // קינמטיקת CoreXY
  long stepsA = dX + dY;
  long stepsB = dX - dY;

  // קביעת כיווני סיבוב
  digitalWrite(MOTOR_A_DIR, stepsA >= 0 ? HIGH : LOW);
  digitalWrite(MOTOR_B_DIR, stepsB >= 0 ? HIGH : LOW);

  stepsA = labs(stepsA);
  stepsB = labs(stepsB);

  long maxSteps = max(stepsA, stepsB);
  if (maxSteps == 0) return;

  long countA = 0;
  long countB = 0;

  // סנכרון פולסים (Bresenham)
  for (long i = 0; i < maxSteps; i++) {
    countA += stepsA;
    countB += stepsB;

    bool stepA = false;
    bool stepB = false;

    if (countA >= maxSteps) {
      digitalWrite(MOTOR_A_STEP, HIGH);
      stepA = true;
      countA -= maxSteps;
    }
    if (countB >= maxSteps) {
      digitalWrite(MOTOR_B_STEP, HIGH);
      stepB = true;
      countB -= maxSteps;
    }

    delayMicroseconds(STEP_DELAY_US);

    if (stepA) digitalWrite(MOTOR_A_STEP, LOW);
    if (stepB) digitalWrite(MOTOR_B_STEP, LOW);

    delayMicroseconds(STEP_DELAY_US);
  }
}

void loop() {
  Serial.println("Drawing 3x Larger & Faster Circle...");

  float lastX = RADIUS_STEPS;
  float lastY = 0.0;

  for (int i = 1; i <= TOTAL_SEGMENTS; i++) {
    float angle = (2.0 * PI * i) / TOTAL_SEGMENTS;
    
    float targetX = RADIUS_STEPS * cos(angle);
    float targetY = RADIUS_STEPS * sin(angle);

    long dX = lround(targetX - lastX);
    long dY = lround(targetY - lastY);

    moveCoreXY(dX, dY);

    lastX = targetX;
    lastY = targetY;
  }

  Serial.println("Completed. Waiting 3 seconds...");
  delay(3000);
}
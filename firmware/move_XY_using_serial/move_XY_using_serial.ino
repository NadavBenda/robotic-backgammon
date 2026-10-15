#include <Arduino.h>

// פיני מנועים
const int MOTOR_A_STEP = 1;   // X-STEP
const int MOTOR_A_DIR  = 2;   // X-DIR
const int MOTOR_B_STEP = 41;  // Y-STEP
const int MOTOR_B_DIR  = 42;  // Y-DIR
const int EN_PIN       = 21;  // Enable

// פיני מפסקי גבול
const int LIMIT_X_PIN  = 15;
const int LIMIT_Y_PIN  = 16;

// כיול תנועה
const float STEPS_PER_CM  = 400.0; // 400 צעדים לכל 1 ס"מ
const float MAX_LIMIT_CM  = 49.0;  // גבול עליון למגרש בס"מ
const int HOMING_DELAY_US = 500;
const int BACKOFF_STEPS   = 300;
const long MAX_TRAVEL     = 25000;

// הגדרות פרופיל תנועה (בצעדים לשנייה)
const float START_SPEED_STEPS = 800.0;   // מהירות זינוק התחלתית (20 מ"מ/שנייה)
const float MAX_SPEED_STEPS   = 4800.0;  // מהירות מקסימלית (120 מ"מ/שנייה)
const float ACCEL_STEPS       = 24000.0; // תאוצה מקסימלית (600 מ"מ/שנייה בריבוע)

// מעקב מיקום נוכחי בצעדים
long currentXSteps = 0;
long currentYSteps = 0;

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
  while (digitalRead(LIMIT_Y_PIN) == HIGH) {
    stepCartesian(0, +1);
    stepsCount++;
    if (stepsCount > MAX_TRAVEL) {
      Serial.println("Error: Y Limit Switch not reached!");
      return;
    }
  }
  delay(100);
  for (int i = 0; i < BACKOFF_STEPS; i++) {
    stepCartesian(0, -1);
  }
  delay(200);

  Serial.println("Homing Axis X...");
  stepsCount = 0;
  while (digitalRead(LIMIT_X_PIN) == HIGH) {
    stepCartesian(+1, 0);
    stepsCount++;
    if (stepsCount > MAX_TRAVEL) {
      Serial.println("Error: X Limit Switch not reached!");
      return;
    }
  }
  delay(100);
  for (int i = 0; i < BACKOFF_STEPS; i++) {
    stepCartesian(-1, 0);
  }
  delay(200);

  currentXSteps = 0;
  currentYSteps = 0;
  Serial.println("Homing Completed! Origin set to (0.0, 0.0) cm");
}

// תנועה מתקדמת: Bresenham + פרופיל תאוצה טרפזי (האצה-שיוט-האטה)
void moveTrapezoidal(long dX, long dY) {
  long motorDX = -dX;
  long motorDY = -dY;

  long stepsA = motorDX + motorDY;
  long stepsB = motorDX - motorDY;

  digitalWrite(MOTOR_A_DIR, stepsA >= 0 ? HIGH : LOW);
  digitalWrite(MOTOR_B_DIR, stepsB >= 0 ? HIGH : LOW);

  stepsA = labs(stepsA);
  stepsB = labs(stepsB);

  long totalSteps = max(stepsA, stepsB);
  if (totalSteps == 0) return;

  // חישוב מספר הצעדים הנדרשים להגעה למהירות מקסימלית: s = (v^2 - v0^2) / (2a)
  long accelSteps = lround(((MAX_SPEED_STEPS * MAX_SPEED_STEPS) - (START_SPEED_STEPS * START_SPEED_STEPS)) / (2.0 * ACCEL_STEPS));
  if (accelSteps < 1) accelSteps = 1;

  // אם המרחק קצר מדי להגעה למהירות שיא, התאוצה תימשך בדיוק חצי מהדרך
  if (accelSteps * 2 > totalSteps) {
    accelSteps = totalSteps / 2;
  }
  long decelStart = totalSteps - accelSteps;

  float currentSpeed = START_SPEED_STEPS;
  long countA = 0;
  long countB = 0;

  for (long i = 0; i < totalSteps; i++) {
    // בדיקת Hard Limit בזמן נסיעה חזרה
    if (dX < 0 && digitalRead(LIMIT_X_PIN) == LOW) {
      Serial.println("\n[HARD LIMIT HIT] X switch triggered! Emergency Stop.");
      currentXSteps = 0;
      return;
    }
    if (dY < 0 && digitalRead(LIMIT_Y_PIN) == LOW) {
      Serial.println("\n[HARD LIMIT HIT] Y switch triggered! Emergency Stop.");
      currentYSteps = 0;
      return;
    }

    // פרופיל מהירות: האצה / שיוט / האטה
    if (i < accelSteps) {
      currentSpeed = sqrt((START_SPEED_STEPS * START_SPEED_STEPS) + (2.0 * ACCEL_STEPS * i));
      if (currentSpeed > MAX_SPEED_STEPS) currentSpeed = MAX_SPEED_STEPS;
    } else if (i >= decelStart) {
      long remaining = totalSteps - i;
      currentSpeed = sqrt((START_SPEED_STEPS * START_SPEED_STEPS) + (2.0 * ACCEL_STEPS * remaining));
      if (currentSpeed < START_SPEED_STEPS) currentSpeed = START_SPEED_STEPS;
    } else {
      currentSpeed = MAX_SPEED_STEPS;
    }

    // השהיה לחצי פולס במיקרו-שניות
    int halfDelayUs = lround((1000000.0 / currentSpeed) / 2.0);
    if (halfDelayUs < 50) halfDelayUs = 50;

    countA += stepsA;
    countB += stepsB;

    bool pulseA = false;
    bool pulseB = false;

    if (countA >= totalSteps) {
      digitalWrite(MOTOR_A_STEP, HIGH);
      pulseA = true;
      countA -= totalSteps;
    }
    if (countB >= totalSteps) {
      digitalWrite(MOTOR_B_STEP, HIGH);
      pulseB = true;
      countB -= totalSteps;
    }

    delayMicroseconds(halfDelayUs);

    if (pulseA) digitalWrite(MOTOR_A_STEP, LOW);
    if (pulseB) digitalWrite(MOTOR_B_STEP, LOW);

    delayMicroseconds(halfDelayUs);
  }

  currentXSteps += dX;
  currentYSteps += dY;

  Serial.print("Position: X=");
  Serial.print((float)currentXSteps / STEPS_PER_CM);
  Serial.print(" cm, Y=");
  Serial.print((float)currentYSteps / STEPS_PER_CM);
  Serial.println(" cm");
}

void parseAndExecute(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  char type = cmd.charAt(0);

  // פקודה אבסולוטית: G <X> <Y>
  if (type == 'G' || type == 'g') {
    int firstSpace = cmd.indexOf(' ');
    int secondSpace = cmd.indexOf(' ', firstSpace + 1);
    if (firstSpace != -1 && secondSpace != -1) {
      float targetX = cmd.substring(firstSpace + 1, secondSpace).toFloat();
      float targetY = cmd.substring(secondSpace + 1).toFloat();

      // גבולות עליונים ותחתונים (0 עד 49 ס"מ)
      if (targetX < 0.0 || targetX > MAX_LIMIT_CM || targetY < 0.0 || targetY > MAX_LIMIT_CM) {
        Serial.println("Error: Coordinate out of bounds! Range is 0.0 to 49.0 cm.");
        return;
      }

      long targetXSteps = lround(targetX * STEPS_PER_CM);
      long targetYSteps = lround(targetY * STEPS_PER_CM);

      long dX = targetXSteps - currentXSteps;
      long dY = targetYSteps - currentYSteps;

      moveTrapezoidal(dX, dY);
    }
  }
  // פקודה יחסית: R <dX> <dY>
  else if (type == 'R' || type == 'r') {
    int firstSpace = cmd.indexOf(' ');
    int secondSpace = cmd.indexOf(' ', firstSpace + 1);
    if (firstSpace != -1 && secondSpace != -1) {
      float deltaX = cmd.substring(firstSpace + 1, secondSpace).toFloat();
      float deltaY = cmd.substring(secondSpace + 1).toFloat();

      long dX = lround(deltaX * STEPS_PER_CM);
      long dY = lround(deltaY * STEPS_PER_CM);

      float newX = (float)(currentXSteps + dX) / STEPS_PER_CM;
      float newY = (float)(currentYSteps + dY) / STEPS_PER_CM;

      // בדיקת חריגה מגבולות 0 עד 49 ס"מ
      if (newX < 0.0 || newX > MAX_LIMIT_CM || newY < 0.0 || newY > MAX_LIMIT_CM) {
        Serial.println("Error: Relative movement would exceed bounds (0.0 to 49.0 cm)!");
        return;
      }

      moveTrapezoidal(dX, dY);
    }
  }
  else if (type == 'H' || type == 'h') {
    runHoming();
  }
}

void setup() {
  Serial.begin(115200);

  pinMode(MOTOR_A_STEP, OUTPUT);
  pinMode(MOTOR_A_DIR, OUTPUT);
  pinMode(MOTOR_B_STEP, OUTPUT);
  pinMode(MOTOR_B_DIR, OUTPUT);
  pinMode(EN_PIN, OUTPUT);

  pinMode(LIMIT_X_PIN, INPUT_PULLUP);
  pinMode(LIMIT_Y_PIN, INPUT_PULLUP);

  digitalWrite(EN_PIN, LOW);
  delay(150);

  runHoming();

  Serial.println("Ready. Bounds: 0.0 to 49.0 cm.");
  Serial.println("Commands: G <X> <Y>, R <dX> <dY>, H");
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    parseAndExecute(input);
  }
}
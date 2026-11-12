/*
 * board — AI-bg main firmware (ESP32-S3): CoreXY motion + electromagnet + camera, all over USB serial.
 *
 * Protocol (921600 baud; on the native USB port the baud is ignored):
 *   PC sends one command per line. The board answers with zero or more lines and ends with
 *   exactly ONE of:  DONE   or   ERR <reason>
 *   The PC must not send the next command before it gets DONE / ERR.
 *
 *   H              home (Y then X), then slowly off each switch until it releases   ERR HOME_X / HOME_Y / HOME_X_STUCK / HOME_Y_STUCK
 *   G <x> <y>      absolute move, mm                       ERR NOT_HOMED / BOUNDS / LIMIT_X / LIMIT_Y
 *   R <dx> <dy>    relative move, mm                       same
 *   M 1 / M 0      electromagnet on / off; M 1 answers DONE after 300 ms (the checker centres
 *                  over the magnet), M 0 after 500 ms (it is let go)  ERR MAG_HOT
 *   C              capture: "IMG <n>" line, then exactly n JPEG bytes, then DONE   ERR CAM
 *                  (allowed with the magnet on, but warns "! CAPTURE_WITH_MAG_ON")
 *   P <v> <a>      drag profile (magnet on): top speed mm/s, accel mm/s^2, default 180 / 800
 *   F <v> <a>      free profile (magnet off), default 150 / 1200; starts at 40 mm/s so the
 *                  carriage leaves a checker sharply. Both up to 600 / 5000, kept until reset  ERR ARGS
 *   ?              "POS <x> <y> HOMED <0/1> MAG <0/1> HEAT <%> CAM <0/1> DRAG_V <v> DRAG_A <a> FREE_V <v> FREE_A <a>",
 *                  then DONE
 *
 *   Other lines from the board:
 *     READY          once, at boot (the board does NOT home by itself; send H)
 *     # <text>       log, ignore
 *     ! <event>      async warning, e.g. "! MAG_TIMEOUT"
 *
 * Board: Freenove ESP32-S3 WROOM CAM (N16R8), settings enforced by BoardCheck.h.
 * Pins: X STEP/DIR 1/2, Y STEP/DIR 41/42, EN 21, driver VIO 38, limits 39/40, magnet 47, camera 4-18.
 * GPIO 2 is X DIR here, so the camera test's LED on GPIO 2 is not used.
 */

#include <Arduino.h>
#include "BoardCheck.h"
#include "driver/gpio.h"
#include "Magnet.h"
#include "esp_camera.h"

// ---------------------------------------------------------------------------
// פינים
// ---------------------------------------------------------------------------
const int MOTOR_A_STEP = 1;   // X-STEP
const int MOTOR_A_DIR  = 2;   // X-DIR
const int MOTOR_B_STEP = 41;  // Y-STEP
const int MOTOR_B_DIR  = 42;  // Y-DIR
const int EN_PIN       = 21;  // Enable (LOW = פעיל)
const int VIO_PIN      = 38;  // 3.3V לוגיקה לדרייברים (Shield 5V/VIO), ~1mA
const int LIMIT_X_PIN  = 39;
const int LIMIT_Y_PIN  = 40;
const int MAG_PIN      = 47;

// מצלמה (זהה ל-camera_modes_test)
#define XCLK_GPIO_NUM  15
#define SIOD_GPIO_NUM   4
#define SIOC_GPIO_NUM   5
#define Y9_GPIO_NUM    16
#define Y8_GPIO_NUM    17
#define Y7_GPIO_NUM    18
#define Y6_GPIO_NUM    12
#define Y5_GPIO_NUM    10
#define Y4_GPIO_NUM     8
#define Y3_GPIO_NUM     9
#define Y2_GPIO_NUM    11
#define VSYNC_GPIO_NUM  6
#define HREF_GPIO_NUM   7
#define PCLK_GPIO_NUM  13

// ---------------------------------------------------------------------------
// כיול תנועה
// ---------------------------------------------------------------------------
const float STEPS_PER_MM = 40.0;   // 400 צעדים לס"מ
const float MAX_X_MM     = 410.0;  // גבול תוכנה (נמדד 11.11: קצה הלוח ב-x 410)
const float MAX_Y_MM     = 420.0;  // גבול תוכנה (נמדד 11.11: קצה הלוח ב-y 420)
const int HOMING_DELAY_US = 500;
const int BACKOFF_DELAY_US   = 2000;  // יציאה איטית מהמפסק: כ-6 מ"מ/ש'
const int BACKOFF_MAX_STEPS  = 2000;  // 50 מ"מ. אם המפסק לא השתחרר עד אז, זו תקלה
const int BACKOFF_MARGIN     = 8;     // 0.2 מ"מ אחרי השחרור, כדי שלא ירצד על הסף
const long MAX_TRAVEL     = 25000;

const float START_SPEED_STEPS = 800.0;   // 20 מ"מ/ש' (עם אבן)
const float FREE_START_STEPS  = 1600.0;  // 40 מ"מ/ש' (בלי אבן: יציאה חדה מתחת לאבן שהונחה)
const int   RELEASE_MS        = 500;     // אחרי M 0: הסליל נפרק והאבן משתחררת לפני התנועה הבאה (12.11: 250 לא הספיק)
const int   GRAB_MS           = 300;     // אחרי M 1: האבן מתמרכזת מעל המגנט לפני התנועה
// פרופיל חופשי (א"מ כבוי). משתנה בפקודה F
float freeMaxSteps   = 150.0 * STEPS_PER_MM;   // 150 מ"מ/ש'
float freeAccelSteps = 1200.0 * STEPS_PER_MM;  // 1200 מ"מ/ש²
// פרופיל גרירה כשהא"מ דולק: איטי ועדין יותר כדי שהאבן לא תישאר מאחור (משתנה בפקודה P)
float dragMaxSteps   = 180.0 * STEPS_PER_MM;  // 180 מ"מ/ש' (drag_test 12.11, Vref 0.86: 200 עבר, 250 חורק/נכשל)
float dragAccelSteps = 800.0 * STEPS_PER_MM;  // 800 מ"מ/ש²

Magnet mag(MAG_PIN, 40000);  // מקסימום 40 ש' ברצף, תקציב חום 60 ש'

long currentXSteps = 0;
long currentYSteps = 0;
bool homed = false;
bool camReady = false;

// ---------------------------------------------------------------------------
// תנועה
// ---------------------------------------------------------------------------
void stepCartesian(int dirX, int dirY, int halfDelayUs = HOMING_DELAY_US) {
  long stepA = dirX + dirY;
  long stepB = dirX - dirY;
  digitalWrite(MOTOR_A_DIR, stepA >= 0 ? HIGH : LOW);
  digitalWrite(MOTOR_B_DIR, stepB >= 0 ? HIGH : LOW);
  bool pulseA = (stepA != 0);
  bool pulseB = (stepB != 0);
  if (pulseA) digitalWrite(MOTOR_A_STEP, HIGH);
  if (pulseB) digitalWrite(MOTOR_B_STEP, HIGH);
  delayMicroseconds(halfDelayUs);
  if (pulseA) digitalWrite(MOTOR_A_STEP, LOW);
  if (pulseB) digitalWrite(MOTOR_B_STEP, LOW);
  delayMicroseconds(halfDelayUs);
}

// יוצא לאט מהמפסק עד שהוא משתחרר, ועוד BACKOFF_MARGIN צעדים. מחזיר false אם לא השתחרר.
bool backOff(int dirX, int dirY, int pin) {
  int n = 0;
  while (digitalRead(pin) == LOW) {
    stepCartesian(dirX, dirY, BACKOFF_DELAY_US);
    if (++n > BACKOFF_MAX_STEPS) return false;
  }
  for (int i = 0; i < BACKOFF_MARGIN; i++) stepCartesian(dirX, dirY, BACKOFF_DELAY_US);
  Serial.printf("# released after %.2f mm\n", (n + BACKOFF_MARGIN) / STEPS_PER_MM);
  return true;
}

// מחזיר nullptr בהצלחה, או את סיבת השגיאה
const char* runHoming() {
  homed = false;
  Serial.println("# homing Y");
  long n = 0;
  while (digitalRead(LIMIT_Y_PIN) == HIGH) {
    stepCartesian(0, +1);
    if (++n > MAX_TRAVEL) return "HOME_Y";
  }
  delay(100);
  if (!backOff(0, -1, LIMIT_Y_PIN)) return "HOME_Y_STUCK";
  delay(100);

  Serial.println("# homing X");
  n = 0;
  while (digitalRead(LIMIT_X_PIN) == HIGH) {
    stepCartesian(+1, 0);
    if (++n > MAX_TRAVEL) return "HOME_X";
  }
  delay(100);
  if (!backOff(-1, 0, LIMIT_X_PIN)) return "HOME_X_STUCK";
  delay(100);

  currentXSteps = 0;
  currentYSteps = 0;
  homed = true;
  return nullptr;
}

// Bresenham + פרופיל טרפזי. מחזיר nullptr בהצלחה, או את סיבת השגיאה.
const char* moveTrapezoidal(long dX, long dY) {
  long motorDX = -dX;
  long motorDY = -dY;
  long stepsA = motorDX + motorDY;
  long stepsB = motorDX - motorDY;
  digitalWrite(MOTOR_A_DIR, stepsA >= 0 ? HIGH : LOW);
  digitalWrite(MOTOR_B_DIR, stepsB >= 0 ? HIGH : LOW);
  stepsA = labs(stepsA);
  stepsB = labs(stepsB);

  long totalSteps = max(stepsA, stepsB);
  if (totalSteps == 0) return nullptr;

  // עם אבן על הא"מ: פרופיל הגרירה. בלי: המהיר
  const bool  drag  = mag.isOn();
  const float vMax  = drag ? dragMaxSteps : freeMaxSteps;
  const float accel = drag ? dragAccelSteps : freeAccelSteps;
  const float v0    = min(drag ? START_SPEED_STEPS : FREE_START_STEPS, vMax);
  long accelSteps = lround(((vMax * vMax) - (v0 * v0)) / (2.0 * accel));
  if (accelSteps < 1) accelSteps = 1;
  if (accelSteps * 2 > totalSteps) accelSteps = totalSteps / 2;
  long decelStart = totalSteps - accelSteps;

  float currentSpeed = v0;
  long countA = 0, countB = 0;
  long doneX = 0, doneY = 0;  // מיקום בפועל, למקרה של עצירה באמצע

  for (long i = 0; i < totalSteps; i++) {
    if (dX < 0 && digitalRead(LIMIT_X_PIN) == LOW) {
      mag.off();
      currentXSteps = 0;
      currentYSteps += doneY;
      homed = false;
      return "LIMIT_X";
    }
    if (dY < 0 && digitalRead(LIMIT_Y_PIN) == LOW) {
      mag.off();
      currentYSteps = 0;
      currentXSteps += doneX;
      homed = false;
      return "LIMIT_Y";
    }

    if (i < accelSteps) {
      currentSpeed = sqrt((v0 * v0) + (2.0 * accel * i));
      if (currentSpeed > vMax) currentSpeed = vMax;
    } else if (i >= decelStart) {
      long remaining = totalSteps - i;
      currentSpeed = sqrt((v0 * v0) + (2.0 * accel * remaining));
      if (currentSpeed < v0) currentSpeed = v0;
    } else {
      currentSpeed = vMax;
    }

    int halfDelayUs = lround((1000000.0 / currentSpeed) / 2.0);
    if (halfDelayUs < 15) halfDelayUs = 15;  // ~30000 צעדים/ש' = ~750 מ"מ/ש', רק כדי שלא ייתקע

    countA += stepsA;
    countB += stepsB;
    bool pulseA = false, pulseB = false;
    if (countA >= totalSteps) { digitalWrite(MOTOR_A_STEP, HIGH); pulseA = true; countA -= totalSteps; }
    if (countB >= totalSteps) { digitalWrite(MOTOR_B_STEP, HIGH); pulseB = true; countB -= totalSteps; }
    delayMicroseconds(halfDelayUs);
    if (pulseA) digitalWrite(MOTOR_A_STEP, LOW);
    if (pulseB) digitalWrite(MOTOR_B_STEP, LOW);
    delayMicroseconds(halfDelayUs);

    doneX = dX * (i + 1) / totalSteps;
    doneY = dY * (i + 1) / totalSteps;
  }

  currentXSteps += dX;
  currentYSteps += dY;
  return nullptr;
}

const char* moveToMm(float x, float y) {
  if (!homed) return "NOT_HOMED";
  if (x < 0 || x > MAX_X_MM || y < 0 || y > MAX_Y_MM) return "BOUNDS";
  long tx = lround(x * STEPS_PER_MM);
  long ty = lround(y * STEPS_PER_MM);
  uint32_t t0 = millis();
  const char* err = moveTrapezoidal(tx - currentXSteps, ty - currentYSteps);
  Serial.printf("# pos %.1f %.1f (%lu ms)\n", currentXSteps / STEPS_PER_MM,
                currentYSteps / STEPS_PER_MM, (unsigned long)(millis() - t0));
  return err;
}

// ---------------------------------------------------------------------------
// מצלמה
// ---------------------------------------------------------------------------
bool initCamera() {
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer   = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM; c.pin_d1 = Y3_GPIO_NUM; c.pin_d2 = Y4_GPIO_NUM; c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM; c.pin_d5 = Y7_GPIO_NUM; c.pin_d6 = Y8_GPIO_NUM; c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM;
  c.pin_pclk = PCLK_GPIO_NUM;
  c.pin_vsync = VSYNC_GPIO_NUM;
  c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM;
  c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = -1;
  c.pin_reset = -1;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_JPEG;
  c.frame_size   = FRAMESIZE_XGA;  // 1024x768, כל שדה הראייה (4:3)
  c.jpeg_quality = 12;
  c.fb_count     = 2;
  c.fb_location  = CAMERA_FB_IN_PSRAM;
  c.grab_mode    = CAMERA_GRAB_LATEST;

  if (esp_camera_init(&c) != ESP_OK) {
    esp_camera_deinit();
    delay(200);
    if (esp_camera_init(&c) != ESP_OK) return false;
  }
  return true;
}

// שולח "IMG <n>" ואחריו n בתים. מחזיר nullptr בהצלחה.
const char* captureToSerial() {
  if (!camReady) return "CAM";
  if (mag.isOn()) Serial.println("! CAPTURE_WITH_MAG_ON");
  // זורקים פריים אחד, כדי שהתמונה תצולם אחרי הפקודה ולא לפניה (למשל באמצע תנועה)
  camera_fb_t* fb = esp_camera_fb_get();
  if (fb) esp_camera_fb_return(fb);
  fb = esp_camera_fb_get();
  if (!fb) return "CAM";
  Serial.printf("# frame %ux%u\n", (unsigned)fb->width, (unsigned)fb->height);
  Serial.printf("IMG %u\n", (unsigned)fb->len);
  const uint8_t* p = fb->buf;
  size_t left = fb->len;
  while (left > 0) {
    size_t n = Serial.write(p, left > 4096 ? 4096 : left);
    p += n;
    left -= n;
  }
  Serial.flush();
  esp_camera_fb_return(fb);
  Serial.println();  // סוף שורה אחרי הבינארי, כך ש-DONE תמיד בשורה משלה
  return nullptr;
}

// ---------------------------------------------------------------------------
// פרוטוקול
// ---------------------------------------------------------------------------
void reply(const char* err) {
  if (err) {
    Serial.print("ERR ");
    Serial.println(err);
  } else {
    Serial.println("DONE");
  }
}

bool parseTwo(const String& cmd, float& a, float& b) {
  int s1 = cmd.indexOf(' ');
  if (s1 < 0) return false;
  String rest = cmd.substring(s1 + 1);
  rest.trim();
  int s2 = rest.indexOf(' ');
  if (s2 < 0) return false;
  a = rest.substring(0, s2).toFloat();
  b = rest.substring(s2 + 1).toFloat();
  return true;
}

void execute(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;
  char type = toupper(cmd.charAt(0));
  float a, b;

  switch (type) {
    case 'H':
      reply(runHoming());
      break;
    case 'G':
      if (!parseTwo(cmd, a, b)) { reply("ARGS"); break; }
      reply(moveToMm(a, b));
      break;
    case 'R':
      if (!parseTwo(cmd, a, b)) { reply("ARGS"); break; }
      reply(moveToMm(currentXSteps / STEPS_PER_MM + a, currentYSteps / STEPS_PER_MM + b));
      break;
    case 'M':
      if (cmd.substring(1).toInt() != 0) {
        bool ok = mag.on();
        if (ok) delay(GRAB_MS);
        Serial.printf("# magnet heat %d%%\n", mag.heatPercent());
        reply(ok ? nullptr : "MAG_HOT");
      } else {
        mag.off();
        delay(RELEASE_MS);
        reply(nullptr);
      }
      break;
    case 'C':
      reply(captureToSerial());
      break;
    case 'P':
    case 'F':
      if (!parseTwo(cmd, a, b) || a < 20 || a > 600 || b < 50 || b > 5000) { reply("ARGS"); break; }
      if (type == 'P') { dragMaxSteps = a * STEPS_PER_MM; dragAccelSteps = b * STEPS_PER_MM; }
      else             { freeMaxSteps = a * STEPS_PER_MM; freeAccelSteps = b * STEPS_PER_MM; }
      reply(nullptr);
      break;
    case '?':
      Serial.printf("POS %.1f %.1f HOMED %d MAG %d HEAT %d CAM %d DRAG_V %d DRAG_A %d FREE_V %d FREE_A %d\n",
                    currentXSteps / STEPS_PER_MM, currentYSteps / STEPS_PER_MM,
                    homed, mag.isOn(), mag.heatPercent(), camReady,
                    (int)lround(dragMaxSteps / STEPS_PER_MM), (int)lround(dragAccelSteps / STEPS_PER_MM),
                    (int)lround(freeMaxSteps / STEPS_PER_MM), (int)lround(freeAccelSteps / STEPS_PER_MM));
      reply(nullptr);
      break;
    default:
      reply("UNKNOWN");
  }
}

void setup() {
  // ראשון: מתח לוגיקה לדרייברים לפני STEP/DIR/EN, כדי שלא ידלוף דרך דיודות ההגנה
  pinMode(VIO_PIN, OUTPUT);
  gpio_set_drive_capability((gpio_num_t)VIO_PIN, GPIO_DRIVE_CAP_3);
  digitalWrite(VIO_PIN, HIGH);
  delay(5);

  mag.begin();  // ראשון: הא"מ כבוי לפני כל דבר אחר
  Serial.setRxBufferSize(1024);
  Serial.begin(921600);
  delay(1500);
  boardCheck();

  pinMode(MOTOR_A_STEP, OUTPUT);
  pinMode(MOTOR_A_DIR, OUTPUT);
  pinMode(MOTOR_B_STEP, OUTPUT);
  pinMode(MOTOR_B_DIR, OUTPUT);
  pinMode(EN_PIN, OUTPUT);
  pinMode(LIMIT_X_PIN, INPUT_PULLUP);
  pinMode(LIMIT_Y_PIN, INPUT_PULLUP);
  digitalWrite(EN_PIN, LOW);

  camReady = initCamera();
  Serial.println(camReady ? "# camera OK (XGA)" : "! CAM_INIT_FAILED");
  Serial.println("READY");
}

void loop() {
  if (mag.tripped()) Serial.println("! MAG_TIMEOUT");
  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    execute(line);
  }
}

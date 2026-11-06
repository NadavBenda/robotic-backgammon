#pragma once
// Magnet: שליטה בטוחה באלקטרומגנט (AOD4184 על GPIO 47).
//
// שלוש שכבות הגנה:
// 1. כבוי ב-boot: begin() מוריד את הפין ל-LOW לפני כל דבר אחר.
//    בזמן reset הפין צף, ומודול האופטו לא מקבל זרם, כך שהא"מ כבוי גם אז.
// 2. זמן הפעלה מקסימלי: כל on() מפעיל טיימר חומרה (esp_timer) שרץ במקביל.
//    אם לא קראו ל-off() עד maxOnMs, הטיימר מכבה לבד. זה עובד גם כשה-loop
//    תקוע בתנועה ארוכה (moveTrapezoidal חוסם).
// 3. תקציב חום: "דלי דולף". זמן דלוק ממלא את הדלי, וזמן כבוי מרוקן אותו
//    לפי coolRatio. כשהדלי מלא, on() מסרב עד שהא"מ מתקרר.
//
// שימוש:
//   Magnet mag(MAG_PIN);
//   setup(): mag.begin();
//   loop():  mag.update();       // מדפיס הודעה כשהטיימר כיבה
//   mag.on(); ... mag.off();     // on() מחזיר false אם הא"מ חם מדי

#include <Arduino.h>
#include "esp_timer.h"

class Magnet {
 public:
  // maxOnMs: הכי הרבה זמן ברצף. heatBudgetMs: כמה זמן דלוק מצטבר עד נעילה.
  // coolRatio: כמה מ"ש כבוי מנקים 1 מ"ש דלוק (2 = צריך פי 2 זמן קירור).
  Magnet(int pin, uint32_t maxOnMs = 10000, uint32_t heatBudgetMs = 60000,
         float coolRatio = 2.0)
      : pin_(pin), maxOnMs_(maxOnMs), heatBudgetMs_(heatBudgetMs),
        coolRatio_(coolRatio) {}

  void begin() {
    digitalWrite(pin_, LOW);
    pinMode(pin_, OUTPUT);
    digitalWrite(pin_, LOW);
    esp_timer_create_args_t args = {};
    args.callback = &Magnet::timeoutCb;
    args.arg = this;
    args.name = "mag_guard";
    esp_timer_create(&args, &timer_);
    lastChangeMs_ = millis();
  }

  // מדליק. מחזיר false אם הא"מ בנעילת חום.
  bool on() {
    uint32_t allowed = 0;
    portENTER_CRITICAL(&lock_);
    settleHeat();
    bool locked = heatMs_ >= heatBudgetMs_;
    bool start = !locked && !isOn_;
    if (start) {
      // לא חורגים מתקציב החום גם בהפעלה הזאת
      allowed = min<uint32_t>(maxOnMs_, heatBudgetMs_ - (uint32_t)heatMs_);
      digitalWrite(pin_, HIGH);
      isOn_ = true;
    }
    portEXIT_CRITICAL(&lock_);
    if (locked) return false;
    if (start) {
      esp_timer_stop(timer_);
      esp_timer_start_once(timer_, (uint64_t)allowed * 1000ULL);
    }
    return true;
  }

  void off() {
    esp_timer_stop(timer_);
    forceOff();
  }

  bool isOn() const { return isOn_; }

  // כמה אחוז מתקציב החום נוצל (0–100)
  int heatPercent() {
    portENTER_CRITICAL(&lock_);
    settleHeat();
    float h = heatMs_;
    portEXIT_CRITICAL(&lock_);
    return (int)(100.0f * h / heatBudgetMs_);
  }

  // true פעם אחת אחרי שהטיימר כיבה לבד
  bool tripped() {
    bool t = tripped_;
    tripped_ = false;
    return t;
  }

  // לקרוא מ-loop(): מדווח בסריאל אם הטיימר כיבה
  void update() {
    if (tripped()) {
      Serial.println("ERR MAG_TIMEOUT magnet forced off");
    }
  }

 private:
  static void timeoutCb(void* arg) {
    Magnet* self = static_cast<Magnet*>(arg);
    self->forceOff();
    self->tripped_ = true;
  }

  // נקרא גם מהטיימר (משימה אחרת), ולכן בתוך נעילה
  void forceOff() {
    portENTER_CRITICAL(&lock_);
    digitalWrite(pin_, LOW);
    if (isOn_) {
      settleHeat();
      isOn_ = false;
    }
    portEXIT_CRITICAL(&lock_);
  }

  // מעדכן את הדלי (לקרוא רק בתוך lock_) לפי הזמן שעבר מאז השינוי האחרון
  void settleHeat() {
    uint32_t now = millis();
    uint32_t dt = now - lastChangeMs_;
    lastChangeMs_ = now;
    if (isOn_) {
      heatMs_ += dt;
    } else {
      heatMs_ -= dt / coolRatio_;
      if (heatMs_ < 0) heatMs_ = 0;
    }
  }

  const int pin_;
  const uint32_t maxOnMs_;
  const uint32_t heatBudgetMs_;
  const float coolRatio_;
  esp_timer_handle_t timer_ = nullptr;
  portMUX_TYPE lock_ = portMUX_INITIALIZER_UNLOCKED;
  volatile bool isOn_ = false;
  volatile bool tripped_ = false;
  float heatMs_ = 0;
  uint32_t lastChangeMs_ = 0;
};

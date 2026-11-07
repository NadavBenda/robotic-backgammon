#pragma once
// BoardCheck.h: בדיקת הגדרות Tools של Arduino IDE ללוח Freenove ESP32-S3 WROOM CAM (N16R8).
// מה שאפשר לבדוק בקומפילציה עוצר עם #error. מה שאפשר רק בזמן ריצה נבדק ב-boardCheck().
// להעתיק לכל תיקיית סקיצה, להוסיף #include "BoardCheck.h", ולקרוא ל-boardCheck() אחרי Serial.begin().
//
// ההגדרות הנכונות:
//   Board: ESP32S3 Dev Module        PSRAM: OPI PSRAM
//   Flash Mode: QIO 80MHz            Flash Size: 16MB (128Mb)
//   Partition: 16M Flash (3MB APP/9.9MB FATFS)
//   CPU: 240MHz (WiFi)               USB Mode: Hardware CDC and JTAG
//   USB CDC On Boot: Enabled בשקע USB, Disabled בשקע UART

#include <Arduino.h>
#include "sdkconfig.h"

#ifndef ARDUINO_ESP32S3_DEV
#error "Tools > Board: בחר ESP32S3 Dev Module"
#endif

#ifndef BOARD_HAS_PSRAM
#error "Tools > PSRAM: בחר OPI PSRAM (כרגע כבוי)"
#elif !defined(CONFIG_SPIRAM_MODE_OCT)
#error "Tools > PSRAM: בחר OPI PSRAM (כרגע QSPI)"
#endif

#ifndef CONFIG_FLASHMODE_QIO
#error "Tools > Flash Mode: בחר QIO 80MHz"
#endif

#if F_CPU != 240000000L
#error "Tools > CPU Frequency: בחר 240MHz (WiFi)"
#endif

#ifndef ARDUINO_PARTITION_app3M_fat9M_16MB
#error "Tools > Partition Scheme: בחר 16M Flash (3MB APP/9.9MB FATFS)"
#endif

#if ARDUINO_USB_MODE != 1
#error "Tools > USB Mode: בחר Hardware CDC and JTAG"
#endif

// בדיקות זמן ריצה. מחזיר false אם משהו לא תקין.
inline bool boardCheck() {
  bool ok = true;
  uint32_t flash = ESP.getFlashChipSize();
  if (flash != 16UL * 1024 * 1024) {
    Serial.printf("[BOARD] ERROR Flash Size = %u MB. Tools > Flash Size: 16MB\n", flash >> 20);
    ok = false;
  }
  if (!psramFound()) {
    Serial.println("[BOARD] ERROR PSRAM not found (check Tools > PSRAM = OPI PSRAM)");
    ok = false;
  } else if (ESP.getPsramSize() < 7UL * 1024 * 1024) {
    Serial.printf("[BOARD] ERROR PSRAM only %u KB, expected 8 MB\n", ESP.getPsramSize() >> 10);
    ok = false;
  }
  Serial.printf("[BOARD] %s: flash %u MB, PSRAM %u KB, CPU %u MHz, CDC on boot %d\n",
                ok ? "OK" : "PROBLEM", flash >> 20, ESP.getPsramSize() >> 10,
                getCpuFrequencyMhz(), ARDUINO_USB_CDC_ON_BOOT);
  return ok;
}

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

#define PIN_LED_R 12
#define PIN_LED_G 13
#define PIN_LED_B 14

#define SAMPLE_INTERVAL_US 5000UL  // 200 Hz → 5 ms

Adafruit_MPU6050 mpu;

static inline void setLED(bool r, bool g, bool b) {
    digitalWrite(PIN_LED_R, r ? HIGH : LOW);
    digitalWrite(PIN_LED_G, g ? HIGH : LOW);
    digitalWrite(PIN_LED_B, b ? HIGH : LOW);
}

void setup() {
    Serial.begin(115200);

    pinMode(PIN_LED_R, OUTPUT);
    pinMode(PIN_LED_G, OUTPUT);
    pinMode(PIN_LED_B, OUTPUT);
    setLED(false, false, false);

    Wire.begin(21, 22);  // SDA=21, SCL=22

    if (!mpu.begin(MPU6050_I2CADDR_DEFAULT, &Wire)) {
        setLED(true, false, false);  // red: I2C failure
        while (1) { delay(10); }
    }

    // 94 Hz LPF sits just under the 100 Hz Nyquist for 200 Hz sampling
    mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
    mpu.setFilterBandwidth(MPU6050_BAND_94_HZ);

    setLED(false, true, false);  // green: ready
}

void loop() {
    static unsigned long lastUs = 0;
    const unsigned long now = micros();

    if (now - lastUs < SAMPLE_INTERVAL_US) return;
    lastUs += SAMPLE_INTERVAL_US;  // increment keeps cadence drift-free

    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    // timestamp_ms,accel_z_m_per_s2
    Serial.print(millis());
    Serial.print(',');
    Serial.println(a.acceleration.z, 4);
}

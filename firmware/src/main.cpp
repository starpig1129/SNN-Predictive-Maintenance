#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <LittleFS.h>

#define PIN_LED_R 12
#define PIN_LED_G 13
#define PIN_LED_B 14

#define SAMPLE_INTERVAL_US 5000UL  // 200 Hz → 5 ms

Adafruit_MPU6050 mpu;

#ifndef INFERENCE_MODE
static File     g_dataFile;
static uint32_t g_sampleCount = 0;
#endif

static inline void setLED(bool r, bool g, bool b) {
    digitalWrite(PIN_LED_R, r ? HIGH : LOW);
    digitalWrite(PIN_LED_G, g ? HIGH : LOW);
    digitalWrite(PIN_LED_B, b ? HIGH : LOW);
}

// ── I2C bus recovery — 9 SCL pulses to release a locked slave ─────────────
static void i2c_recover() {
    pinMode(22, OUTPUT);  // SCL
    pinMode(21, OUTPUT);  // SDA
    digitalWrite(21, HIGH);
    for (int i = 0; i < 9; i++) {
        digitalWrite(22, HIGH); delayMicroseconds(5);
        digitalWrite(22, LOW);  delayMicroseconds(5);
    }
    digitalWrite(21, LOW);                 // STOP: SDA low → high while SCL high
    digitalWrite(22, HIGH); delayMicroseconds(5);
    digitalWrite(21, HIGH); delayMicroseconds(5);
    Wire.begin(21, 22);
    Wire.setClock(100000);
}

// ── Shared hardware init ───────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);

    pinMode(PIN_LED_R, OUTPUT);
    pinMode(PIN_LED_G, OUTPUT);
    pinMode(PIN_LED_B, OUTPUT);
    setLED(false, false, false);

    Wire.begin(21, 22);   // SDA=21, SCL=22
    Wire.setClock(100000); // 100 kHz — more tolerant to motor EMI than default 400 kHz
    Wire.setTimeOut(15);   // 15 ms hard cap per transaction — prevents long I2C hangs under EMI

    int retries = 0;
    while (!mpu.begin(MPU6050_I2CADDR_DEFAULT, &Wire)) {
        setLED(true, false, false);
        delay(200);
        setLED(false, false, false);
        delay(200);
        if (++retries % 5 == 0) i2c_recover();
    }

    // 94 Hz LPF sits just under the 100 Hz Nyquist for 200 Hz sampling
    mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
    mpu.setFilterBandwidth(MPU6050_BAND_94_HZ);

    setLED(false, true, false);  // green: ready

#ifndef INFERENCE_MODE
    if (!LittleFS.begin(true)) {
        setLED(true, false, true);  // magenta = flash init failed
    } else {
        // Only create a fresh file when none exists — preserves data across
        // accidental resets caused by motor EMI.  Send 'c' over serial to
        // explicitly start a new recording session.
        if (!LittleFS.exists("/data.csv")) {
            File hdr = LittleFS.open("/data.csv", FILE_WRITE);
            if (hdr) { hdr.println("timestamp_ms,accel_z"); hdr.close(); }
        }
        g_dataFile = LittleFS.open("/data.csv", FILE_APPEND);
    }
#endif
}

// ═══════════════════════════════════════════════════════════════════════════════
//  DATA COLLECTION MODE  (default — no -DINFERENCE_MODE build flag)
//
//  Samples are written to LittleFS /data.csv so the motor can run without an
//  active USB connection.  When the motor is off and USB is stable, send the
//  single character 'd' over serial to dump the file back to the PC.
//
//  LED during recording:
//    Green steady   — idle, ready
//    Green + blue   — actively logging (blue toggles every ~1 s)
// ═══════════════════════════════════════════════════════════════════════════════
#ifndef INFERENCE_MODE

void loop() {
    // ── Serial commands ───────────────────────────────────────────────────────
    if (Serial.available()) {
        const char c = (char)Serial.read();
        if (c == 'd') {
            // Dump flash to serial
            if (g_dataFile) { g_dataFile.flush(); g_dataFile.close(); }
            File f = LittleFS.open("/data.csv", FILE_READ);
            if (f) {
                Serial.println("DATA_START");
                while (f.available()) Serial.write(f.read());
                Serial.println("DATA_END");
                f.close();
            } else {
                Serial.println("ERROR:no_file");
            }
            g_dataFile = LittleFS.open("/data.csv", FILE_APPEND);
            return;
        }
        if (c == 'c') {
            // Clear flash and start a new recording session
            if (g_dataFile) { g_dataFile.close(); }
            LittleFS.remove("/data.csv");
            File hdr = LittleFS.open("/data.csv", FILE_WRITE);
            if (hdr) { hdr.println("timestamp_ms,accel_z"); hdr.close(); }
            g_dataFile   = LittleFS.open("/data.csv", FILE_APPEND);
            g_sampleCount = 0;
            Serial.println("OK:cleared");
            return;
        }
    }

    // ── 200 Hz cadence ────────────────────────────────────────────────────────
    static unsigned long lastUs    = 0;
    static uint8_t       i2c_fails = 0;
    const unsigned long now = micros();
    if (now - lastUs < SAMPLE_INTERVAL_US) return;
    lastUs += SAMPLE_INTERVAL_US;

    sensors_event_t a, g, temp;
    if (!mpu.getEvent(&a, &g, &temp)) {
        // Consecutive I2C failures → bus may be locked; attempt recovery
        if (++i2c_fails >= 10) {
            i2c_recover();
            mpu.begin(MPU6050_I2CADDR_DEFAULT, &Wire);
            mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
            mpu.setFilterBandwidth(MPU6050_BAND_94_HZ);
            i2c_fails = 0;
        }
        return;
    }
    i2c_fails = 0;

    const float az = a.acceleration.x;

    // Discard only extreme I2C glitches (motor vibration can reach ~30–40 m/s²)
    static float prev_az  = 0.0f;
    static bool  has_prev = false;
    if (has_prev && fabsf(az - prev_az) > 50.0f) return;
    prev_az  = az;
    has_prev = true;

    // ── Write to flash ────────────────────────────────────────────────────────
    if (g_dataFile) {
        g_dataFile.print(millis());
        g_dataFile.print(',');
        g_dataFile.println(az, 4);
        g_sampleCount++;
        if (g_sampleCount % 200 == 0) {
            g_dataFile.flush();
            setLED(false, true, (g_sampleCount / 200) & 1u);  // blue blinks every ~1 s
        }
    }

    // Also echo over serial when USB is stable (e.g. while collecting idle.csv)
    Serial.print(millis());
    Serial.print(',');
    Serial.println(az, 4);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  INFERENCE MODE  (-DINFERENCE_MODE in platformio.ini [env:nodemcu-32s-infer])
//
//  Streaming SNN: one delta-encoded spike per sample, mem1 never reset.
//  Requires firmware/include/snn_weights.h from training/export_weights.py
//
//  Serial output (two line types, interleaved):
//    <ts_ms>,<accel_z>                                ← raw waveform (every sample)
//    INF,<hidden_rate>,<fault_acc>,<status>,<fault_acc>  ← every SNN_DECISION_INTERVAL samples
//
//  LED status:
//    Green steady    — normal
//    Green + Blue    — normal; blue blinks proportional to hidden spike rate
//    Red rapid flash — fault detected
// ═══════════════════════════════════════════════════════════════════════════════
#else

#include "snn_weights.h"

// ── Persistent streaming state ────────────────────────────────────────────
static float    mem1[SNN_HIDDEN] = {};   // never reset between samples
static float    out_acc[2]       = {};   // [fault, normal] logit accumulators
static uint32_t spk_count        = 0;   // hidden spikes this interval
static int      dec_count        = 0;   // samples since last decision
static float    prev_az          = 0.0f;
static bool     has_prev         = false;

// ── Inference outputs ─────────────────────────────────────────────────────
static float   g_hidden_rate  = 0.0f;
static float   g_anomaly_spk  = 0.0f;
static float   g_mem2_anomaly = 0.0f;
static uint8_t g_status       = 0;      // 0 = normal, 1 = fault

// ── LED controller ────────────────────────────────────────────────────────
static void update_leds(unsigned long now_ms) {
    static unsigned long led_toggle_ms = 0;
    static bool          blue_state    = false;

    if (g_status == 1) {
        bool flash = ((now_ms / 50UL) & 1UL) != 0UL;
        setLED(flash, false, false);
    } else {
        unsigned long blue_interval_ms =
            (g_hidden_rate > 0.01f)
            ? (unsigned long)(80.0f / g_hidden_rate)
            : 500UL;
        blue_interval_ms = constrain(blue_interval_ms, 20UL, 500UL);
        if (now_ms - led_toggle_ms >= blue_interval_ms) {
            led_toggle_ms = now_ms;
            blue_state    = !blue_state;
        }
        setLED(false, true, blue_state);
    }
}

// ── Main loop ─────────────────────────────────────────────────────────────
void loop() {
    const unsigned long now_ms = millis();
    update_leds(now_ms);

    static unsigned long lastUs    = 0;
    static uint8_t       i2c_fails = 0;
    const unsigned long  nowUs  = micros();
    if (nowUs - lastUs < SAMPLE_INTERVAL_US) return;
    lastUs += SAMPLE_INTERVAL_US;

    sensors_event_t a, g, temp;
    if (!mpu.getEvent(&a, &g, &temp)) {
        // Skip spike computation on I2C failure — do not update prev_az,
        // otherwise the next good reading would produce a spurious large delta.
        if (++i2c_fails >= 10) {
            i2c_recover();
            mpu.begin(MPU6050_I2CADDR_DEFAULT, &Wire);
            mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
            mpu.setFilterBandwidth(MPU6050_BAND_94_HZ);
            i2c_fails = 0;
        }
        return;
    }
    i2c_fails = 0;
    const float az = a.acceleration.x;

    Serial.print(millis());
    Serial.print(',');
    Serial.println(az, 4);

    // ── Delta encode: 1 spike if |Δaz| exceeds threshold ──────────────────
    const float spike = (has_prev && fabsf(az - prev_az) > SNN_DELTA_THRESH) ? 1.0f : 0.0f;
    prev_az  = az;
    has_prev = true;

    // ── FC1 + LIF1: persistent membrane, never reset ──────────────────────
    // SNN_W1 is [SNN_HIDDEN]: cur1[j] = spike * W1[j]
    for (int j = 0; j < SNN_HIDDEN; j++) {
        mem1[j] = SNN_BETA1 * mem1[j] + spike * SNN_W1[j];
        if (mem1[j] >= SNN_THRESH1) {
            mem1[j]    -= SNN_THRESH1;
            out_acc[0] += SNN_W2[0 * SNN_HIDDEN + j];  // fault logit
            out_acc[1] += SNN_W2[1 * SNN_HIDDEN + j];  // normal logit
            spk_count++;
        }
    }

    // ── Decision every SNN_DECISION_INTERVAL samples ──────────────────────
    if (++dec_count >= SNN_DECISION_INTERVAL) {
        g_hidden_rate  = (float)spk_count / (float)(SNN_DECISION_INTERVAL * SNN_HIDDEN);
        if (g_hidden_rate < SNN_IDLE_RATE_THRESH) {
            g_status = 1u;  // motor stopped = anomaly
        } else {
            g_status = (out_acc[1] <= out_acc[0]) ? 1u : 0u;
        }
        g_anomaly_spk  = out_acc[0];
        g_mem2_anomaly = out_acc[0];

        Serial.print("INF,");
        Serial.print(g_hidden_rate,  2);
        Serial.print(',');
        Serial.print(g_anomaly_spk,  1);
        Serial.print(',');
        Serial.print(g_status);
        Serial.print(',');
        Serial.println(g_mem2_anomaly, 4);

        out_acc[0] = out_acc[1] = 0.0f;
        spk_count  = 0;
        dec_count  = 0;
        // Reset membrane to match training (each chunk starts from mem1=0)
        memset(mem1, 0, sizeof(mem1));
    }
}

#endif  // INFERENCE_MODE

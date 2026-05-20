#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

#define PIN_LED_R 25
#define PIN_LED_G 26
#define PIN_LED_B 27

#define SAMPLE_INTERVAL_US 5000UL  // 200 Hz → 5 ms

Adafruit_MPU6050 mpu;

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
}

// ═══════════════════════════════════════════════════════════════════════════════
//  DATA COLLECTION MODE  (default — no -DINFERENCE_MODE build flag)
//  Serial output: "timestamp_ms,accel_z\n"
// ═══════════════════════════════════════════════════════════════════════════════
#ifndef INFERENCE_MODE

void loop() {
    static unsigned long lastUs = 0;
    const unsigned long now = micros();

    if (now - lastUs < SAMPLE_INTERVAL_US) return;
    lastUs += SAMPLE_INTERVAL_US;  // drift-free cadence

    sensors_event_t a, g, temp;
    if (!mpu.getEvent(&a, &g, &temp)) return;

    const float az = a.acceleration.x;

    // Reject readings that imply physically impossible jerk at 200 Hz.
    // A 130-motor can't change accel_x by >15 m/s² in 5 ms; larger jumps are I2C noise.
    static float prev_az  = 0.0f;
    static bool  has_prev = false;
    if (has_prev && fabsf(az - prev_az) > 15.0f) return;
    prev_az  = az;
    has_prev = true;

    Serial.print(millis());
    Serial.print(',');
    Serial.println(az, 4);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  INFERENCE MODE  (-DINFERENCE_MODE in platformio.ini [env:nodemcu-32s-infer])
//
//  Requires firmware/include/snn_weights.h produced by training/export_weights.py
//
//  Serial output (two line types, interleaved):
//    <ts_ms>,<accel_z>                           ← raw waveform (every sample)
//    INF,<hidden_rate>,<anomaly_spk>,<status>,<mem2>  ← inference (every window)
//
//  LED status:
//    Green steady    — normal, low vibration
//    Green + Blue    — normal; blue blinks in sync with hidden-layer spike rate
//    Red rapid flash — anomaly detected (output spike rate > 50 % of T_STEPS)
// ═══════════════════════════════════════════════════════════════════════════════
#else

#include "snn_weights.h"

// 50 % of output timesteps firing on anomaly neuron triggers alert
#define ANOMALY_THRESHOLD  (SNN_T_STEPS * 0.5f)

// ── Ring buffer ───────────────────────────────────────────────────────────
static float   sample_buf[SNN_WINDOW];
static int     buf_idx   = 0;

// ── LIF membrane potentials (reset each window) ───────────────────────────
static float   mem1[SNN_HIDDEN];
static float   mem2[2];

// ── Inference outputs (updated after each window) ─────────────────────────
static float   g_hidden_rate   = 0.0f;  // avg hidden spikes per timestep
static float   g_anomaly_spk   = 0.0f;  // anomaly output spike count
static float   g_mem2_anomaly  = 0.0f;  // output neuron 1 membrane potential
static uint8_t g_status        = 0;     // 0 = normal, 1 = anomaly

// ── Delta modulation ──────────────────────────────────────────────────────
static void delta_encode(const float *sig, uint8_t *spk, int n) {
    spk[0] = 0;
    for (int i = 1; i < n; i++) {
        spk[i] = (fabsf(sig[i] - sig[i - 1]) > SNN_DELTA_THRESH) ? 1u : 0u;
    }
}

// ── SNN forward pass ──────────────────────────────────────────────────────
// Binary spike inputs → FC1 is pure accumulation (no multiply).
// FC2 inputs (spk1) are also binary → same trick applies there.
static void run_inference(const uint8_t *spk_in) {
    static float cur1[SNN_HIDDEN];
    static float cur2[2];
    static float spk1[SNN_HIDDEN];

    float anomaly_sum = 0.0f;
    float hidden_sum  = 0.0f;

    memset(mem1, 0, sizeof(mem1));
    memset(mem2, 0, sizeof(mem2));

    for (int t = 0; t < SNN_T_STEPS; t++) {

        // FC1: for each active input spike, accumulate its weight column
        memset(cur1, 0, sizeof(cur1));
        for (int i = 0; i < SNN_WINDOW; i++) {
            if (spk_in[i]) {
                for (int j = 0; j < SNN_HIDDEN; j++) {
                    cur1[j] += SNN_W1[j * SNN_WINDOW + i];
                }
            }
        }

        // LIF1: integrate + fire with soft reset
        for (int j = 0; j < SNN_HIDDEN; j++) {
            mem1[j] = SNN_BETA1 * mem1[j] + cur1[j];
            if (mem1[j] >= SNN_THRESH1) {
                spk1[j]  = 1.0f;
                mem1[j] -= SNN_THRESH1;
                hidden_sum += 1.0f;
            } else {
                spk1[j] = 0.0f;
            }
        }

        // FC2: accumulate hidden spikes
        cur2[0] = 0.0f;  cur2[1] = 0.0f;
        for (int j = 0; j < SNN_HIDDEN; j++) {
            if (spk1[j] > 0.0f) {
                cur2[0] += SNN_W2[0 * SNN_HIDDEN + j];
                cur2[1] += SNN_W2[1 * SNN_HIDDEN + j];
            }
        }

        // LIF2: integrate + fire
        for (int k = 0; k < 2; k++) {
            mem2[k] = SNN_BETA2 * mem2[k] + cur2[k];
            if (mem2[k] >= SNN_THRESH2) {
                mem2[k] -= SNN_THRESH2;
                if (k == 1) anomaly_sum += 1.0f;
            }
        }
    }

    g_hidden_rate  = hidden_sum  / (float)SNN_T_STEPS;
    g_anomaly_spk  = anomaly_sum;
    g_mem2_anomaly = mem2[1];
    g_status       = (anomaly_sum > ANOMALY_THRESHOLD) ? 1u : 0u;

    // Inference result line for the dashboard
    Serial.print("INF,");
    Serial.print(g_hidden_rate,  2);
    Serial.print(',');
    Serial.print(g_anomaly_spk,  1);
    Serial.print(',');
    Serial.print(g_status);
    Serial.print(',');
    Serial.println(g_mem2_anomaly, 4);
}

// ── LED controller ────────────────────────────────────────────────────────
static void update_leds(unsigned long now_ms) {
    static unsigned long led_toggle_ms = 0;
    static bool          blue_state    = false;

    if (g_status == 1) {
        // Anomaly: red rapid flash at ~10 Hz
        bool flash = ((now_ms / 50UL) & 1UL) != 0UL;
        setLED(flash, false, false);

    } else {
        // Normal: green on; blue blinks at rate proportional to hidden activity.
        // A higher hidden_rate → shorter blue toggle interval.
        unsigned long blue_interval_ms =
            (g_hidden_rate > 0.01f)
            ? (unsigned long)(80.0f / g_hidden_rate)  // e.g. rate=1 → 80 ms
            : 500UL;                                   // very slow blink when quiet
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
    update_leds(now_ms);  // runs freely for smooth LED visuals

    static unsigned long lastUs = 0;
    const unsigned long  nowUs  = micros();
    if (nowUs - lastUs < SAMPLE_INTERVAL_US) return;
    lastUs += SAMPLE_INTERVAL_US;

    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    const float az = a.acceleration.x;

    // Raw waveform line — dashboard uses this for the live plot
    Serial.print(millis());
    Serial.print(',');
    Serial.println(az, 4);

    // Accumulate into ring buffer; run inference when full
    sample_buf[buf_idx++] = az;
    if (buf_idx >= SNN_WINDOW) {
        static uint8_t spikes[SNN_WINDOW];
        delta_encode(sample_buf, spikes, SNN_WINDOW);
        run_inference(spikes);
        buf_idx = 0;  // slide window (no overlap in inference mode)
    }
}

#endif  // INFERENCE_MODE

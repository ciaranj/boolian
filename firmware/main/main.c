/*
 * Boolian's microphone and motion sensor: an ESP32-C5 with an INMP441 I2S microphone and an AM312 PIR.
 *
 * Streams 16 kHz mono s16le audio in 20 ms GHST frames (docs/protocol-v1.md) to the server's /mic WebSocket,
 * and sends {"type":"motion","on":...} whenever the PIR changes. The server relays both to the browser.
 * Wi-Fi and the server address live in secrets.h (copy secrets.example.h).
 */

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "driver/gpio.h"
#include "driver/i2s_std.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "esp_websocket_client.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "nvs_flash.h"

#include "secrets.h"

// Wiring. None of these are strapping pins on the C5; the AM312 runs from 3V3.
#define PIN_MIC_BCLK GPIO_NUM_4  // INMP441 SCK
#define PIN_MIC_WS GPIO_NUM_5    // INMP441 WS (its L/R pin to GND: left channel)
#define PIN_MIC_DATA GPIO_NUM_6  // INMP441 SD
#define PIN_PIR GPIO_NUM_3       // AM312 OUT

#define SAMPLE_RATE 16000
#define FRAME_SAMPLES 320  // 20 ms
#define QUEUE_FRAMES 10    // 200 ms: on a network stall, old audio is dropped rather than piling up latency
// The INMP441 is quiet (-26 dBFS at 94 dB SPL), so speech a metre away is around -60 dBFS raw. Raise this if the
// operator panel's meter barely moves when someone talks; lower it if the serial log reports clipping.
#define MIC_GAIN 16.0f

#define HEADER_BYTES 16

static const char *TAG = "spectre-mic";

typedef struct {
    uint32_t seq;
    uint32_t capture_ms;
    int16_t pcm[FRAME_SAMPLES];
} frame_t;

static QueueHandle_t frames;
static esp_websocket_client_handle_t ws;
static volatile bool ws_up;
static volatile bool say_hello;  // set on connecting; the main loop sends it, not the WebSocket's own task
static i2s_chan_handle_t rx;

// Diagnostics: reset each time the serial line is printed, apart from the running totals.
static volatile float level_sum;  // sum of squares, normalised
static volatile uint32_t level_n;
static volatile float level_peak;
static volatile uint32_t clips, dropped, sent, reconnects, connects;

static void wifi_event(void *arg, esp_event_base_t base, int32_t id, void *data) {
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGW(TAG, "Wi-Fi lost; reconnecting");
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *e = data;
        ESP_LOGI(TAG, "Wi-Fi up: " IPSTR, IP2STR(&e->ip_info.ip));
    }
}

static void wifi_start(void) {
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_event, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, wifi_event, NULL));
    wifi_config_t wc = {0};
    strlcpy((char *)wc.sta.ssid, WIFI_SSID, sizeof(wc.sta.ssid));
    strlcpy((char *)wc.sta.password, WIFI_PASSWORD, sizeof(wc.sta.password));
    wc.sta.scan_method = WIFI_ALL_CHANNEL_SCAN;  // pick the strongest access point, on either band
    wc.sta.sort_method = WIFI_CONNECT_AP_BY_SIGNAL;
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wc));
    ESP_ERROR_CHECK(esp_wifi_start());
    esp_wifi_set_ps(WIFI_PS_NONE);  // power saving adds tens of ms of jitter to a steady stream
}

static void send_text(const char *json) {
    if (ws_up) esp_websocket_client_send_text(ws, json, strlen(json), pdMS_TO_TICKS(100));
}

static void ws_event(void *arg, esp_event_base_t base, int32_t id, void *data) {
    if (id == WEBSOCKET_EVENT_CONNECTED) {
        if (connects++) reconnects++;
        ESP_LOGI(TAG, "connected to %s", SERVER_URI);
        ws_up = true;
        say_hello = true;
    } else if (id == WEBSOCKET_EVENT_DISCONNECTED || id == WEBSOCKET_EVENT_CLOSED) {
        if (ws_up) ESP_LOGW(TAG, "server connection lost; retrying");
        ws_up = false;
    }
}

static void ws_start(void) {
    esp_websocket_client_config_t cfg = {
        .uri = SERVER_URI,
        .reconnect_timeout_ms = 2000,
        .network_timeout_ms = 5000,
        .buffer_size = 1024,
        .ping_interval_sec = 5,
    };
    ws = esp_websocket_client_init(&cfg);
    esp_websocket_register_events(ws, WEBSOCKET_EVENT_ANY, ws_event, NULL);
    esp_websocket_client_start(ws);
}

static void mic_start(void) {
    i2s_chan_config_t chan = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
    chan.dma_desc_num = 6;
    chan.dma_frame_num = FRAME_SAMPLES;
    ESP_ERROR_CHECK(i2s_new_channel(&chan, NULL, &rx));
    i2s_std_config_t std = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        // The INMP441 sends 24-bit samples, MSB first, in 32-bit slots.
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = PIN_MIC_BCLK,
            .ws = PIN_MIC_WS,
            .dout = I2S_GPIO_UNUSED,
            .din = PIN_MIC_DATA,
        },
    };
    std.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx, &std));
    ESP_ERROR_CHECK(i2s_channel_enable(rx));
}

/** Reads the microphone forever, converting each 20 ms to s16 and queueing it for the sender. */
static void capture_task(void *arg) {
    static int32_t raw[FRAME_SAMPLES];
    static frame_t f;
    uint32_t seq = 0;
    float dc = 0;  // the INMP441 has a small DC offset; this follows it slowly and takes it off
    for (;;) {
        size_t got = 0;
        if (i2s_channel_read(rx, raw, sizeof(raw), &got, portMAX_DELAY) != ESP_OK || got != sizeof(raw)) continue;
        f.seq = seq++;
        f.capture_ms = (uint32_t)(esp_timer_get_time() / 1000);
        float sum = 0, peak = 0;
        for (int i = 0; i < FRAME_SAMPLES; i++) {
            float v = (float)(raw[i] >> 8) / 8388608.0f;  // 24-bit sample to -1..1
            dc += 0.0005f * (v - dc);
            v = (v - dc) * MIC_GAIN;
            if (v > 1.0f || v < -1.0f) {
                clips++;
                v = v > 0 ? 1.0f : -1.0f;
            }
            f.pcm[i] = (int16_t)lrintf(v * 32767.0f);
            sum += v * v;
            if (fabsf(v) > peak) peak = fabsf(v);
        }
        level_sum += sum;
        level_n += FRAME_SAMPLES;
        if (peak > level_peak) level_peak = peak;
        if (!ws_up) continue;
        if (xQueueSend(frames, &f, 0) != pdTRUE) {
            frame_t old;
            xQueueReceive(frames, &old, 0);  // full: drop the oldest, keep the newest
            xQueueSend(frames, &f, 0);
            dropped++;
        }
    }
}

/** Sends queued frames to the server. */
static void send_task(void *arg) {
    static frame_t f;
    static uint8_t buf[HEADER_BYTES + FRAME_SAMPLES * 2];
    for (;;) {
        if (xQueueReceive(frames, &f, portMAX_DELAY) != pdTRUE || !ws_up) continue;
        const uint16_t len = FRAME_SAMPLES * 2;
        memcpy(buf, "GHST", 4);
        buf[4] = 1;  // protocol version
        buf[5] = 0;  // flags
        buf[6] = len >> 8, buf[7] = len & 0xff;
        buf[8] = f.seq >> 24, buf[9] = f.seq >> 16, buf[10] = f.seq >> 8, buf[11] = f.seq;
        buf[12] = f.capture_ms >> 24, buf[13] = f.capture_ms >> 16, buf[14] = f.capture_ms >> 8, buf[15] = f.capture_ms;
        memcpy(buf + HEADER_BYTES, f.pcm, len);  // the C5 is little-endian, as the protocol wants
        if (esp_websocket_client_send_bin(ws, (const char *)buf, sizeof(buf), pdMS_TO_TICKS(200)) < 0) dropped++;
        else sent++;
    }
}

static float to_db(float x) { return 20.0f * log10f(x + 1e-9f); }

void app_main(void) {
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);

    gpio_config_t pir = {.pin_bit_mask = 1ULL << PIN_PIR, .mode = GPIO_MODE_INPUT, .pull_down_en = GPIO_PULLDOWN_ENABLE};
    ESP_ERROR_CHECK(gpio_config(&pir));

    frames = xQueueCreate(QUEUE_FRAMES, sizeof(frame_t));
    mic_start();
    wifi_start();
    ws_start();
    xTaskCreate(capture_task, "capture", 4096, NULL, 10, NULL);
    xTaskCreate(send_task, "send", 4096, NULL, 8, NULL);

    // Watch the PIR, and once a second print the diagnostics (sending them to the server every 5 s).
    int motion = -1;
    int tick = 0;
    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(50));
        int now = gpio_get_level(PIN_PIR);
        if (say_hello) {
            say_hello = false;
            send_text("{\"v\":1,\"type\":\"hello\",\"role\":\"mic\",\"device\":\"" DEVICE_NAME "\",\"sample_rate\":16000}");
            motion = -1;  // tell the new connection whether anyone's there
        }
        if (now != motion) {
            if (motion != -1 || now) ESP_LOGI(TAG, "motion %s", now ? "seen" : "stopped");
            motion = now;
            send_text(now ? "{\"v\":1,\"type\":\"motion\",\"on\":true}" : "{\"v\":1,\"type\":\"motion\",\"on\":false}");
        }
        if (++tick % 20) continue;

        float rms = level_n ? sqrtf(level_sum / level_n) : 0, peak = level_peak;
        level_sum = 0, level_n = 0, level_peak = 0;
        wifi_ap_record_t ap;
        int rssi = esp_wifi_sta_get_ap_info(&ap) == ESP_OK ? ap.rssi : 0;
        uint32_t heap = esp_get_free_heap_size();
        ESP_LOGI(TAG, "level %.1f dBFS peak %.1f | clips %lu sent %lu dropped %lu queued %u | %s rssi %d reconnects %lu "
                      "heap %lu | pir %d",
                 to_db(rms), to_db(peak), clips, sent, dropped, (unsigned)uxQueueMessagesWaiting(frames),
                 ws_up ? "online" : "OFFLINE", rssi, reconnects, heap, motion);
        if (tick % 100 == 0) {
            char json[200];
            snprintf(json, sizeof(json),
                     "{\"v\":1,\"type\":\"stats\",\"rssi\":%d,\"db\":%.1f,\"peak_db\":%.1f,\"clips\":%lu,\"dropped\":%lu,"
                     "\"reconnects\":%lu,\"heap\":%lu,\"uptime_s\":%lu}",
                     rssi, to_db(rms), to_db(peak), clips, dropped, reconnects, heap,
                     (unsigned long)(esp_timer_get_time() / 1000000));
            send_text(json);
        }
    }
}

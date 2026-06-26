#include "captive_portal.h"

#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <netinet/in.h>

#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "nvs_flash.h"
#include "esp_spiffs.h"
#include "esp_https_server.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "portal";

#define SPIFFS_MOUNT "/spiffs"
#define CONFIG_PATH  SPIFFS_MOUNT "/config.json"
#define MAX_SLOTS    5

// Embedded web UI (from main/web/index.html via EMBED_FILES)
extern const uint8_t _binary_index_html_start[];
extern const uint8_t _binary_index_html_end[];

// Embedded TLS certificate and key (from main/server.crt / server.key via EMBED_FILES)
extern const uint8_t _binary_server_crt_start[];
extern const uint8_t _binary_server_crt_end[];
extern const uint8_t _binary_server_key_start[];
extern const uint8_t _binary_server_key_end[];

// =============================================================================
// Helpers
// =============================================================================

static void track_path(int slot, char *out, size_t len) {
    snprintf(out, len, SPIFFS_MOUNT "/track_%d.webm", slot);
}

static void track_type_path(int slot, char *out, size_t len) {
    snprintf(out, len, SPIFFS_MOUNT "/track_%d.type", slot);
}

static bool track_exists(int slot) {
    char path[48];
    track_path(slot, path, sizeof(path));
    struct stat st;
    return stat(path, &st) == 0;
}

// Minimal JSON parser: populate active[MAX_SLOTS] from config.json
static void read_config(bool active[MAX_SLOTS]) {
    memset(active, 0, MAX_SLOTS * sizeof(bool));
    FILE *f = fopen(CONFIG_PATH, "r");
    if (!f) return;
    char buf[128];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = '\0';
    const char *p = strstr(buf, "\"active\":");
    if (!p) return;
    p = strchr(p, '[');
    if (!p) return;
    for (p++; *p && *p != ']'; p++) {
        if (*p >= '0' && *p - '0' < MAX_SLOTS) active[*p - '0'] = true;
    }
}

// =============================================================================
// HTTPS handlers
// =============================================================================

static esp_err_t h_root(httpd_req_t *req) {
    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req,
        (const char *)_binary_index_html_start,
        _binary_index_html_end - _binary_index_html_start);
}

static esp_err_t h_status(httpd_req_t *req) {
    bool active[MAX_SLOTS];
    read_config(active);
    char json[256];
    int n = snprintf(json, sizeof(json), "{\"slots\":[");
    for (int i = 0; i < MAX_SLOTS; i++) {
        n += snprintf(json + n, (int)sizeof(json) - n,
            "{\"id\":%d,\"recorded\":%s,\"active\":%s}%s",
            i,
            track_exists(i) ? "true" : "false",
            active[i]       ? "true" : "false",
            i < MAX_SLOTS - 1 ? "," : "");
    }
    snprintf(json + n, (int)sizeof(json) - n, "]}");
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_sendstr(req, json);
}

static esp_err_t h_upload(httpd_req_t *req) {
    const char *uri = req->uri;           // /upload/N
    int slot = uri[strlen(uri) - 1] - '0';
    if (slot < 0 || slot >= MAX_SLOTS) {
        return httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Invalid slot");
    }

    // Persist Content-Type so playback can serve the correct MIME
    char ct_buf[64] = "audio/webm";
    httpd_req_get_hdr_value_str(req, "Content-Type", ct_buf, sizeof(ct_buf));
    char type_path[48];
    track_type_path(slot, type_path, sizeof(type_path));
    FILE *tf = fopen(type_path, "w");
    if (tf) { fputs(ct_buf, tf); fclose(tf); }

    char path[48];
    track_path(slot, path, sizeof(path));
    FILE *f = fopen(path, "wb");
    if (!f) return httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Cannot create file");

    char buf[2048];
    int remaining = req->content_len;
    bool ok = true;

    if (remaining > 0) {
        while (remaining > 0) {
            int want = (remaining < (int)sizeof(buf)) ? remaining : (int)sizeof(buf);
            int got  = httpd_req_recv(req, buf, want);
            if (got <= 0) { ok = false; break; }
            fwrite(buf, 1, got, f);
            remaining -= got;
        }
    } else {
        // Chunked transfer: read until closed
        int got;
        while ((got = httpd_req_recv(req, buf, sizeof(buf))) > 0) {
            fwrite(buf, 1, got, f);
        }
        if (got < 0) ok = false;
    }
    fclose(f);

    if (!ok) {
        remove(path); remove(type_path);
        return httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Receive error");
    }
    ESP_LOGI(TAG, "Saved slot %d → %s (%s)", slot, path, ct_buf);
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_sendstr(req, "{\"ok\":true}");
}

static esp_err_t h_play(httpd_req_t *req) {
    const char *uri = req->uri;           // /play/N
    int slot = uri[strlen(uri) - 1] - '0';
    if (slot < 0 || slot >= MAX_SLOTS) {
        return httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Invalid slot");
    }

    char path[48];
    track_path(slot, path, sizeof(path));
    FILE *f = fopen(path, "rb");
    if (!f) return httpd_resp_send_err(req, HTTPD_404_NOT_FOUND, "Track not found");

    // Serve with the MIME type that was used when recording
    char mime[64] = "audio/webm";
    char type_path[48];
    track_type_path(slot, type_path, sizeof(type_path));
    FILE *tf = fopen(type_path, "r");
    if (tf) { fgets(mime, sizeof(mime), tf); fclose(tf); }

    httpd_resp_set_type(req, mime);
    httpd_resp_set_hdr(req, "Accept-Ranges", "none");

    char buf[2048];
    size_t n;
    esp_err_t err = ESP_OK;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        if ((err = httpd_resp_send_chunk(req, buf, (ssize_t)n)) != ESP_OK) break;
    }
    fclose(f);
    httpd_resp_send_chunk(req, NULL, 0);
    return err;
}

static esp_err_t h_config_get(httpd_req_t *req) {
    bool active[MAX_SLOTS];
    read_config(active);
    char json[64];
    int n = snprintf(json, sizeof(json), "{\"active\":[");
    bool first = true;
    for (int i = 0; i < MAX_SLOTS; i++) {
        if (active[i]) {
            n += snprintf(json + n, (int)sizeof(json) - n, "%s%d", first ? "" : ",", i);
            first = false;
        }
    }
    snprintf(json + n, (int)sizeof(json) - n, "]}");
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_sendstr(req, json);
}

static esp_err_t h_config_post(httpd_req_t *req) {
    char body[256] = {};
    int len = req->content_len;
    if (len <= 0 || len >= (int)sizeof(body)) {
        return httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Bad body length");
    }
    int got = httpd_req_recv(req, body, len);
    if (got <= 0) return httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Receive error");
    body[got] = '\0';
    FILE *f = fopen(CONFIG_PATH, "w");
    if (f) { fwrite(body, 1, got, f); fclose(f); }
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_sendstr(req, "{\"ok\":true}");
}

// =============================================================================
// Init
// =============================================================================

static void init_nvs(void) {
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    }
}

static void init_spiffs(void) {
    esp_vfs_spiffs_conf_t conf = {
        .base_path              = SPIFFS_MOUNT,
        .partition_label        = "storage",
        .max_files              = 12,
        .format_if_mount_failed = true,
    };
    esp_err_t err = esp_vfs_spiffs_register(&conf);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "SPIFFS mount failed: %s", esp_err_to_name(err));
        return;
    }
    size_t total = 0, used = 0;
    esp_spiffs_info("storage", &total, &used);
    ESP_LOGI(TAG, "SPIFFS mounted: %u / %u bytes used", (unsigned)used, (unsigned)total);
}

static void init_wifi_ap(void) {
    ESP_ERROR_CHECK(esp_netif_init());

    esp_err_t err = esp_event_loop_create_default();
    if (err != ESP_ERR_INVALID_STATE) ESP_ERROR_CHECK(err);

    esp_netif_create_default_wifi_ap();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));

    wifi_config_t ap_cfg = {};
    strlcpy((char *)ap_cfg.ap.ssid, "Dosypator", sizeof(ap_cfg.ap.ssid));
    ap_cfg.ap.ssid_len       = (uint8_t)strlen("Dosypator");
    ap_cfg.ap.channel        = 6;
    ap_cfg.ap.authmode       = WIFI_AUTH_OPEN;
    ap_cfg.ap.max_connection = 4;

    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &ap_cfg));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_LOGI(TAG, "Wi-Fi AP started: SSID=Dosypator  IP=192.168.4.1");
}

static void start_https_server(void) {
    httpd_ssl_config_t config = HTTPD_SSL_CONFIG_DEFAULT();
    config.httpd.max_uri_handlers  = 16;
    config.httpd.uri_match_fn      = httpd_uri_match_wildcard;
    config.httpd.recv_wait_timeout = 30;
    config.httpd.send_wait_timeout = 30;
    config.httpd.stack_size        = 8192;

    config.servercert     = _binary_server_crt_start;
    config.servercert_len = _binary_server_crt_end - _binary_server_crt_start;
    config.prvtkey_pem    = _binary_server_key_start;
    config.prvtkey_len    = _binary_server_key_end - _binary_server_key_start;

    httpd_handle_t server = NULL;
    ESP_ERROR_CHECK(httpd_ssl_start(&server, &config));

    static const httpd_uri_t uris[] = {
        {.uri = "/",        .method = HTTP_GET,  .handler = h_root,        .user_ctx = NULL},
        {.uri = "/status",  .method = HTTP_GET,  .handler = h_status,      .user_ctx = NULL},
        {.uri = "/upload/*",.method = HTTP_POST, .handler = h_upload,      .user_ctx = NULL},
        {.uri = "/play/*",  .method = HTTP_GET,  .handler = h_play,        .user_ctx = NULL},
        {.uri = "/config",  .method = HTTP_GET,  .handler = h_config_get,  .user_ctx = NULL},
        {.uri = "/config",  .method = HTTP_POST, .handler = h_config_post, .user_ctx = NULL},
    };
    for (size_t i = 0; i < sizeof(uris) / sizeof(uris[0]); i++) {
        httpd_register_uri_handler(server, &uris[i]);
    }
    ESP_LOGI(TAG, "HTTPS server started on :443");
}

// =============================================================================
// Entry point — called from app_main
// =============================================================================

static void portal_task(void *arg) {
    init_nvs();
    init_spiffs();
    init_wifi_ap();
    start_https_server();
    ESP_LOGI(TAG, "Portal ready — connect to 'Dosypator' and open https://192.168.4.1");
    vTaskDelete(NULL);
}

void captive_portal_start(void) {
    xTaskCreate(portal_task, "portal", 8192, NULL, 4, NULL);
}

// =============================================================================
// Dosypator v2 — Baby Cry Detector for Waveshare ESP32-S3 Audio Board
//
// Hardware: ESP32-S3 + ES7210 (mic ADC) + ES8311 (DAC/speaker)
// Pins:     MCLK=12, BCLK=13, WS=14, DIN=15 (mic→ESP), DOUT=16 (ESP→spk)
//           I2C SCL=10, SDA=11
//
// Pipeline: I2S mic → microfrontend (log-mel filterbanks, 128 bands) →
//           DCT-II → 40 MFCC coefficients → TFLite INT8 CNN → p(cry)
//
// First-stage speaker test: 440 Hz sine wave when cry detected.
// Replace play_sine_440hz() with WAV playback once audio pipeline is verified.
// =============================================================================

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <inttypes.h>

// --- ESP-IDF ---
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_system.h"
#include "driver/i2s_std.h"
#include "driver/i2c.h"
#include "driver/gpio.h"
#include "esp_heap_caps.h"

// --- TFLite Micro ---
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h"

// --- KissFFT (local float-only copy, isolated from tflite-micro's int16 version) ---
#include "local_kiss_fftr.h"

// --- Captive portal (Wi-Fi AP + DNS + HTTP server for audio slot management) ---
#include "captive_portal.h"

// --- Embedded model binary (via EMBED_FILES in CMakeLists.txt) ---
extern const unsigned char _binary_dosypator_model_tflite_start[];
extern const unsigned char _binary_dosypator_model_tflite_end[];

// --- Embedded shh WAV (via EMBED_FILES in CMakeLists.txt) ---
extern const uint8_t _binary_shh_wav_start[];
extern const uint8_t _binary_shh_wav_end[];

static const char *TAG = "DOSY";

// =============================================================================
// HARDWARE CONFIGURATION
// =============================================================================

// I2C
#define I2C_PORT            I2C_NUM_0
#define I2C_SCL             GPIO_NUM_10
#define I2C_SDA             GPIO_NUM_11
#define I2C_FREQ_HZ         100000

// Codec I2C addresses
#define ES8311_ADDR         0x18
#define ES7210_ADDR         0x40

// GPIO expander
#define TCA9555_ADDR        0x20
// TCA9555 registers
#define TCA9555_REG_OUT0    0x02   // Output Port 0
#define TCA9555_REG_OUT1    0x03   // Output Port 1
#define TCA9555_REG_CFG0    0x06   // Config Port 0 (0=output, 1=input)
#define TCA9555_REG_CFG1    0x07   // Config Port 1

// I2S
#define I2S_PORT            I2S_NUM_0
#define I2S_MCLK            GPIO_NUM_12
#define I2S_BCLK            GPIO_NUM_13
#define I2S_WS              GPIO_NUM_14
#define I2S_DIN             GPIO_NUM_15   // ES7210 data out → ESP32 in
#define I2S_DOUT            GPIO_NUM_16   // ESP32 out → ES8311 data in

// =============================================================================
// AUDIO AND MFCC PARAMETERS
// Pipeline matches train_firmware_compatible.py exactly:
//   melspectrogram(sr=16000, n_fft=2048, hop_length=512, n_mels=128, center=True)
//   log(mel + 1e-6), DCT-II ortho → 40 coefficients
//   normalize: (x - MFCC_MEAN) / MFCC_STD  (stats from mfcc_stats.npy)
// =============================================================================

#define SAMPLE_RATE         16000
#define MCLK_MULTIPLE       I2S_MCLK_MULTIPLE_256  // MCLK = 256 * 16000 = 4.096 MHz

#define FRAME_SIZE          2048    // n_fft
#define HOP_SIZE            512     // hop_length
#define N_MEL               128     // mel filterbank channels
#define N_MFCC              40      // output MFCC coefficients
#define N_FFT_BINS          (FRAME_SIZE / 2 + 1)   // 1025 real FFT output bins

// Segment: 3 seconds at 16 kHz
#define SEGMENT_SAMPLES     48000
#define N_TIME_FRAMES       94      // 1 + SEGMENT_SAMPLES / HOP_SIZE

// Audio buffer: center-pad FRAME_SIZE/2 zeros on each side → total 50048 floats
#define AUDIO_BUF_LEN       (SEGMENT_SAMPLES + FRAME_SIZE)  // 50048
#define CENTER_PAD          (FRAME_SIZE / 2)                // 1024

// Normalisation stats computed by train_firmware_compatible.py on the training set
#define MFCC_MEAN           (-1.52898598f)
#define MFCC_STD            (13.41511822f)
#define MFCC_RSTD           (1.0f / MFCC_STD)

// Streaming chunk size for I2S read (stereo int16, so 4 bytes per frame)
#define CHUNK_FRAMES        512

// =============================================================================
// TFLITE CONFIGURATION
// =============================================================================

// CNN ops: Conv2D×2, MaxPool2D×2, Reshape(Flatten), FullyConnected×2,
//          Logistic(sigmoid), Quantize, Dequantize — 9 ops total, reserve 10.
#define TENSOR_ARENA_SIZE   (512 * 1024)

// Cry detection threshold
#define CRY_THRESHOLD       0.50f

// 440 Hz test tone parameters
#define SINE_FREQ_HZ        440.0f
#define SINE_DURATION_MS    2000

// =============================================================================
// GLOBALS
// =============================================================================

// TFLite
namespace {
    static uint8_t* tensor_arena = nullptr;  // allocated in PSRAM at runtime
    static tflite::MicroMutableOpResolver<10> resolver;
    static const tflite::Model* tflite_model  = nullptr;
    static tflite::MicroInterpreter* interpreter = nullptr;
    static TfLiteTensor* input_tensor  = nullptr;
    static TfLiteTensor* output_tensor = nullptr;
}

// I2S channel handles (full-duplex)
static i2s_chan_handle_t tx_handle = nullptr;
static i2s_chan_handle_t rx_handle = nullptr;

// MFCC output buffer: [N_TIME_FRAMES][N_MFCC] floats = 94×40×4 = ~15 KB
static float mfcc_features[N_TIME_FRAMES][N_MFCC];

// MFCC pipeline state (replaces TFLM microfrontend)
struct MelBand {
    int   start, end;               // first/last FFT bin (inclusive)
    float f_start, f_peak, f_end;   // Hz boundaries
    float enorm;                    // Slaney normalisation: 2/(f_end-f_start)
};
static MelBand       mel_bands[N_MEL];
static float         hann_win[FRAME_SIZE];    // periodic Hann window
static float*        audio_buf = nullptr;     // PSRAM: AUDIO_BUF_LEN floats
static kiss_fftr_cfg fft_cfg   = nullptr;

// Frame-level work buffers (static to avoid 4 KB+ stack allocation)
static float        s_fft_in[FRAME_SIZE];
static kiss_fft_cpx s_fft_out[N_FFT_BINS];
static float        s_power[N_FFT_BINS];
static float        s_mel[N_MEL];

// Streaming reusable buffers (static = no stack pressure)
static int16_t s_stereo_buf[CHUNK_FRAMES * 2];  // stereo: L+R interleaved
static int16_t s_mono_buf[CHUNK_FRAMES];         // left channel extracted

// =============================================================================
// I2C HELPERS
// =============================================================================

static esp_err_t codec_write_reg(uint8_t dev_addr, uint8_t reg, uint8_t val)
{
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (dev_addr << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(cmd, reg, true);
    i2c_master_write_byte(cmd, val, true);
    i2c_master_stop(cmd);
    esp_err_t ret = i2c_master_cmd_begin(I2C_PORT, cmd, pdMS_TO_TICKS(100));
    i2c_cmd_link_delete(cmd);
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "I2C write to 0x%02X reg 0x%02X failed: %s",
                 dev_addr, reg, esp_err_to_name(ret));
    }
    return ret;
}

// =============================================================================
// I2C BUS INIT
// =============================================================================

static void init_i2c(void)
{
    i2c_config_t conf = {};
    conf.mode             = I2C_MODE_MASTER;
    conf.sda_io_num       = I2C_SDA;
    conf.scl_io_num       = I2C_SCL;
    conf.sda_pullup_en    = GPIO_PULLUP_ENABLE;
    conf.scl_pullup_en    = GPIO_PULLUP_ENABLE;
    conf.master.clk_speed = I2C_FREQ_HZ;

    ESP_ERROR_CHECK(i2c_param_config(I2C_PORT, &conf));
    ESP_ERROR_CHECK(i2c_driver_install(I2C_PORT, I2C_MODE_MASTER, 0, 0, 0));
    ESP_LOGI(TAG, "I2C ready: SCL=GPIO%d SDA=GPIO%d @ %d Hz",
             I2C_SCL, I2C_SDA, I2C_FREQ_HZ);
}

// =============================================================================
// TCA9555 GPIO EXPANDER INIT
//
// The Waveshare ESP32-S3 Audio Board uses a TCA9555 (I2C 0x20) to control
// power-enable signals: PA_EN (amplifier), MIC_EN (mic bias), and others.
// All 16 pins are configured as outputs and driven HIGH to activate every
// enable line. Adjust per-pin below if any signal is active-low.
// =============================================================================

static void init_tca9555(void)
{
    ESP_LOGI(TAG, "Initializing TCA9555 GPIO expander (addr=0x%02X)...", TCA9555_ADDR);

    // Drive all outputs HIGH before switching to output mode
    // to avoid any glitch where pins briefly float low.
    esp_err_t r0 = codec_write_reg(TCA9555_ADDR, TCA9555_REG_OUT0, 0xFF);
    esp_err_t r1 = codec_write_reg(TCA9555_ADDR, TCA9555_REG_OUT1, 0xFF);

    // Configure all 16 pins as outputs (0 = output in TCA9555)
    esp_err_t r2 = codec_write_reg(TCA9555_ADDR, TCA9555_REG_CFG0, 0x00);
    esp_err_t r3 = codec_write_reg(TCA9555_ADDR, TCA9555_REG_CFG1, 0x00);

    if (r0 == ESP_OK && r1 == ESP_OK && r2 == ESP_OK && r3 == ESP_OK) {
        ESP_LOGI(TAG, "TCA9555: all 16 pins → OUTPUT HIGH (PA_EN, MIC_EN, etc. activated)");
    } else {
        ESP_LOGE(TAG, "TCA9555: one or more I2C writes FAILED — "
                      "r0=%s r1=%s r2=%s r3=%s",
                 esp_err_to_name(r0), esp_err_to_name(r1),
                 esp_err_to_name(r2), esp_err_to_name(r3));
    }

    vTaskDelay(pdMS_TO_TICKS(10));  // give amplifier/mic bias time to stabilise
}

// =============================================================================
// ES8311 CODEC INIT (speaker DAC)
//
// Target: 16 kHz, 16-bit, standard I2S (Philips), I2S slave mode.
// MCLK = 4.096 MHz (256 * fs), provided by ESP32-S3 I2S master.
//
// Register values derived from ESP-ADF es8311.c for MCLK_DIV=256, fs=16 kHz.
// If speaker is silent: check I2C ACK on 0x18, verify MCLK signal on GPIO12,
// and consult the ES8311 datasheet for your board revision.
// =============================================================================

static void init_es8311(void)
{
    ESP_LOGI(TAG, "Initializing ES8311 (speaker codec, addr=0x%02X)...", ES8311_ADDR);

    // Reset
    codec_write_reg(ES8311_ADDR, 0x00, 0x1F);
    vTaskDelay(pdMS_TO_TICKS(20));
    codec_write_reg(ES8311_ADDR, 0x00, 0x00);   // release reset

    // Enable all internal clocks
    codec_write_reg(ES8311_ADDR, 0x01, 0x3F);
    codec_write_reg(ES8311_ADDR, 0x02, 0x00);
    codec_write_reg(ES8311_ADDR, 0x03, 0x10);   // ADC OSR
    codec_write_reg(ES8311_ADDR, 0x04, 0x20);   // DAC OSR (0x20 for 16 kHz / 256fs)
    codec_write_reg(ES8311_ADDR, 0x05, 0x00);
    codec_write_reg(ES8311_ADDR, 0x0B, 0x00);
    codec_write_reg(ES8311_ADDR, 0x0C, 0x00);
    codec_write_reg(ES8311_ADDR, 0x10, 0x1F);
    codec_write_reg(ES8311_ADDR, 0x11, 0x7F);
    codec_write_reg(ES8311_ADDR, 0x00, 0x80);   // power on, slave mode (bit6=0)

    // Clock coefficients: MCLK=4.096 MHz, rate=16 kHz, 256fs
    // pre_div=1, bclk_div=4(→3), lrck=0x00FF
    codec_write_reg(ES8311_ADDR, 0x02, 0x00);   // pre_div=1
    codec_write_reg(ES8311_ADDR, 0x03, 0x10);   // ADC OSR
    codec_write_reg(ES8311_ADDR, 0x04, 0x20);   // DAC OSR
    codec_write_reg(ES8311_ADDR, 0x05, 0x00);
    codec_write_reg(ES8311_ADDR, 0x06, 0x03);   // BCLK divider (bclk_div-1=3)
    codec_write_reg(ES8311_ADDR, 0x07, 0x00);   // LRCK high byte
    codec_write_reg(ES8311_ADDR, 0x08, 0xFF);   // LRCK low byte (256-1)

    // SDP format: standard Philips I2S, 16-bit
    // REG09/0A: bits[5:4]=00 (I2S), bits[5:2] word length: 16-bit=(3<<2)=0x0C
    codec_write_reg(ES8311_ADDR, 0x09, 0x0C);
    codec_write_reg(ES8311_ADDR, 0x0A, 0x0C);

    // Analog / system power-up
    codec_write_reg(ES8311_ADDR, 0x13, 0x10);
    codec_write_reg(ES8311_ADDR, 0x1B, 0x0A);   // ADC HPF
    codec_write_reg(ES8311_ADDR, 0x1C, 0x6A);   // ADC equaliser bypass, cancel DC

    // Start DAC path
    codec_write_reg(ES8311_ADDR, 0x17, 0xBF);   // ADC volume = 0 dB
    codec_write_reg(ES8311_ADDR, 0x0E, 0x02);   // enable analog PGA + ADC modulator
    codec_write_reg(ES8311_ADDR, 0x12, 0x00);   // power up DAC
    codec_write_reg(ES8311_ADDR, 0x14, 0x1A);   // enable analog MIC input + PGA gain
    codec_write_reg(ES8311_ADDR, 0x0D, 0x01);   // power up analog circuitry
    codec_write_reg(ES8311_ADDR, 0x15, 0x40);   // ADC ramp rate
    codec_write_reg(ES8311_ADDR, 0x37, 0x48);   // DAC ramp rate
    codec_write_reg(ES8311_ADDR, 0x45, 0x00);   // GP control (power on)
    codec_write_reg(ES8311_ADDR, 0x32, 0xFF);   // DAC digital volume = maximum

    ESP_LOGI(TAG, "ES8311 initialized.");
}

// =============================================================================
// ES7210 MICROPHONE ADC INIT
//
// ES7210: quad-microphone I2S ADC (I2C addr 0x40).
// Target: 16 kHz, 16-bit, standard I2S slave, MCLK=256fs, channels 1+2.
//
// If mic is silent: verify I2C ACK on 0x40 and check mic power/gain registers.
// =============================================================================

static void init_es7210(void)
{
    ESP_LOGI(TAG, "Initializing ES7210 (mic ADC, addr=0x%02X)...", ES7210_ADDR);

    // Reset
    codec_write_reg(ES7210_ADDR, 0x00, 0xFF);
    vTaskDelay(pdMS_TO_TICKS(20));
    codec_write_reg(ES7210_ADDR, 0x00, 0x41);   // release reset

    codec_write_reg(ES7210_ADDR, 0x01, 0x1F);   // all clocks off initially

    // Timing registers
    codec_write_reg(ES7210_ADDR, 0x09, 0x30);
    codec_write_reg(ES7210_ADDR, 0x0A, 0x30);

    // HPF enable on all channels
    codec_write_reg(ES7210_ADDR, 0x23, 0x2A);
    codec_write_reg(ES7210_ADDR, 0x22, 0x0A);
    codec_write_reg(ES7210_ADDR, 0x20, 0x0A);
    codec_write_reg(ES7210_ADDR, 0x21, 0x2A);

    // Clock coefficients: MCLK=4.096 MHz, rate=16 kHz
    // adc_div=1, doubler=1 (bit6), dll bypass=1 (bit7) → 0xC1
    codec_write_reg(ES7210_ADDR, 0x02, 0xC1);
    codec_write_reg(ES7210_ADDR, 0x07, 0x20);   // OSR
    codec_write_reg(ES7210_ADDR, 0x04, 0x01);   // LRCK high byte
    codec_write_reg(ES7210_ADDR, 0x05, 0x00);   // LRCK low byte

    // Analog: VDDA=3.3V, VMID=5kΩ, MIC bias=2.87V
    codec_write_reg(ES7210_ADDR, 0x40, 0x43);
    codec_write_reg(ES7210_ADDR, 0x41, 0x70);   // MIC1/2 bias
    codec_write_reg(ES7210_ADDR, 0x42, 0x70);   // MIC3/4 bias

    // SDP: standard Philips I2S, 16-bit, slave
    codec_write_reg(ES7210_ADDR, 0x11, 0x60);   // 16-bit word length
    codec_write_reg(ES7210_ADDR, 0x12, 0x00);   // standard I2S (not TDM)

    // Power: MIC1+MIC2 on, MIC3+MIC4 off (not populated on Waveshare board)
    codec_write_reg(ES7210_ADDR, 0x4B, 0x00);   // MIC1+MIC2 power on
    codec_write_reg(ES7210_ADDR, 0x4C, 0xFF);   // MIC3+MIC4 power off

    // Gain = 30 dB on MIC1+MIC2 (bit4=gain enable, bits[3:0]=10)
    codec_write_reg(ES7210_ADDR, 0x43, 0x1A);   // MIC1: gain enable + 30 dB
    codec_write_reg(ES7210_ADDR, 0x44, 0x1A);   // MIC2: gain enable + 30 dB

    // Start: enable all clocks, power everything up
    codec_write_reg(ES7210_ADDR, 0x01, 0x00);   // all clocks ON
    codec_write_reg(ES7210_ADDR, 0x06, 0x00);   // power-down reg: all active
    codec_write_reg(ES7210_ADDR, 0x40, 0x43);   // analog on
    codec_write_reg(ES7210_ADDR, 0x47, 0x00);   // MIC1 power on
    codec_write_reg(ES7210_ADDR, 0x48, 0x00);   // MIC2 power on
    codec_write_reg(ES7210_ADDR, 0x49, 0xFF);   // MIC3 power off
    codec_write_reg(ES7210_ADDR, 0x4A, 0xFF);   // MIC4 power off

    ESP_LOGI(TAG, "ES7210 initialized.");
}

// =============================================================================
// I2S FULL-DUPLEX INIT (ESP-IDF 5.x new driver API)
//
// One I2S port, master mode: ESP32-S3 generates MCLK, BCLK, and WS.
// RX: reads stereo audio from ES7210 (mic input via DIN=GPIO15)
// TX: sends stereo audio to ES8311 (speaker output via DOUT=GPIO16)
// =============================================================================

static void init_i2s(void)
{
    // Create a full-duplex channel pair on I2S_NUM_0
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_PORT, I2S_ROLE_MASTER);
    chan_cfg.auto_clear = true;  // TX DMA outputs silence when no data written
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, &tx_handle, &rx_handle));

    i2s_std_config_t std_cfg = {};

    // Clock: 16 kHz, MCLK = 256 * 16000 = 4.096 MHz
    std_cfg.clk_cfg.sample_rate_hz = SAMPLE_RATE;
    std_cfg.clk_cfg.clk_src        = I2S_CLK_SRC_DEFAULT;
    std_cfg.clk_cfg.mclk_multiple  = MCLK_MULTIPLE;

    // Slot: standard Philips I2S, 16-bit, stereo
    // ES7210 outputs stereo (ch1+ch2). We take left channel as our mono mic.
    std_cfg.slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
        I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO);

    // GPIO
    std_cfg.gpio_cfg.mclk = I2S_MCLK;
    std_cfg.gpio_cfg.bclk = I2S_BCLK;
    std_cfg.gpio_cfg.ws   = I2S_WS;
    std_cfg.gpio_cfg.dout = I2S_DOUT;
    std_cfg.gpio_cfg.din  = I2S_DIN;
    std_cfg.gpio_cfg.invert_flags.mclk_inv = false;
    std_cfg.gpio_cfg.invert_flags.bclk_inv = false;
    std_cfg.gpio_cfg.invert_flags.ws_inv   = false;

    ESP_ERROR_CHECK(i2s_channel_init_std_mode(tx_handle, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_handle, &std_cfg));

    ESP_ERROR_CHECK(i2s_channel_enable(tx_handle));
    ESP_ERROR_CHECK(i2s_channel_enable(rx_handle));

    ESP_LOGI(TAG, "I2S ready: %d Hz, MCLK=GPIO%d BCLK=GPIO%d WS=GPIO%d "
             "DIN(mic)=GPIO%d DOUT(spk)=GPIO%d",
             SAMPLE_RATE, I2S_MCLK, I2S_BCLK, I2S_WS, I2S_DIN, I2S_DOUT);
}

// =============================================================================
// MFCC PIPELINE INIT
//
// Computes exact same features as train_firmware_compatible.py:
//   1. Periodic Hann window (fftbins=True, matches librosa)
//   2. 128 triangular mel filters, HTK scale, 0–8000 Hz, Slaney normalised
//   3. kiss_fftr 2048-point real FFT
//   4. Power spectrum → mel filterbank → log(mel + 1e-6) → DCT-II ortho
//   5. Normalise: (x - MFCC_MEAN) / MFCC_STD
// =============================================================================

static bool init_mfcc_pipeline(void)
{
    // --- Periodic Hann window (librosa default: fftbins=True) ---
    for (int i = 0; i < FRAME_SIZE; i++) {
        hann_win[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / FRAME_SIZE));
    }
    // Sanity: win[0]=0.0, win[512]≈0.5, win[1024]=1.0, win[2047]≈0.0
    ESP_LOGI(TAG, "[HANN] win[0]=%.5f win[512]=%.5f win[1024]=%.5f win[2047]=%.5f",
             hann_win[0], hann_win[512], hann_win[1024], hann_win[2047]);

    // --- HTK mel filterbank (librosa defaults: fmin=0, fmax=sr/2, norm='slaney') ---
    auto hz2mel = [](float hz) -> float {
        return 2595.0f * log10f(1.0f + hz / 700.0f);
    };
    auto mel2hz = [](float mel) -> float {
        return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
    };

    const float fmax     = (float)SAMPLE_RATE / 2.0f;   // 8000 Hz
    const float mel_min  = hz2mel(0.0f);                 // = 0
    const float mel_max  = hz2mel(fmax);                 // ≈ 2840

    // n_mels+2 equally spaced mel points
    float mel_pts[N_MEL + 2];
    float hz_pts[N_MEL + 2];
    for (int i = 0; i < N_MEL + 2; i++) {
        mel_pts[i] = mel_min + (mel_max - mel_min) * (float)i / (float)(N_MEL + 1);
        hz_pts[i]  = mel2hz(mel_pts[i]);
    }

    for (int k = 0; k < N_MEL; k++) {
        mel_bands[k].f_start = hz_pts[k];
        mel_bands[k].f_peak  = hz_pts[k + 1];
        mel_bands[k].f_end   = hz_pts[k + 2];
        mel_bands[k].enorm   = 2.0f / (hz_pts[k + 2] - hz_pts[k]);  // Slaney

        // FFT bin range (inclusive): b * sr / n_fft = f_start..f_end
        mel_bands[k].start = (int)floorf(hz_pts[k]     * FRAME_SIZE / SAMPLE_RATE);
        mel_bands[k].end   = (int)ceilf (hz_pts[k + 2] * FRAME_SIZE / SAMPLE_RATE);
        if (mel_bands[k].start < 0)           mel_bands[k].start = 0;
        if (mel_bands[k].end   > N_FFT_BINS-1) mel_bands[k].end   = N_FFT_BINS - 1;
    }

    // --- kiss_fftr config (2048-pt, forward, internal malloc) ---
    fft_cfg = kiss_fftr_alloc(FRAME_SIZE, 0, NULL, NULL);
    if (!fft_cfg) {
        ESP_LOGE(TAG, "kiss_fftr_alloc failed");
        return false;
    }

    // --- FFT sanity test: 440 Hz pure sine → expect peak at bin 56 ---
    {
        const float test_freq = 440.0f;
        for (int i = 0; i < FRAME_SIZE; i++)
            s_fft_in[i] = 0.5f * sinf(2.0f * (float)M_PI * test_freq / SAMPLE_RATE * i);
        kiss_fftr(fft_cfg, s_fft_in, s_fft_out);
        // bin 56 ≈ round(440 * 2048 / 16000)
        float p56  = s_fft_out[56].r * s_fft_out[56].r + s_fft_out[56].i * s_fft_out[56].i;
        float p100 = s_fft_out[100].r * s_fft_out[100].r + s_fft_out[100].i * s_fft_out[100].i;
        float p1   = s_fft_out[1].r * s_fft_out[1].r + s_fft_out[1].i * s_fft_out[1].i;
        ESP_LOGI(TAG, "[FFT-SANITY] 440Hz: p[56]=%.2f (expect>>0)  p[1]=%.6f p[100]=%.6f (expect~0)",
                 p56, p1, p100);
        if (p56 < 1.0f)
            ESP_LOGE(TAG, "[FFT-SANITY] FAIL: FFT not producing expected peak — kiss_fftr broken!");
        else
            ESP_LOGI(TAG, "[FFT-SANITY] PASS");
        memset(s_fft_in,  0, sizeof(s_fft_in));
        memset(s_fft_out, 0, sizeof(s_fft_out));
    }

    // --- Audio buffer in PSRAM (50048 floats ≈ 196 KB) ---
    audio_buf = (float*)heap_caps_malloc(AUDIO_BUF_LEN * sizeof(float),
                                         MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!audio_buf) {
        ESP_LOGE(TAG, "audio_buf alloc failed (%u bytes)", (unsigned)(AUDIO_BUF_LEN * sizeof(float)));
        return false;
    }

    ESP_LOGI(TAG, "MFCC pipeline ready: n_fft=%d hop=%d n_mel=%d n_mfcc=%d",
             FRAME_SIZE, HOP_SIZE, N_MEL, N_MFCC);
    return true;
}

// =============================================================================
// DCT-II WITH ORTHONORMAL NORMALIZATION
//
// Matches scipy.fftpack.dct(x, type=2, norm='ortho') used in training.
//   out[k] = s(k) * Σ_{n=0}^{N-1} in[n] * cos(π k (n+0.5) / N)
//   s(0) = sqrt(1/N),  s(k>0) = sqrt(2/N)
// =============================================================================

static void dct2_ortho(const float *in, float *out, int N_in, int N_out)
{
    const float inv_N = 1.0f / (float)N_in;
    for (int k = 0; k < N_out; k++) {
        float sum = 0.0f;
        const float w = (float)M_PI * k * inv_N;
        for (int n = 0; n < N_in; n++) {
            sum += in[n] * cosf(w * (n + 0.5f));
        }
        float scale = (k == 0) ? sqrtf(inv_N) : sqrtf(2.0f * inv_N);
        out[k] = scale * sum;
    }
}

// =============================================================================
// MFCC COMPUTATION
//
// 1. Read 48000 I2S samples → audio_buf[CENTER_PAD .. CENTER_PAD+48000-1]
// 2. Pad zeros on both sides (center=True semantics, 1024 each)
// 3. For each of 94 frames:
//    - Apply Hann window to 2048-sample slice
//    - Real FFT → power spectrum
//    - Mel filterbank (128 bands, Slaney normalised)
//    - log(mel + 1e-6)
//    - DCT-II ortho → 40 coefficients
//    - Normalise: (x - MFCC_MEAN) / MFCC_STD
// =============================================================================

static bool compute_mfcc(void)
{
    // Zero the entire buffer (handles both center-pad regions)
    memset(audio_buf, 0, AUDIO_BUF_LEN * sizeof(float));

    // Read SEGMENT_SAMPLES from I2S into audio_buf[CENTER_PAD..]
    int buf_pos    = CENTER_PAD;
    int remaining  = SEGMENT_SAMPLES;

    while (remaining > 0) {
        int chunk = (remaining < CHUNK_FRAMES) ? remaining : CHUNK_FRAMES;

        size_t bytes_read = 0;
        esp_err_t err = i2s_channel_read(rx_handle, s_stereo_buf,
                                         (size_t)(chunk * 2 * sizeof(int16_t)),
                                         &bytes_read, pdMS_TO_TICKS(2000));
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "I2S read error: %s", esp_err_to_name(err));
            return false;
        }

        int n = (int)(bytes_read / (2 * sizeof(int16_t)));
        for (int i = 0; i < n; i++) {
            // Left channel, normalise int16 → float [-1, 1]
            audio_buf[buf_pos++] = (float)s_stereo_buf[i * 2] / 32768.0f;
        }
        remaining -= n;
    }

    // --- DIAG 1: Audio buffer content after I2S fill ---
    {
        // Check a few samples in the middle of the audio region
        ESP_LOGI(TAG, "[DIAG-AUDIO] buf_pos after fill = %d (expected %d)",
                 buf_pos, CENTER_PAD + SEGMENT_SAMPLES);
        ESP_LOGI(TAG, "[DIAG-AUDIO] audio_buf[1024]=%.5f [8000]=%.5f [16000]=%.5f "
                 "[24000]=%.5f [32000]=%.5f",
                 audio_buf[1024], audio_buf[8000], audio_buf[16000],
                 audio_buf[24000], audio_buf[32000]);
        // Compute RMS of the audio region to confirm signal is non-zero
        float sum2 = 0.0f;
        for (int i = CENTER_PAD; i < CENTER_PAD + 4000; i++)
            sum2 += audio_buf[i] * audio_buf[i];
        ESP_LOGI(TAG, "[DIAG-AUDIO] RMS of first 4000 samples = %.5f",
                 sqrtf(sum2 / 4000.0f));
    }

    // Process N_TIME_FRAMES frames
    for (int t = 0; t < N_TIME_FRAMES; t++) {
        const int frame_start = t * HOP_SIZE;   // index into audio_buf

        // 1. Apply Hann window
        for (int i = 0; i < FRAME_SIZE; i++) {
            s_fft_in[i] = audio_buf[frame_start + i] * hann_win[i];
        }

        // --- DIAG-PRE-FFT: check actual windowed values for frame t=5 ---
        if (t == 5) {
            // Center of window: hann_win[1024]=1.0, so fft_in[1024]=audio_buf[frame_start+1024]
            ESP_LOGI(TAG, "[DIAG-WIN] t=5 audio_buf[%d]=%.6f  hann[1024]=%.6f  fft_in[1024]=%.6f",
                     frame_start + 1024,
                     audio_buf[frame_start + 1024], hann_win[1024], s_fft_in[1024]);
            ESP_LOGI(TAG, "[DIAG-WIN] fft_in[500]=%.6f  fft_in[1000]=%.6f  fft_in[1500]=%.6f",
                     s_fft_in[500], s_fft_in[1000], s_fft_in[1500]);
        }

        // 2. Real FFT → 1025 complex bins
        kiss_fftr(fft_cfg, s_fft_in, s_fft_out);

        // 3. Power spectrum + DIAG for frame t=5
        for (int k = 0; k < N_FFT_BINS; k++) {
            s_power[k] = s_fft_out[k].r * s_fft_out[k].r
                       + s_fft_out[k].i * s_fft_out[k].i;
        }
        if (t == 5) {
            // Log raw FFT output (before squaring to power) to see if it is inf/NaN directly
            ESP_LOGI(TAG, "[DIAG-FFT-OUT] out[56].r=%.4f .i=%.4f  out[100].r=%.6f .i=%.6f",
                     s_fft_out[56].r, s_fft_out[56].i,
                     s_fft_out[100].r, s_fft_out[100].i);
            float pow_sum = 0.0f;
            for (int k = 0; k < N_FFT_BINS; k++) pow_sum += s_power[k];
            ESP_LOGI(TAG, "[DIAG-FFT-PWR] power_sum=%.3f  "
                     "p[10]=%.3f p[56]=%.3f p[100]=%.6f p[200]=%.3f p[500]=%.3f",
                     pow_sum,
                     s_power[10], s_power[56], s_power[100],
                     s_power[200], s_power[500]);
        }

        // 4. Mel filterbank + log(mel + 1e-6)
        for (int m = 0; m < N_MEL; m++) {
            float energy = 0.0f;
            const MelBand& band = mel_bands[m];

            for (int b = band.start; b <= band.end; b++) {
                float fbin   = (float)b * SAMPLE_RATE / FRAME_SIZE;
                float w_rise = (band.f_peak > band.f_start)
                               ? (fbin - band.f_start) / (band.f_peak - band.f_start)
                               : 0.0f;
                float w_fall = (band.f_end > band.f_peak)
                               ? (band.f_end - fbin)   / (band.f_end - band.f_peak)
                               : 0.0f;
                float w = fmaxf(0.0f, fminf(w_rise, w_fall)) * band.enorm;
                energy += s_power[b] * w;
            }
            s_mel[m] = logf(energy + 1e-6f);
        }

        // --- DIAG 3: Mel filterbank output for frame t=5 ---
        if (t == 5) {
            ESP_LOGI(TAG, "[DIAG-MEL] t=5 mel[0]=%.4f mel[16]=%.4f mel[32]=%.4f "
                     "mel[64]=%.4f mel[96]=%.4f mel[127]=%.4f",
                     s_mel[0], s_mel[16], s_mel[32],
                     s_mel[64], s_mel[96], s_mel[127]);
            ESP_LOGI(TAG, "[DIAG-MEL] band[0]: start=%d end=%d f_start=%.1f f_peak=%.1f f_end=%.1f",
                     mel_bands[0].start, mel_bands[0].end,
                     mel_bands[0].f_start, mel_bands[0].f_peak, mel_bands[0].f_end);
            ESP_LOGI(TAG, "[DIAG-MEL] band[32]: start=%d end=%d f_start=%.1f f_peak=%.1f f_end=%.1f",
                     mel_bands[32].start, mel_bands[32].end,
                     mel_bands[32].f_start, mel_bands[32].f_peak, mel_bands[32].f_end);
        }

        // 5. DCT-II ortho → 40 MFCC coefficients
        dct2_ortho(s_mel, mfcc_features[t], N_MEL, N_MFCC);

        // 6. Normalise: (x - mean) / std
        for (int c = 0; c < N_MFCC; c++) {
            mfcc_features[t][c] = (mfcc_features[t][c] - MFCC_MEAN) * MFCC_RSTD;
        }
    }

    return true;
}

// =============================================================================
// TFLITE MICRO INIT
//
// CNN model ops (after INT8 quantization of the Keras model):
//   Conv2D(32)  → fused ReLU (RELU activation stored inside the op)
//   MaxPool2D
//   Conv2D(64)  → fused ReLU
//   MaxPool2D
//   Reshape     (Flatten layer = Reshape op in TFLite)
//   FullyConnected(128) → fused ReLU
//   FullyConnected(1)   → Logistic (sigmoid)
// Plus Quantize / Dequantize for INT8 I/O boundary handling.
// =============================================================================

static TfLiteStatus init_tflite(void)
{
    const unsigned char *model_data = _binary_dosypator_model_tflite_start;
    tflite_model = tflite::GetModel(model_data);

    if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "TFLite schema mismatch: model=%" PRIu32 " expected=%d",
                 tflite_model->version(), TFLITE_SCHEMA_VERSION);
        return kTfLiteError;
    }

    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddReshape();
    resolver.AddFullyConnected();
    resolver.AddLogistic();     // sigmoid output activation
    resolver.AddQuantize();     // INT8 input quantization
    resolver.AddDequantize();   // float output dequantization
    resolver.AddRelu();         // standalone ReLU if not fused

    tensor_arena = (uint8_t*)heap_caps_aligned_alloc(16, TENSOR_ARENA_SIZE,
                                                      MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!tensor_arena) {
        ESP_LOGE(TAG, "Failed to allocate tensor arena (%u bytes) in PSRAM",
                 (unsigned)TENSOR_ARENA_SIZE);
        return kTfLiteError;
    }
    ESP_LOGI(TAG, "Tensor arena: %u KB allocated in PSRAM", TENSOR_ARENA_SIZE / 1024);

    interpreter = new tflite::MicroInterpreter(
        tflite_model, resolver, tensor_arena, TENSOR_ARENA_SIZE);

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed — try increasing TENSOR_ARENA_SIZE");
        return kTfLiteError;
    }

    input_tensor  = interpreter->input(0);
    output_tensor = interpreter->output(0);

    // Sanity-check input shape: expect [1, 94, 40, 1] INT8
    ESP_LOGI(TAG, "Input tensor:  type=%d dims=%d [%d,%d,%d,%d]",
             (int)input_tensor->type,
             (int)input_tensor->dims->size,
             (int)input_tensor->dims->data[0], (int)input_tensor->dims->data[1],
             (int)input_tensor->dims->data[2], (int)input_tensor->dims->data[3]);
    ESP_LOGI(TAG, "Input quant:   scale=%.6f zero_point=%d",
             input_tensor->params.scale, (int)input_tensor->params.zero_point);
    ESP_LOGI(TAG, "Output quant:  scale=%.6f zero_point=%d",
             output_tensor->params.scale, (int)output_tensor->params.zero_point);
    ESP_LOGI(TAG, "Free heap after TFLite init: %" PRIu32 " bytes",
             esp_get_free_heap_size());

    return kTfLiteOk;
}

// =============================================================================
// 440 Hz SINE WAVE PLAYBACK (speaker test — replace with WAV later)
// =============================================================================

static void play_sine_440hz(void)
{
    const int total_samples = (SAMPLE_RATE * SINE_DURATION_MS) / 1000;
    const int kBufSamples   = 256;
    // Stack-local buffer: 256 stereo frames × 4 bytes = 2 KB — safe for default stack
    int16_t buf[kBufSamples * 2];

    ESP_LOGI(TAG, "[SPK] Playing 440 Hz sine for %d ms ...", SINE_DURATION_MS);

    float phase     = 0.0f;
    const float inc = 2.0f * (float)M_PI * SINE_FREQ_HZ / (float)SAMPLE_RATE;

    int written = 0;
    while (written < total_samples) {
        int chunk = total_samples - written;
        if (chunk > kBufSamples) chunk = kBufSamples;

        for (int i = 0; i < chunk; i++) {
            int16_t s = (int16_t)(14000.0f * sinf(phase));  // ~43% of INT16 max
            buf[i * 2]     = s;   // left
            buf[i * 2 + 1] = s;   // right
            phase += inc;
            if (phase >= 2.0f * (float)M_PI) phase -= 2.0f * (float)M_PI;
        }

        size_t bytes_written = 0;
        esp_err_t werr = i2s_channel_write(tx_handle, buf,
                                           (size_t)(chunk * 2 * sizeof(int16_t)),
                                           &bytes_written, pdMS_TO_TICKS(1000));
        if (werr != ESP_OK || bytes_written == 0) {
            ESP_LOGE(TAG, "[SPK] i2s_channel_write err=%s bytes_written=%u",
                     esp_err_to_name(werr), (unsigned)bytes_written);
        }
        written += chunk;
    }

    ESP_LOGI(TAG, "[SPK] Done. Total samples written: %d (expected %d)",
             written, total_samples);
}

static void play_shh(void) {
    const int16_t *wav_data = (const int16_t *)(_binary_shh_wav_start + 44);
    size_t wav_samples = (_binary_shh_wav_end - _binary_shh_wav_start - 44) / 2;

    ESP_LOGI(TAG, "[SHH] Playing shh.wav (%d samples)...", (int)wav_samples);

    // Конвертируем mono в stereo и уменьшаем громкость в 4 раза
    static int16_t stereo_buf[512];
    size_t offset = 0;
    size_t written = 0;

    while (offset < wav_samples) {
        size_t chunk = (wav_samples - offset);
        if (chunk > 256) chunk = 256;

        for (size_t i = 0; i < chunk; i++) {
            int16_t sample = wav_data[offset + i] / 12;  // уменьшаем громкость
            stereo_buf[i * 2]     = sample;  // левый канал
            stereo_buf[i * 2 + 1] = sample;  // правый канал
        }
        i2s_channel_write(tx_handle, stereo_buf, chunk * 4, &written, 1000);
        offset += chunk;
    }
    ESP_LOGI(TAG, "[SHH] Done.");
}

// =============================================================================
// DIAGNOSTIC: MICROPHONE TEST
// Reads 1 s of I2S audio (left channel) and logs min/max/RMS.
// =============================================================================

static void test_microphone(void)
{
    const int test_samples = SAMPLE_RATE;   // 1 second
    int32_t   sum_sq       = 0;
    int16_t   s_min        = INT16_MAX;
    int16_t   s_max        = INT16_MIN;
    int       total_read   = 0;

    ESP_LOGI(TAG, "[MIC TEST] Reading 1 s of audio...");

    while (total_read < test_samples) {
        int chunk = ((test_samples - total_read) < CHUNK_FRAMES)
                    ? (test_samples - total_read) : CHUNK_FRAMES;
        size_t bytes_read = 0;
        esp_err_t err = i2s_channel_read(rx_handle,
                                         s_stereo_buf,
                                         (size_t)(chunk * 2 * sizeof(int16_t)),
                                         &bytes_read,
                                         pdMS_TO_TICKS(2000));
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "[MIC TEST] I2S read error: %s", esp_err_to_name(err));
            return;
        }
        int frames = (int)(bytes_read / (2 * sizeof(int16_t)));
        for (int i = 0; i < frames; i++) {
            int16_t s = s_stereo_buf[i * 2];  // left channel
            if (s < s_min) s_min = s;
            if (s > s_max) s_max = s;
            sum_sq += (int32_t)s * (int32_t)s;
        }
        total_read += frames;
    }

    float rms   = sqrtf((float)sum_sq / (float)total_read);
    int   range = (int)s_max - (int)s_min;

    if (range < 100) {
        ESP_LOGE(TAG, "[MIC TEST] MIC: DEAD (no signal) — range=%d, RMS=%.1f", range, rms);
    } else {
        ESP_LOGI(TAG, "[MIC TEST] MIC: OK, RMS=%.1f  min=%d  max=%d  range=%d",
                 rms, (int)s_min, (int)s_max, range);
    }
}

// =============================================================================
// MAIN INFERENCE LOOP
// =============================================================================

static float run_one_inference_cycle(void) {
    if (!compute_mfcc()) {
        ESP_LOGE(TAG, "[CYCLE] MFCC failed");
        return 0.0f;
    }

    const float q_scale      = input_tensor->params.scale;
    const int   q_zero_point = input_tensor->params.zero_point;

    int8_t *inp = input_tensor->data.int8;
    for (int t = 0; t < N_TIME_FRAMES; t++) {
        for (int m = 0; m < N_MFCC; m++) {
            int32_t q = (int32_t)roundf(mfcc_features[t][m] / q_scale)
                        + q_zero_point;
            if      (q < -128) q = -128;
            else if (q >  127) q =  127;
            inp[t * N_MFCC + m] = (int8_t)q;
        }
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "[CYCLE] Invoke() failed");
        return 0.0f;
    }

    int8_t raw = output_tensor->data.int8[0];
    float prob = (float)(raw - output_tensor->params.zero_point)
                 * output_tensor->params.scale;
    return prob;
}

static void run_inference_loop(void)
{
    // --- Hardware diagnostics before starting inference ---
    ESP_LOGI(TAG, "[DIAG] Speaker test: playing shh.wav...");
    play_shh();
    test_microphone();

    if (init_tflite() != kTfLiteOk) {
        ESP_LOGE(TAG, "TFLite init failed — halting.");
        return;
    }

    const float q_scale      = input_tensor->params.scale;
    const int   q_zero_point = input_tensor->params.zero_point;

    ESP_LOGI(TAG, "Entering inference loop. P(cry) threshold = %.2f", CRY_THRESHOLD);

    while (true) {
        ESP_LOGI(TAG, "\n[CYCLE] Capturing 3 s of audio...");

        // --- Step 1: Capture + compute MFCC ---
        if (!compute_mfcc()) {
            ESP_LOGE(TAG, "MFCC computation failed, retrying in 1 s");
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }

        // --- Step 2: Diagnostic: log MFCC statistics before quantisation ---
        {
            float mfcc_min = mfcc_features[0][0];
            float mfcc_max = mfcc_features[0][0];
            float mfcc_sum = 0.0f;
            const int total = N_TIME_FRAMES * N_MFCC;
            for (int t = 0; t < N_TIME_FRAMES; t++) {
                for (int c = 0; c < N_MFCC; c++) {
                    float v = mfcc_features[t][c];
                    if (v < mfcc_min) mfcc_min = v;
                    if (v > mfcc_max) mfcc_max = v;
                    mfcc_sum += v;
                }
            }
            float mfcc_mean = mfcc_sum / total;
            ESP_LOGI(TAG, "[MFCC] min=%.3f max=%.3f mean=%.3f  (q_scale=%.6f zp=%d)",
                     mfcc_min, mfcc_max, mfcc_mean, q_scale, q_zero_point);
            // First 10 values of frame 0
            ESP_LOGI(TAG, "[MFCC] f0: %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f",
                     mfcc_features[0][0], mfcc_features[0][1], mfcc_features[0][2],
                     mfcc_features[0][3], mfcc_features[0][4], mfcc_features[0][5],
                     mfcc_features[0][6], mfcc_features[0][7], mfcc_features[0][8],
                     mfcc_features[0][9]);
        }

        // --- Step 3: Quantize MFCC floats → INT8 input tensor ---
        // Model input layout: [1, N_TIME_FRAMES, N_MFCC, 1]
        int8_t *inp = input_tensor->data.int8;
        for (int t = 0; t < N_TIME_FRAMES; t++) {
            for (int m = 0; m < N_MFCC; m++) {
                int32_t q = (int32_t)roundf(mfcc_features[t][m] / q_scale)
                            + q_zero_point;
                if      (q < -128) q = -128;
                else if (q >  127) q =  127;
                inp[t * N_MFCC + m] = (int8_t)q;
            }
        }

        // --- Step 3: Run inference ---
        if (interpreter->Invoke() != kTfLiteOk) {
            ESP_LOGE(TAG, "Invoke() failed, retrying");
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }

        // --- Step 4: Dequantize output → probability ---
        int8_t raw = output_tensor->data.int8[0];
        float  prob = (float)(raw - output_tensor->params.zero_point)
                      * output_tensor->params.scale;

        ESP_LOGI(TAG, "[RESULT] P(cry) = %.4f", prob);

        // --- Step 5: Act ---
        if (prob > CRY_THRESHOLD) {
            ESP_LOGW(TAG, "=== CRY DETECTED (p=%.4f) — playing shh loop ===", prob);
            // Играем шикание в loop пока ребёнок плачет
            // Каждые N циклов делаем паузу и проверяем — затих ли плач
            int silent_cycles = 0;
            while (true) {
                play_shh();  // ~13 секунд шикания

                // После каждого воспроизведения — один цикл детекции
                // чтобы понять, продолжается ли плач
                float check_prob = run_one_inference_cycle();
                ESP_LOGI(TAG, "[LOOP] check P(cry)=%.4f", check_prob);

                if (check_prob < 0.25f) {
                    silent_cycles++;
                    ESP_LOGI(TAG, "[LOOP] Silent cycle %d/2", silent_cycles);
                    if (silent_cycles >= 2) {
                        ESP_LOGI(TAG, "[LOOP] Baby quiet — stopping shh");
                        break;
                    }
                } else {
                    silent_cycles = 0;  // сбрасываем счётчик если плач продолжается
                }
            }
        } else {
            ESP_LOGI(TAG, "[RESULT] No cry detected.");
        }
        // No sleep: next iteration immediately starts a new 3-second capture
    }
}

// =============================================================================
// APP_MAIN
// =============================================================================

// Inference task wrapper — runs on CPU1 with stack in PSRAM
static void inference_task(void* /*arg*/)
{
    run_inference_loop();
    vTaskDelete(nullptr);
}

#define INFERENCE_TASK_STACK_BYTES  (32 * 1024)

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "=== Dosypator v2 on Waveshare ESP32-S3 Audio Board ===");
    ESP_LOGI(TAG, "Free heap at start: %" PRIu32 " bytes", esp_get_free_heap_size());

    init_i2c();
    init_tca9555();   // enable PA_EN, MIC_EN and other power rails via GPIO expander
    init_es8311();
    init_es7210();
    init_i2s();

    if (!init_mfcc_pipeline()) {
        ESP_LOGE(TAG, "MFCC pipeline init failed — halting.");
        return;
    }

    // Allocate task stack and TCB in PSRAM, pin task to CPU1 so CPU0 IDLE
    // can service its own watchdog unimpeded during long Conv operations.
    static StaticTask_t tcb;
    static StackType_t* stack = (StackType_t*)heap_caps_malloc(
        INFERENCE_TASK_STACK_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!stack) {
        ESP_LOGE(TAG, "Failed to allocate inference task stack in PSRAM");
        return;
    }

    xTaskCreateStaticPinnedToCore(
        inference_task,
        "inference",
        INFERENCE_TASK_STACK_BYTES / sizeof(StackType_t),
        nullptr,
        5,
        stack,
        &tcb,
        1   // CPU1
    );

    captive_portal_start();  // Wi-Fi AP + DNS + HTTP server on CPU0

    // app_main returns here; the inference task keeps running on CPU1.
}

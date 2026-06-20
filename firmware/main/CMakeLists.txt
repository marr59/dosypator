idf_component_register(
    SRCS "main.cpp"
         "local_kiss_fft.c"
         "local_kiss_fftr.c"
    INCLUDE_DIRS "." "include"
    REQUIRES tflite-micro
    PRIV_REQUIRES esp_timer driver
    EMBED_FILES "dosypator_model.tflite"
)

# Match TFLM component settings - strip error strings to reduce binary size
target_compile_definitions(${COMPONENT_LIB} PRIVATE TF_LITE_STRIP_ERROR_STRINGS)

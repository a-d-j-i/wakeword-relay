cmake_minimum_required(VERSION 3.20)
project(web_trainer)

set(CMAKE_CXX_STANDARD 20)

set(FRONTEND_SOURCES
    vendor/micro_frontend/frontend.c
    vendor/micro_frontend/frontend_util.c
    vendor/micro_frontend/filterbank.c
    vendor/micro_frontend/filterbank_util.c
    vendor/micro_frontend/window.c
    vendor/micro_frontend/window_util.c
    vendor/micro_frontend/log_scale.c
    vendor/micro_frontend/log_scale_util.c
    vendor/micro_frontend/noise_reduction.c
    vendor/micro_frontend/noise_reduction_util.c
    vendor/micro_frontend/pcan_gain_control.c
    vendor/micro_frontend/pcan_gain_control_util.c
    vendor/micro_frontend/fft.cc
    vendor/micro_frontend/fft_util.cc
    vendor/micro_frontend/kiss_fft_int16.cc
    vendor/micro_frontend/log_lut.c
)

add_executable(web_trainer
    src/main.cpp
    src/nn.cpp
    src/mixednet.cpp
    ${FRONTEND_SOURCES}
    # kiss_fft.c is #included directly by kiss_fft_int16.cc — do not compile separately
)

target_include_directories(web_trainer PRIVATE
    vendor/micro_frontend
    vendor/tensorflow/lite/experimental/microfrontend/lib
    vendor/kissfft
    vendor/kissfft/tools
    vendor
)

if(EMSCRIPTEN)
    set_target_properties(web_trainer PROPERTIES
        SUFFIX ".js"
        LINK_FLAGS "\
            -sEXPORTED_RUNTIME_METHODS=['ccall','cwrap'] \
            -sEXPORTED_FUNCTIONS=['_malloc','_free','_frontend_create','_frontend_process','_frontend_destroy','_frontend_num_features','_mixednet_create','_mixednet_destroy','_mixednet_init_random','_mixednet_num_params','_mixednet_get_params','_mixednet_set_params','_mixednet_forward','_mixednet_min_frames','_adam_create','_adam_destroy','_adam_set_lr','_train_step'] \
            -sALLOW_MEMORY_GROWTH=1 \
            -sINITIAL_MEMORY=67108864 \
            -sENVIRONMENT=web,worker \
            -sMODULARIZE=1 \
            -sEXPORT_NAME=createWebTrainer \
        "
    )
endif()

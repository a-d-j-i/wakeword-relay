cmake_minimum_required(VERSION 3.20)
project(web_trainer)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 11)

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
    src/tflite_export.cpp
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
            -sEXPORTED_RUNTIME_METHODS=['ccall','cwrap','HEAP8','HEAPU8','HEAPU16','HEAP32','HEAPF32'] \
            -sEXPORTED_FUNCTIONS=['_malloc','_free','_frontend_create','_frontend_process','_frontend_destroy','_frontend_num_features','_mixednet_create','_mixednet_destroy','_mixednet_init_random','_mixednet_num_params','_mixednet_get_params','_mixednet_set_params','_mixednet_forward','_mixednet_min_frames','_adam_create','_adam_destroy','_adam_set_lr','_train_step','_mixednet_export_tflite'] \
            -sALLOW_MEMORY_GROWTH=1 \
            -sINITIAL_MEMORY=67108864 \
            -sENVIRONMENT=web,worker \
            -sMODULARIZE=1 \
            -sEXPORT_NAME=createWebTrainer \
        "
    )
endif()

# ── piper_phonemize ──────────────────────────────────────────────────────────
# Standalone WASM module: espeak-ng text→IPA phonemes.
# Embeds minimal en+es data files at virtual path /espeak-ng-data.

set(ESPEAK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/espeak-ng)
set(ESPEAK_DATA_DIR ${ESPEAK_DIR}/data)

file(GLOB ESPEAK_SOURCES
    ${ESPEAK_DIR}/libespeak-ng/*.c
    ${ESPEAK_DIR}/ucd-tools/*.c
)
# Exclude optional audio features not needed for phonemization.
# compiledata.c and compiledict.c export symbols referenced by dictionary.c
# (debug trace paths), so they stay in despite being primarily build-time tools.
list(REMOVE_ITEM ESPEAK_SOURCES
    ${ESPEAK_DIR}/libespeak-ng/compilembrola.c
    ${ESPEAK_DIR}/libespeak-ng/sPlayer.c
)

add_executable(piper_phonemize
    src/phonemize.cpp
    ${ESPEAK_SOURCES}
)

target_include_directories(piper_phonemize PRIVATE
    ${ESPEAK_DIR}/include
    ${ESPEAK_DIR}/libespeak-ng
    ${ESPEAK_DIR}/ucd-tools
)

target_compile_definitions(piper_phonemize PRIVATE
    LIBESPEAK_NG_EXPORT
    PATH_ESPEAK_DATA="/espeak-ng-data"
)

# espeak-ng uses C99 and expects _POSIX_C_SOURCE for some platforms.
target_compile_options(piper_phonemize PRIVATE
    -Wno-implicit-function-declaration
    -Wno-unused-parameter
)

if(EMSCRIPTEN)
    set_target_properties(piper_phonemize PROPERTIES
        SUFFIX ".js"
        LINK_FLAGS "\
            -sEXPORTED_RUNTIME_METHODS=['ccall','cwrap','UTF8ToString','FS'] \
            -sEXPORTED_FUNCTIONS=['_malloc','_free','_phonemize_init','_phonemize','_phonemize_free'] \
            -sALLOW_MEMORY_GROWTH=1 \
            -sINITIAL_MEMORY=33554432 \
            -sENVIRONMENT=web,worker \
            -sMODULARIZE=1 \
            -sEXPORT_NAME=createPiperPhonemize \
            --preload-file ${ESPEAK_DATA_DIR}@/espeak-ng-data \
        "
    )

    # Copy WASM outputs into the GitHub-Pages tree (main branch, /docs folder) —
    # commit the generated files after building. Layout:
    #   docs/trainer/  the trainer app (piper_phonemize outputs live here)
    #   docs/common/   web_trainer.js/.wasm, shared by trainer AND emulator pages
    set(WWW_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../docs/trainer)
    set(COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../docs/common)
    add_custom_command(TARGET piper_phonemize POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_BINARY_DIR}/piper_phonemize.js
            ${CMAKE_CURRENT_BINARY_DIR}/piper_phonemize.wasm
            ${CMAKE_CURRENT_BINARY_DIR}/piper_phonemize.data
            ${WWW_DIR}/
        COMMENT "Copying piper_phonemize outputs to docs/trainer/"
    )

    add_custom_command(TARGET web_trainer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_BINARY_DIR}/web_trainer.js
            ${CMAKE_CURRENT_BINARY_DIR}/web_trainer.wasm
            ${COMMON_DIR}/
        COMMENT "Copying web_trainer outputs to docs/common/"
    )
endif()

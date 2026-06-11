#pragma once
#include "mixednet.h"
#include <cstdint>
#include <cstddef>

// Build a streaming INT8 TFLite from trained MixedNet weights.
//
// tmpl / tmpl_size: the "template" TFLite produced by quantize_from_browser.py
//   (or any train.py output for pooled=0). Its graph topology is reused;
//   only weight buffers and quantization params are replaced.
// calib_n: how many synthetic spectrograms to run for INT8 calibration.
// out_size: receives byte count of returned buffer.
//
// Returns a heap-allocated buffer; caller must free().
// Returns nullptr on failure.
uint8_t* mixednet_build_tflite(const MixedNet* net,
                               const uint8_t* tmpl, size_t tmpl_size,
                               int calib_n,
                               size_t* out_size);

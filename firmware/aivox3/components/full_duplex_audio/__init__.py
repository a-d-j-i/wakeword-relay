import esphome.codegen as cg
import esphome.config_validation as cv
from esphome import pins
from esphome.components import audio, microphone
from esphome.const import CONF_ID, CONF_SAMPLE_RATE

CODEOWNERS = []
DEPENDENCIES = []
AUTO_LOAD = ["microphone", "audio"]

full_duplex_audio_ns = cg.esphome_ns.namespace("full_duplex_audio")
FullDuplexAudio = full_duplex_audio_ns.class_(
    "FullDuplexAudio",
    cg.Component,
    microphone.Microphone,
)

CONF_MCLK_PIN = "mclk_pin"
CONF_BCLK_PIN = "bclk_pin"
CONF_WS_PIN = "ws_pin"
CONF_DOUT_PIN = "dout_pin"
CONF_DIN_PIN = "din_pin"
CONF_UPLOAD_URL = "upload_url"
CONF_MIC_GAIN = "mic_gain"


def _set_stream_limits(config):
    # Fixed: 16-bit mono I2S at the configured sample rate
    audio.set_stream_limits(
        min_bits_per_sample=16,
        max_bits_per_sample=16,
        min_channels=1,
        max_channels=1,
        min_sample_rate=config[CONF_SAMPLE_RATE],
        max_sample_rate=config[CONF_SAMPLE_RATE],
    )(config)
    return config


CONFIG_SCHEMA = cv.All(
    cv.Schema(
        {
            cv.GenerateID(): cv.declare_id(FullDuplexAudio),
            cv.Required(CONF_MCLK_PIN): pins.internal_gpio_output_pin_number,
            cv.Required(CONF_BCLK_PIN): pins.internal_gpio_output_pin_number,
            cv.Required(CONF_WS_PIN): pins.internal_gpio_output_pin_number,
            cv.Required(CONF_DOUT_PIN): pins.internal_gpio_output_pin_number,
            cv.Required(CONF_DIN_PIN): pins.internal_gpio_input_pin_number,
            cv.Optional(CONF_SAMPLE_RATE, default=16000): cv.positive_int,
            cv.Optional(CONF_UPLOAD_URL, default=""): cv.string,
            cv.Optional(CONF_MIC_GAIN, default=1.0): cv.float_range(min=0.1, max=20.0),
        }
    )
    .extend(cv.COMPONENT_SCHEMA)
    .extend(microphone.MICROPHONE_SCHEMA),
    _set_stream_limits,
)


async def to_code(config):
    var = cg.new_Pvariable(config[CONF_ID])
    await cg.register_component(var, config)
    await microphone.register_microphone(var, config)
    cg.add(var.set_mclk_pin(config[CONF_MCLK_PIN]))
    cg.add(var.set_bclk_pin(config[CONF_BCLK_PIN]))
    cg.add(var.set_ws_pin(config[CONF_WS_PIN]))
    cg.add(var.set_dout_pin(config[CONF_DOUT_PIN]))
    cg.add(var.set_din_pin(config[CONF_DIN_PIN]))
    cg.add(var.set_sample_rate(config[CONF_SAMPLE_RATE]))
    cg.add(var.set_upload_url(config[CONF_UPLOAD_URL]))
    cg.add(var.set_mic_gain(config[CONF_MIC_GAIN]))

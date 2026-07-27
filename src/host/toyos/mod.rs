use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::traits::{DeviceTrait, HostTrait, StreamTrait};
use crate::{
    BufferSize, BuildStreamError, Data, DefaultStreamConfigError, DeviceDescription,
    DeviceDescriptionBuilder, DeviceId, DeviceIdError, DeviceNameError, DevicesError,
    InputCallbackInfo, OutputCallbackInfo, OutputStreamTimestamp, PauseStreamError,
    PlayStreamError, SampleFormat, SampleRate, StreamConfig, StreamError, StreamInstant,
    SupportedBufferSize, SupportedStreamConfig, SupportedStreamConfigRange,
    SupportedStreamConfigsError,
};

const CHANNELS: u16 = 2;
const SAMPLE_RATE: SampleRate = 44100;
// soundd's fixed device period — the only buffer size a stream can get, so
// it is also the only size advertised (advertised configs must build).
const PERIOD_FRAMES: crate::FrameCount = 128;

// Stream-thread state machine, also the futex word the paused thread parks on.
const STATE_PAUSED: u32 = 0;
const STATE_PLAYING: u32 = 1;
const STATE_DEAD: u32 = 2;

pub struct Host;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Device;

pub struct Stream {
    state: Arc<AtomicU32>,
    thread: Option<std::thread::JoinHandle<()>>,
    creation: std::time::Instant,
    buffer_frames: u32,
}

crate::assert_stream_send!(Stream);
crate::assert_stream_sync!(Stream);

pub type SupportedInputConfigs = crate::iter::SupportedInputConfigs;
pub type SupportedOutputConfigs = crate::iter::SupportedOutputConfigs;

#[derive(Default)]
pub struct Devices {
    yielded: bool,
}

impl Host {
    pub fn new() -> Result<Self, crate::HostUnavailable> {
        Ok(Host)
    }
}

impl HostTrait for Host {
    type Devices = Devices;
    type Device = Device;

    fn is_available() -> bool {
        true
    }

    fn devices(&self) -> Result<Self::Devices, DevicesError> {
        Ok(Devices { yielded: false })
    }

    fn default_input_device(&self) -> Option<Device> {
        None
    }

    fn default_output_device(&self) -> Option<Device> {
        Some(Device)
    }
}

impl DeviceTrait for Device {
    type SupportedInputConfigs = SupportedInputConfigs;
    type SupportedOutputConfigs = SupportedOutputConfigs;
    type Stream = Stream;

    fn name(&self) -> Result<String, DeviceNameError> {
        Ok("ToyOS Audio".to_string())
    }

    fn description(&self) -> Result<DeviceDescription, DeviceNameError> {
        Ok(DeviceDescriptionBuilder::new("ToyOS Audio".to_string()).build())
    }

    fn id(&self) -> Result<DeviceId, DeviceIdError> {
        Ok(DeviceId(crate::platform::HostId::Toyos, String::new()))
    }

    fn supported_input_configs(
        &self,
    ) -> Result<SupportedInputConfigs, SupportedStreamConfigsError> {
        Ok(Vec::new().into_iter())
    }

    fn supported_output_configs(
        &self,
    ) -> Result<SupportedOutputConfigs, SupportedStreamConfigsError> {
        Ok(vec![SupportedStreamConfigRange::new(
            CHANNELS,
            SAMPLE_RATE,
            SAMPLE_RATE,
            SupportedBufferSize::Range {
                min: PERIOD_FRAMES,
                max: PERIOD_FRAMES,
            },
            SampleFormat::I16,
        )]
        .into_iter())
    }

    fn default_input_config(&self) -> Result<SupportedStreamConfig, DefaultStreamConfigError> {
        Err(DefaultStreamConfigError::StreamTypeNotSupported)
    }

    fn default_output_config(&self) -> Result<SupportedStreamConfig, DefaultStreamConfigError> {
        Ok(SupportedStreamConfig::new(
            CHANNELS,
            SAMPLE_RATE,
            SupportedBufferSize::Range {
                min: PERIOD_FRAMES,
                max: PERIOD_FRAMES,
            },
            SampleFormat::I16,
        ))
    }

    fn build_input_stream_raw<D, E>(
        &self,
        _config: StreamConfig,
        _sample_format: SampleFormat,
        _data_callback: D,
        _error_callback: E,
        _timeout: Option<Duration>,
    ) -> Result<Self::Stream, BuildStreamError>
    where
        D: FnMut(&Data, &InputCallbackInfo) + Send + 'static,
        E: FnMut(StreamError) + Send + 'static,
    {
        Err(BuildStreamError::StreamConfigNotSupported)
    }

    fn build_output_stream_raw<D, E>(
        &self,
        config: StreamConfig,
        sample_format: SampleFormat,
        mut data_callback: D,
        mut error_callback: E,
        _timeout: Option<Duration>,
    ) -> Result<Self::Stream, BuildStreamError>
    where
        D: FnMut(&mut Data, &OutputCallbackInfo) + Send + 'static,
        E: FnMut(StreamError) + Send + 'static,
    {
        // The only config supported end-to-end today is the device native one.
        // Anything else must fail here — passing it through would make soundd
        // misinterpret the slot contents (e.g. f32 written into i16-sized
        // slots overruns them 2x).
        if config.channels != CHANNELS
            || config.sample_rate != SAMPLE_RATE
            || sample_format != SampleFormat::I16
        {
            return Err(BuildStreamError::StreamConfigNotSupported);
        }

        let channels = config.channels as usize;

        let mut audio = toyos::audio::AudioStream::open(
            config.sample_rate as u32,
            config.channels,
            toyos::audio::FORMAT_S16LE,
        ).map_err(|e| BuildStreamError::BackendSpecific {
            err: crate::BackendSpecificError {
                description: format!("failed to open audio stream: {e:?}"),
            },
        })?;

        assert_eq!(audio.device_sample_rate(), SAMPLE_RATE as u32,
            "soundd device sample rate diverged from advertised config");
        assert_eq!(audio.device_channels(), CHANNELS,
            "soundd device channel count diverged from advertised config");
        assert_eq!(audio.period_frames(), PERIOD_FRAMES,
            "soundd client period diverged from advertised buffer size");

        let buffer_frames = audio.period_frames();
        match config.buffer_size {
            BufferSize::Default => {}
            BufferSize::Fixed(n) if n == buffer_frames => {}
            BufferSize::Fixed(_) => return Err(BuildStreamError::StreamConfigNotSupported),
        }
        let buffer_samples = buffer_frames as usize * channels;

        let state = Arc::new(AtomicU32::new(STATE_PAUSED));
        let state2 = state.clone();

        let thread = std::thread::Builder::new()
            .name("cpal-toyos-audio".to_string())
            .spawn(move || {
                let mut frames_delivered: u64 = 0;
                loop {
                    match state2.load(Ordering::Acquire) {
                        STATE_PAUSED => unsafe {
                            // futex_wait revalidates the word, so a state
                            // change between load and here wakes immediately.
                            toyos_abi::syscall::futex_wait(state2.as_ptr(), STATE_PAUSED, None);
                        },
                        STATE_PLAYING => {
                            let result = audio.wait_and_fill(|buf| {
                                let now = StreamInstant::from_nanos(
                                    frames_delivered * 1_000_000_000 / SAMPLE_RATE as u64,
                                );
                                let info = OutputCallbackInfo::new(OutputStreamTimestamp {
                                    callback: now,
                                    playback: now,
                                });
                                let mut data = unsafe {
                                    Data::from_parts(
                                        buf.as_mut_ptr() as *mut (),
                                        buffer_samples,
                                        sample_format,
                                    )
                                };
                                data_callback(&mut data, &info);
                                frames_delivered += buffer_frames as u64;
                            });
                            if result.is_err() {
                                error_callback(StreamError::DeviceNotAvailable);
                                break;
                            }
                        }
                        STATE_DEAD => break,
                        other => unreachable!("invalid stream state {other}"),
                    }
                }
                audio.close();
            })
            .map_err(|e| BuildStreamError::BackendSpecific {
                err: crate::BackendSpecificError {
                    description: format!("failed to spawn audio thread: {e}"),
                },
            })?;

        Ok(Stream {
            state,
            thread: Some(thread),
            creation: std::time::Instant::now(),
            buffer_frames,
        })
    }
}

impl StreamTrait for Stream {
    fn play(&self) -> Result<(), PlayStreamError> {
        self.state.store(STATE_PLAYING, Ordering::Release);
        unsafe {
            toyos_abi::syscall::futex_wake(self.state.as_ptr(), 1);
        }
        Ok(())
    }

    fn pause(&self) -> Result<(), PauseStreamError> {
        self.state.store(STATE_PAUSED, Ordering::Release);
        Ok(())
    }

    fn buffer_size(&self) -> Result<crate::FrameCount, crate::StreamError> {
        Ok(self.buffer_frames)
    }

    fn now(&self) -> crate::StreamInstant {
        let d = self.creation.elapsed();
        crate::StreamInstant::new(d.as_secs(), d.subsec_nanos())
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        self.state.store(STATE_DEAD, Ordering::Release);
        unsafe {
            toyos_abi::syscall::futex_wake(self.state.as_ptr(), 1);
        }
        if let Some(th) = self.thread.take() {
            let _ = th.join();
        }
    }
}

impl Iterator for Devices {
    type Item = Device;
    fn next(&mut self) -> Option<Self::Item> {
        if self.yielded {
            None
        } else {
            self.yielded = true;
            Some(Device)
        }
    }
}

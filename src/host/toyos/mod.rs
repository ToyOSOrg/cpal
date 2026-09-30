use std::convert::Infallible;
use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::host::{emit_error, ErrorCallbackArc};
use crate::traits::{DeviceTrait, HostTrait, StreamTrait};
use crate::{
    BufferSize, Data, DeviceDescription, DeviceDescriptionBuilder, DeviceId, Error, ErrorKind,
    FrameCount, InputCallbackInfo, OutputCallbackInfo, OutputStreamTimestamp, SampleFormat,
    SampleRate, StreamConfig, StreamInstant, SupportedBufferSize, SupportedStreamConfig,
    SupportedStreamConfigRange,
};

const DEVICE_NAME: &str = "ToyOS Audio";
const CHANNELS: u16 = 2;
const SAMPLE_RATE: SampleRate = 44100;
// soundd's fixed device period — the only buffer size a stream can get, so
// it is also the only size advertised (advertised configs must build).
const PERIOD_FRAMES: crate::FrameCount = 128;

// How long soundd may take to let go of a closed stream. It fades the stream
// out over 5 ms of frames (`toyos_mixer::ramp_frames`), mixed a period at a
// time as the device hands periods back, and a mix loop that has not run for
// the 8 periods of the stream's ring has let the device play out everything it
// was given: past the fade and one ring, soundd is not serving the stream.
const FADE_FRAMES: u32 = SAMPLE_RATE * 5 / 1000;
const RING_PERIODS: u32 = 8;
const RELEASE_FRAMES: u32 = (FADE_FRAMES.div_ceil(PERIOD_FRAMES) + RING_PERIODS) * PERIOD_FRAMES;
const RELEASE_WITHIN: Duration =
    Duration::from_nanos(RELEASE_FRAMES as u64 * 1_000_000_000 / SAMPLE_RATE as u64);

// Stream-thread state machine, also the futex word the paused thread parks on.
const STATE_PAUSED: u32 = 0;
const STATE_PLAYING: u32 = 1;
const STATE_DEAD: u32 = 2;

pub struct Host;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Device;

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(DEVICE_NAME)
    }
}

pub struct Stream {
    state: Arc<AtomicU32>,
    thread: Option<std::thread::JoinHandle<()>>,
    // Never sent on: the stream thread holds the sender, so its end, however
    // it ends, is the disconnect `Drop` waits for. Locked by nothing; the
    // mutex makes the receiver `Sync`.
    ended: Mutex<mpsc::Receiver<Infallible>>,
    error_callback: ErrorCallbackArc,
    creation: std::time::Instant,
    buffer_frames: u32,
}

crate::assert_stream_send!(Stream);
crate::assert_stream_sync!(Stream);

pub type SupportedInputConfigs = crate::iter::SupportedInputConfigs;
pub type SupportedOutputConfigs = crate::iter::SupportedOutputConfigs;

#[derive(Clone, Default)]
pub struct Devices {
    yielded: bool,
}

impl Host {
    pub fn new() -> Result<Self, Error> {
        Ok(Host)
    }
}

impl HostTrait for Host {
    type Devices = Devices;
    type Device = Device;

    fn is_available() -> bool {
        true
    }

    fn devices(&self) -> Result<Self::Devices, Error> {
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

    fn description(&self) -> Result<DeviceDescription, Error> {
        Ok(DeviceDescriptionBuilder::new(DEVICE_NAME).build())
    }

    fn id(&self) -> Result<DeviceId, Error> {
        Ok(DeviceId::new(crate::platform::HostId::Toyos, ""))
    }

    fn supported_input_configs(&self) -> Result<SupportedInputConfigs, Error> {
        Ok(Vec::new().into_iter())
    }

    fn supported_output_configs(&self) -> Result<SupportedOutputConfigs, Error> {
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

    fn default_input_config(&self) -> Result<SupportedStreamConfig, Error> {
        Err(Error::new(ErrorKind::UnsupportedOperation))
    }

    fn default_output_config(&self) -> Result<SupportedStreamConfig, Error> {
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
    ) -> Result<Self::Stream, Error>
    where
        D: FnMut(&Data, &InputCallbackInfo) + Send + 'static,
        E: FnMut(Error) + Send + 'static,
    {
        Err(Error::new(ErrorKind::UnsupportedOperation))
    }

    fn build_output_stream_raw<D, E>(
        &self,
        config: StreamConfig,
        sample_format: SampleFormat,
        mut data_callback: D,
        error_callback: E,
        _timeout: Option<Duration>,
    ) -> Result<Self::Stream, Error>
    where
        D: FnMut(&mut Data, &OutputCallbackInfo) + Send + 'static,
        E: FnMut(Error) + Send + 'static,
    {
        // The only config supported end-to-end today is the device native one.
        // Anything else must fail here — passing it through would make soundd
        // misinterpret the slot contents (e.g. f32 written into i16-sized
        // slots overruns them 2x).
        if config.channels != CHANNELS
            || config.sample_rate != SAMPLE_RATE
            || sample_format != SampleFormat::I16
        {
            return Err(Error::new(ErrorKind::UnsupportedConfig));
        }

        let channels = config.channels as usize;

        let mut audio = toyos::audio::AudioStream::open(
            config.sample_rate as u32,
            config.channels,
            toyos::audio::FORMAT_S16LE,
        ).map_err(|e| {
            Error::with_message(
                ErrorKind::BackendError,
                format!("failed to open audio stream: {e:?}"),
            )
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
            BufferSize::Fixed(_) => return Err(Error::new(ErrorKind::UnsupportedConfig)),
        }
        let buffer_samples = buffer_frames as usize * channels;

        let state = Arc::new(AtomicU32::new(STATE_PAUSED));
        let state2 = state.clone();
        let error_callback: ErrorCallbackArc = Arc::new(Mutex::new(error_callback));
        let error_callback2 = error_callback.clone();
        let (ended_tx, ended) = mpsc::channel();

        let thread = std::thread::Builder::new()
            .name("cpal-toyos-audio".to_string())
            .spawn(move || {
                let _ended = ended_tx;
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
                                // Dropped while this wait blocked: `Drop` may
                                // already have returned, so the data callback
                                // is not called again.
                                if state2.load(Ordering::Acquire) == STATE_DEAD {
                                    buf.fill(0);
                                    return;
                                }
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
                                emit_error(
                                    &error_callback2,
                                    Error::new(ErrorKind::DeviceNotAvailable),
                                );
                                break;
                            }
                        }
                        STATE_DEAD => break,
                        other => unreachable!("invalid stream state {other}"),
                    }
                }
                audio.close();
                // soundd closes the signal pipe once it has faded the stream
                // out of its mix, and this thread's end is what `Drop` waits
                // for. What soundd asks for until then is past the stream's end.
                while audio.wait_and_fill(|period| period.fill(0)).is_ok() {}
            })
            .map_err(|e| {
                Error::with_message(
                    ErrorKind::ResourceExhausted,
                    format!("failed to spawn audio thread: {e}"),
                )
            })?;

        Ok(Stream {
            state,
            thread: Some(thread),
            ended: Mutex::new(ended),
            error_callback,
            creation: std::time::Instant::now(),
            buffer_frames,
        })
    }
}

impl StreamTrait for Stream {
    fn play(&self) -> Result<(), Error> {
        self.state.store(STATE_PLAYING, Ordering::Release);
        unsafe {
            toyos_abi::syscall::futex_wake(self.state.as_ptr(), 1);
        }
        Ok(())
    }

    fn pause(&self) -> Result<(), Error> {
        self.state.store(STATE_PAUSED, Ordering::Release);
        Ok(())
    }

    fn buffer_size(&self) -> Result<FrameCount, Error> {
        Ok(self.buffer_frames)
    }

    fn now(&self) -> StreamInstant {
        let d = self.creation.elapsed();
        StreamInstant::new(d.as_secs(), d.subsec_nanos())
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        self.state.store(STATE_DEAD, Ordering::Release);
        unsafe {
            toyos_abi::syscall::futex_wake(self.state.as_ptr(), 1);
        }
        let ended = self.ended.get_mut().unwrap_or_else(|e| e.into_inner());
        match ended.recv_timeout(RELEASE_WITHIN) {
            Ok(never) => match never {},
            Err(RecvTimeoutError::Disconnected) => {
                if let Some(th) = self.thread.take() {
                    let _ = th.join();
                }
            }
            // The stream thread is left to soundd: it ends when soundd lets
            // go of the stream, or with the process.
            Err(RecvTimeoutError::Timeout) => emit_error(
                &self.error_callback,
                Error::with_message(
                    ErrorKind::HostUnavailable,
                    format!(
                        "soundd did not let go of the closed stream within {RELEASE_WITHIN:?}, \
                         its fade and a ring of periods"
                    ),
                ),
            ),
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

use objc2_core_foundation::{CFBoolean, CFNumber, CFRetained, CFString, CFType};
use objc2_io_kit::{
    IOObjectRelease, IORegistryEntryCreateCFProperty, IOServiceGetMatchingService,
    IOServiceMatching,
};
use serde_json::{Value, json};

fn memory_pressure() -> Option<&'static str> {
    let mut level: libc::c_int = 0;
    let mut size = std::mem::size_of_val(&level);
    // Read the Darwin pressure level into an integer; no write is requested.
    let status = unsafe {
        libc::sysctlbyname(
            c"kern.memorystatus_vm_pressure_level".as_ptr(),
            (&raw mut level).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if status != 0 || size != std::mem::size_of_val(&level) {
        return None;
    }
    match level {
        1 => Some("normal"),
        2 => Some("warning"),
        4 => Some("critical"),
        _ => None,
    }
}

struct Battery(u32);

impl Battery {
    fn open() -> Option<Self> {
        // IOKit consumes the matching dictionary and returns a retained service.
        let service = unsafe {
            let matching = IOServiceMatching(c"AppleSmartBattery".as_ptr())?;
            let matching = CFRetained::cast_unchecked(matching);
            IOServiceGetMatchingService(0, Some(matching))
        };
        (service != 0).then_some(Self(service))
    }

    fn property(&self, key: &str) -> Option<CFRetained<CFType>> {
        let key = CFString::from_str(key);
        // The service is held by self and the default allocator is accepted.
        unsafe { IORegistryEntryCreateCFProperty(self.0, Some(&key), None, 0) }
    }

    fn number(&self, key: &str) -> Option<i64> {
        self.property(key)?.downcast::<CFNumber>().ok()?.as_i64()
    }

    fn drain_watts(&self) -> Option<f64> {
        if self
            .property("ExternalConnected")?
            .downcast::<CFBoolean>()
            .ok()?
            .as_bool()
        {
            return None;
        }
        let (millivolts, milliamps) = (self.number("Voltage")?, self.number("InstantAmperage")?);
        (millivolts > 0 && milliamps < 0).then(|| -(milliamps as f64) * millivolts as f64 / 1e6)
    }
}

impl Drop for Battery {
    fn drop(&mut self) {
        if self.0 != 0 && IOObjectRelease(self.0) != 0 {
            tracing::debug!("无法释放电池传感器句柄");
        }
    }
}

pub(super) fn sample() -> Value {
    let components = sysinfo::Components::new_with_refreshed_list();
    let hottest = components
        .iter()
        .filter_map(|sensor| {
            let celsius = sensor.temperature()?;
            (-20.0..=130.0)
                .contains(&celsius)
                .then_some((sensor.label(), celsius))
        })
        .max_by(|a, b| a.1.total_cmp(&b.1));
    let battery = Battery::open();
    json!({
        "sampled_at": crate::store::timestamp(),
        "memory_pressure": memory_pressure(),
        "temperature": hottest.map(|(sensor, celsius)| json!({"sensor":sensor, "celsius":celsius})),
        "battery_present": battery.is_some(),
        "battery_discharge_watts": battery.and_then(|battery| battery.drain_watts())
    })
}

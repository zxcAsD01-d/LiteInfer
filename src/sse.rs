use bytes::Bytes;

#[derive(Debug, PartialEq)]
pub struct Frame {
    pub event: Option<String>,
    pub data: String,
}

#[derive(Default)]
pub struct Decoder {
    buffer: Vec<u8>,
    lines: Vec<String>,
    size: usize,
}

impl Decoder {
    pub fn feed(&mut self, bytes: &[u8], limit: usize) -> Result<Vec<Frame>, &'static str> {
        if self.buffer.len() + self.size + bytes.len() > limit {
            return Err("SSE 事件超过预算");
        }
        self.buffer.extend_from_slice(bytes);
        let mut frames = Vec::new();
        let mut consumed = 0;
        while let Some(relative) = self.buffer[consumed..].iter().position(|b| *b == b'\n') {
            let end = consumed + relative + 1;
            let raw = &self.buffer[consumed..end];
            let text = std::str::from_utf8(raw)
                .map_err(|_| "SSE 包含无效 UTF-8")?
                .trim_end_matches(['\r', '\n'])
                .to_owned();
            let size = raw.len();
            consumed = end;
            if text.is_empty() {
                if let Some(frame) = self.frame() {
                    frames.push(frame);
                }
            } else {
                self.size += size;
                self.lines.push(text);
            }
        }
        self.buffer.drain(..consumed);
        Ok(frames)
    }

    fn frame(&mut self) -> Option<Frame> {
        let mut event = None;
        let mut data = Vec::new();
        for line in self.lines.drain(..) {
            if let Some(value) = line.strip_prefix("data:") {
                data.push(value.strip_prefix(' ').unwrap_or(value).to_owned());
            } else if let Some(value) = line.strip_prefix("event:") {
                event = Some(value.trim_start().to_owned());
            }
        }
        self.size = 0;
        (!data.is_empty()).then(|| Frame {
            event,
            data: data.join("\n"),
        })
    }

    pub fn is_clean(&self) -> bool {
        self.buffer.is_empty() && self.lines.is_empty()
    }
}

pub fn data(value: &serde_json::Value) -> Bytes {
    Bytes::from(format!("data: {value}\n\n"))
}
pub fn event(name: &str, value: &serde_json::Value) -> Bytes {
    Bytes::from(format!("event: {name}\ndata: {value}\n\n"))
}

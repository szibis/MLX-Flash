use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Message {
    FetchExperts { layer: u32, experts: Vec<u32>, request_id: u64 },
    ExpertData { request_id: u64, expert_sizes: Vec<usize> },
    RoutingReport { layer: u32, activated: Vec<u32>, token_idx: u64 },
    CacheStatsResponse { entries: usize, hit_rate: f64, bytes_used: usize },
    Error { request_id: u64, message: String },
}

pub fn encode_message(msg: &Message) -> Vec<u8> {
    let json = serde_json::to_vec(msg).expect("serialization failed");
    let len = (json.len() as u32).to_be_bytes();
    let mut buf = Vec::with_capacity(4 + json.len());
    buf.extend_from_slice(&len);
    buf.extend_from_slice(&json);
    buf
}

pub fn decode_message(buf: &[u8]) -> Option<(Message, usize)> {
    if buf.len() < 4 {
        return None;
    }
    let len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    if buf.len() < 4 + len {
        return None;
    }
    let msg: Message = serde_json::from_slice(&buf[4..4 + len]).ok()?;
    Some((msg, 4 + len))
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_fetch_request() {
        let msg = Message::FetchExperts {
            layer: 3,
            experts: vec![0, 1, 7],
            request_id: 42,
        };
        let encoded = encode_message(&msg);
        // Skip the 4-byte length prefix, check the JSON payload
        let json_str = std::str::from_utf8(&encoded[4..]).unwrap();
        assert!(json_str.contains("FetchExperts"), "JSON should contain variant name");
        assert!(json_str.contains("42"), "JSON should contain request_id");
    }

    #[test]
    fn test_deserialize_fetch_request() {
        let json = r#"{"FetchExperts":{"layer":5,"experts":[2,4,6],"request_id":99}}"#;
        let len = json.len() as u32;
        let mut buf = len.to_be_bytes().to_vec();
        buf.extend_from_slice(json.as_bytes());

        let (msg, consumed) = decode_message(&buf).unwrap();
        assert_eq!(consumed, 4 + json.len());
        match msg {
            Message::FetchExperts { layer, experts, request_id } => {
                assert_eq!(layer, 5);
                assert_eq!(experts, vec![2, 4, 6]);
                assert_eq!(request_id, 99);
            }
            _ => panic!("unexpected variant"),
        }
    }

    #[test]
    fn test_serialize_expert_data() {
        let msg = Message::ExpertData {
            request_id: 7,
            expert_sizes: vec![256, 512, 1024],
        };
        let encoded = encode_message(&msg);
        let (decoded, _) = decode_message(&encoded).unwrap();
        assert_eq!(msg, decoded);
    }

    #[test]
    fn test_serialize_routing_report() {
        let msg = Message::RoutingReport {
            layer: 12,
            activated: vec![1, 3, 5, 7],
            token_idx: 1000,
        };
        let encoded = encode_message(&msg);
        let json_str = std::str::from_utf8(&encoded[4..]).unwrap();
        assert!(json_str.contains("RoutingReport"), "JSON should contain variant name");
        assert!(json_str.contains("1000"), "JSON should contain token_idx");

        // Also verify roundtrip
        let (decoded, _) = decode_message(&encoded).unwrap();
        assert_eq!(msg, decoded);
    }
}

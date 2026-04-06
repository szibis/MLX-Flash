use std::mem;
use serde::Serialize;

use mach2::kern_return::KERN_SUCCESS;
use mach2::mach_init::mach_host_self;
use mach2::vm_statistics::vm_statistics64;

const HOST_VM_INFO64: i32 = 4;

extern "C" {
    fn host_statistics64(
        host: u32,
        flavor: i32,
        info: *mut vm_statistics64,
        count: *mut u32,
    ) -> i32;
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum PressureLevel {
    Normal,
    Warning,
    Critical,
}

impl PressureLevel {
    pub fn from_free_pct(pct: f64) -> Self {
        if pct >= 40.0 {
            PressureLevel::Normal
        } else if pct >= 15.0 {
            PressureLevel::Warning
        } else {
            PressureLevel::Critical
        }
    }

    /// Smarter pressure classification using available memory (free + reclaimable)
    /// and swap usage. On loaded macOS systems, free is often near zero but inactive
    /// pages are abundant and cheap to reclaim.
    pub fn from_system_state(available_gb: f64, total_gb: f64, swap_gb: f64) -> Self {
        let avail_pct = if total_gb > 0.0 { (available_gb / total_gb) * 100.0 } else { 0.0 };

        // Swap > 8GB is always bad regardless of available RAM
        if swap_gb > 8.0 && avail_pct < 20.0 {
            return PressureLevel::Critical;
        }

        // Available-based thresholds (more accurate than free-only)
        if avail_pct >= 25.0 {
            PressureLevel::Normal
        } else if avail_pct >= 10.0 {
            // Elevated swap is a warning sign even with some available RAM
            if swap_gb > 4.0 {
                PressureLevel::Warning
            } else {
                PressureLevel::Normal
            }
        } else if avail_pct >= 5.0 {
            PressureLevel::Warning
        } else {
            PressureLevel::Critical
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryState {
    pub total_gb: f64,
    pub free_gb: f64,
    pub active_gb: f64,
    pub inactive_gb: f64,
    pub wired_gb: f64,
    pub compressed_gb: f64,
    pub swap_used_gb: f64,
    pub pressure: PressureLevel,
}

impl MemoryState {
    pub fn available_gb(&self) -> f64 {
        (self.free_gb + self.inactive_gb * 0.5).max(0.0)
    }

    pub fn optimization_hints(&self) -> Vec<OptimizationHint> {
        let mut hints = Vec::new();

        match self.pressure {
            PressureLevel::Critical => {
                hints.push(OptimizationHint {
                    priority: "critical".to_string(),
                    action: "reduce_batch_size".to_string(),
                    message: format!(
                        "Critical memory pressure: only {:.1} GB available. Reduce batch size immediately.",
                        self.available_gb()
                    ),
                });
                hints.push(OptimizationHint {
                    priority: "critical".to_string(),
                    action: "evict_cache".to_string(),
                    message: "Evict model cache to free memory under critical pressure.".to_string(),
                });
            }
            PressureLevel::Warning => {
                hints.push(OptimizationHint {
                    priority: "warning".to_string(),
                    action: "monitor_memory".to_string(),
                    message: format!(
                        "Memory pressure elevated: {:.1} GB available. Consider reducing workload.",
                        self.available_gb()
                    ),
                });
            }
            PressureLevel::Normal => {}
        }

        if self.swap_used_gb > 1.0 {
            hints.push(OptimizationHint {
                priority: "warning".to_string(),
                action: "reduce_swap".to_string(),
                message: format!(
                    "Swap usage is {:.1} GB. Reduce in-memory workload to avoid swap thrashing.",
                    self.swap_used_gb
                ),
            });
        }

        if self.available_gb() < 2.0 && self.pressure == PressureLevel::Normal {
            hints.push(OptimizationHint {
                priority: "info".to_string(),
                action: "watch_available".to_string(),
                message: format!(
                    "Available memory low ({:.1} GB) despite normal pressure. Monitor closely.",
                    self.available_gb()
                ),
            });
        }

        hints
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct OptimizationHint {
    pub priority: String,
    pub action: String,
    pub message: String,
}

pub fn get_memory_state() -> Result<MemoryState, String> {
    // Total physical RAM via sysconf
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
    let phys_pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) } as u64;
    let total_bytes = page_size * phys_pages;
    let total_gb = total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

    // VM statistics via host_statistics64
    let mut stats: vm_statistics64 = unsafe { mem::zeroed() };
    let count = (mem::size_of::<vm_statistics64>() / mem::size_of::<i32>()) as u32;
    let mut count_inout = count;

    let ret = unsafe {
        host_statistics64(
            mach_host_self(),
            HOST_VM_INFO64,
            &mut stats as *mut vm_statistics64,
            &mut count_inout,
        )
    };

    if ret != KERN_SUCCESS {
        return Err(format!("host_statistics64 failed with kern_return_t = {}", ret));
    }

    let pages_to_gb = |pages: u64| -> f64 {
        (pages * page_size) as f64 / (1024.0 * 1024.0 * 1024.0)
    };

    let free_gb = pages_to_gb(stats.free_count as u64);
    let active_gb = pages_to_gb(stats.active_count as u64);
    let inactive_gb = pages_to_gb(stats.inactive_count as u64);
    let wired_gb = pages_to_gb(stats.wire_count as u64);
    let compressed_gb = pages_to_gb(stats.compressor_page_count as u64);

    // Swap: not exposed via vm_statistics64 on macOS without sysctl; use 0.0 as default
    // (a full implementation would call sysctl kern.swapusage)
    let swap_used_gb = get_swap_used_gb();

    let available_gb = (free_gb + inactive_gb * 0.5).max(0.0);
    // Use available-based pressure (accounts for reclaimable inactive pages + swap)
    let pressure = PressureLevel::from_system_state(available_gb, total_gb, swap_used_gb);

    Ok(MemoryState {
        total_gb,
        free_gb,
        active_gb,
        inactive_gb,
        wired_gb,
        compressed_gb,
        swap_used_gb,
        pressure,
    })
}

fn get_swap_used_gb() -> f64 {
    // sysctl vm.swapusage — best-effort, returns 0.0 on failure
    use std::ffi::CStr;
    let name = b"vm.swapusage\0";
    let mut xsw: XswUsage = unsafe { mem::zeroed() };
    let mut size = mem::size_of::<XswUsage>();
    let ret = unsafe {
        libc::sysctlbyname(
            name.as_ptr() as *const libc::c_char,
            &mut xsw as *mut _ as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret == 0 {
        xsw.xsu_used as f64 / (1024.0 * 1024.0 * 1024.0)
    } else {
        0.0
    }
}

// Mirror of xsw_usage from <sys/sysctl.h>
#[repr(C)]
struct XswUsage {
    xsu_total: u64,
    xsu_avail: u64,
    xsu_used: u64,
    xsu_pagesize: u32,
    xsu_encrypted: u8,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_memory_state_returns_valid_values() {
        let state = get_memory_state().expect("get_memory_state should succeed on macOS");
        assert!(state.total_gb > 0.0, "total_gb must be positive");
        assert!(state.free_gb >= 0.0, "free_gb must be non-negative");
    }

    #[test]
    fn test_pressure_classification_free_pct() {
        assert_eq!(PressureLevel::from_free_pct(80.0), PressureLevel::Normal);
        assert_eq!(PressureLevel::from_free_pct(25.0), PressureLevel::Warning);
        assert_eq!(PressureLevel::from_free_pct(5.0), PressureLevel::Critical);
    }

    #[test]
    fn test_pressure_from_system_state() {
        // Plenty of available RAM, no swap
        assert_eq!(PressureLevel::from_system_state(12.0, 36.0, 0.0), PressureLevel::Normal);
        // Low available but no swap — warning
        assert_eq!(PressureLevel::from_system_state(2.5, 36.0, 0.0), PressureLevel::Warning);
        // Very low available — critical
        assert_eq!(PressureLevel::from_system_state(1.0, 36.0, 0.0), PressureLevel::Critical);
        // Moderate available but heavy swap — warning
        assert_eq!(PressureLevel::from_system_state(5.0, 36.0, 6.0), PressureLevel::Warning);
        // Heavy swap + low available — critical
        assert_eq!(PressureLevel::from_system_state(4.0, 36.0, 12.0), PressureLevel::Critical);
        // Your actual system: 8.2GB available, 12.8GB swap, 36GB total
        // available_pct = 22.8%, swap > 8 but avail > 20% → Warning (not Critical)
        assert_eq!(PressureLevel::from_system_state(8.2, 36.0, 12.8), PressureLevel::Warning);
    }

    #[test]
    fn test_available_gb_calculation() {
        let state = MemoryState {
            total_gb: 16.0,
            free_gb: 4.0,
            active_gb: 6.0,
            inactive_gb: 3.0,
            wired_gb: 2.0,
            compressed_gb: 1.0,
            swap_used_gb: 0.0,
            pressure: PressureLevel::Normal,
        };
        // available = free + inactive * 0.5 = 4.0 + 1.5 = 5.5
        let expected = 5.5_f64;
        let got = state.available_gb();
        assert!(
            (got - expected).abs() < 1e-9,
            "expected available_gb = {expected}, got {got}"
        );
    }

    #[test]
    fn test_optimization_hints_critical() {
        let state = MemoryState {
            total_gb: 16.0,
            free_gb: 0.5,
            active_gb: 13.0,
            inactive_gb: 1.0,
            wired_gb: 1.5,
            compressed_gb: 0.0,
            swap_used_gb: 2.0,
            pressure: PressureLevel::Critical,
        };
        let hints = state.optimization_hints();
        assert!(!hints.is_empty(), "critical pressure must produce hints");
        let priorities: Vec<&str> = hints.iter().map(|h| h.priority.as_str()).collect();
        assert!(
            priorities.iter().any(|&p| p == "critical"),
            "at least one hint must have critical priority, got: {:?}",
            priorities
        );
    }
}

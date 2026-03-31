use std::cell::Cell;
use std::fmt;

/// Per-operation counter tracking invocation count and total element count.
struct Counter {
    name: &'static str,
    calls: Cell<u64>,
    elements: Cell<u64>,
}

impl Counter {
    const fn new(name: &'static str) -> Self {
        Self {
            name,
            calls: Cell::new(0),
            elements: Cell::new(0),
        }
    }

    fn record(&self, len: usize) {
        self.calls.set(self.calls.get() + 1);
        self.elements.set(self.elements.get() + len as u64);
    }

    fn reset(&self) {
        self.calls.set(0);
        self.elements.set(0);
    }
}

impl fmt::Display for Counter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let calls = self.calls.get();
        let elements = self.elements.get();
        if calls > 0 {
            write!(f, "{}: {} calls, {} elements", self.name, calls, elements)
        } else {
            write!(f, "{}: (none)", self.name)
        }
    }
}

thread_local! {
    static EVAL_EQ: Counter = const { Counter::new("eval_eq") };
    static SCALAR_MUL_ADD: Counter = const { Counter::new("scalar_mul_add") };
    static DOT: Counter = const { Counter::new("dot") };
    static FOLD: Counter = const { Counter::new("fold") };
}

pub fn record_eval_eq(len: usize) {
    EVAL_EQ.with(|c| c.record(len));
}

pub fn record_scalar_mul_add(len: usize) {
    SCALAR_MUL_ADD.with(|c| c.record(len));
}

pub fn record_dot(len: usize) {
    DOT.with(|c| c.record(len));
}

pub fn record_fold(len: usize) {
    FOLD.with(|c| c.record(len));
}

pub fn print_summary() {
    eprintln!("=== Operation Counters ===");
    EVAL_EQ.with(|c| eprintln!("  {c}"));
    SCALAR_MUL_ADD.with(|c| eprintln!("  {c}"));
    DOT.with(|c| eprintln!("  {c}"));
    FOLD.with(|c| eprintln!("  {c}"));
    eprintln!("==========================");

    // Reset after printing.
    EVAL_EQ.with(Counter::reset);
    SCALAR_MUL_ADD.with(Counter::reset);
    DOT.with(Counter::reset);
    FOLD.with(Counter::reset);
}

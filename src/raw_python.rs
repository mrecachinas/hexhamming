//! Raw METH_FASTCALL Python bindings for the hot single-call APIs.
//!
//! Goals:
//! * Bypass PyO3's generic vectorcall trampoline (`extract_arguments_fastcall`,
//!   the `#[pyfunction]` trampoline, and PyO3's per-argument `FromPyObject`
//!   machinery) so the 3.14 interpreter can dispatch straight through its
//!   `CALL_BUILTIN_FAST` specialization.
//! * Preserve the exact CPython-visible semantics of the PyO3 bindings in
//!   `python.rs`: same exception classes, error message fragments the tests
//!   check for, and buffer-protocol acceptance/rejection rules.
//! * Provide two &str decoding paths — `PyUnicode_AsUTF8AndSize` (the public
//!   API) and a compact-ASCII fast path that reads directly from
//!   `PyASCIIObject.length` + the compact payload pointer — so we can compare
//!   them A/B and pick a safe production shape.
//! * Support (or explicitly reject) the free-threaded / GraalPy / PyPy cases,
//!   and fall back to `PyUnicode_AsUTF8AndSize` whenever the bitfield-based
//!   fast path is unsafe.
//!
//! Registration happens at module init time via `PyCFunction_NewEx`. Each
//! function is stored in the module's dict under its existing Python name with
//! `ml_flags = METH_FASTCALL | METH_KEYWORDS`, matching PyO3's keyword-capable
//! surface without its argument-extraction trampoline.

use crate::hex::hex_char_to_nibble;
use crate::{
    bytes_array_all_within_dist, bytes_array_best_within_dist, bytes_array_first_within_dist,
    hamming_distance_bytes_dispatch, hamming_distance_string_dispatch,
    hamming_distance_string_dispatch_with_max, LOOKUP,
};

use std::os::raw::{c_char, c_int};
use std::ptr;
#[cfg(all(not(any(PyPy, GraalPy)), not(Py_GIL_DISABLED), not(Py_LIMITED_API),))]
use std::sync::atomic::{AtomicBool, Ordering};

use pyo3::ffi;

// GIL-release thresholds mirror `python.rs` so behaviour is identical.
const STRING_GIL_RELEASE_THRESHOLD: usize = 4096;
const BYTES_GIL_RELEASE_THRESHOLD: usize = 16 * 1024;
const ARRAY_GIL_RELEASE_THRESHOLD: usize = 64 * 1024;
#[cfg(all(not(any(PyPy, GraalPy)), not(Py_GIL_DISABLED), not(Py_LIMITED_API),))]
static ASCII_PEEK_ENABLED: AtomicBool = AtomicBool::new(true);

// ---------------------------------------------------------------------------
// Small error helpers
// ---------------------------------------------------------------------------

#[cold]
#[inline(never)]
unsafe fn set_type_err(msg: &'static [u8]) -> *mut ffi::PyObject {
    ffi::PyErr_SetString(ffi::PyExc_TypeError, msg.as_ptr() as *const c_char);
    ptr::null_mut()
}

#[cold]
#[inline(never)]
unsafe fn set_value_err(msg: &'static [u8]) -> *mut ffi::PyObject {
    ffi::PyErr_SetString(ffi::PyExc_ValueError, msg.as_ptr() as *const c_char);
    ptr::null_mut()
}

#[cold]
#[inline(never)]
unsafe fn set_value_err_str(msg: &'static str) -> *mut ffi::PyObject {
    // Rust static &str is not NUL-terminated; SetString needs a C string. We
    // use SetObject with a PyUnicode built from UTF-8 for the message; only
    // called on error paths, so the extra alloc is fine.
    let obj = ffi::PyUnicode_FromStringAndSize(
        msg.as_ptr() as *const c_char,
        msg.len() as ffi::Py_ssize_t,
    );
    if obj.is_null() {
        return ptr::null_mut();
    }
    ffi::PyErr_SetObject(ffi::PyExc_ValueError, obj);
    ffi::Py_DecRef(obj);
    ptr::null_mut()
}

#[cold]
#[inline(never)]
unsafe fn wrong_nargs(nargs: ffi::Py_ssize_t, expected: ffi::Py_ssize_t) -> *mut ffi::PyObject {
    if nargs < expected {
        set_type_err(b"missing required positional argument\0")
    } else {
        set_type_err(b"too many positional arguments\0")
    }
}

#[cold]
#[inline(never)]
unsafe fn set_type_err_fmt(format: &[u8], args: &[*const c_char]) -> *mut ffi::PyObject {
    match args {
        [a] => ffi::PyErr_Format(ffi::PyExc_TypeError, format.as_ptr() as *const c_char, *a),
        [a, b] => ffi::PyErr_Format(
            ffi::PyExc_TypeError,
            format.as_ptr() as *const c_char,
            *a,
            *b,
        ),
        [a, b, c] => ffi::PyErr_Format(
            ffi::PyExc_TypeError,
            format.as_ptr() as *const c_char,
            *a,
            *b,
            *c,
        ),
        [a, b, c, d] => ffi::PyErr_Format(
            ffi::PyExc_TypeError,
            format.as_ptr() as *const c_char,
            *a,
            *b,
            *c,
            *d,
        ),
        _ => ptr::null_mut(),
    }
}

#[cold]
#[inline(never)]
unsafe fn missing_args(
    fn_name: &'static [u8],
    names: &[&'static [u8]],
    first: usize,
) -> *mut ffi::PyObject {
    let remaining = names.len() - first;
    match remaining {
        1 => set_type_err_fmt(
            b"%s() missing 1 required positional argument: '%s'\0",
            &[
                fn_name.as_ptr() as *const c_char,
                names[first].as_ptr() as *const c_char,
            ],
        ),
        2 => set_type_err_fmt(
            b"%s() missing 2 required positional arguments: '%s' and '%s'\0",
            &[
                fn_name.as_ptr() as *const c_char,
                names[first].as_ptr() as *const c_char,
                names[first + 1].as_ptr() as *const c_char,
            ],
        ),
        3 => set_type_err_fmt(
            b"%s() missing 3 required positional arguments: '%s', '%s', and '%s'\0",
            &[
                fn_name.as_ptr() as *const c_char,
                names[first].as_ptr() as *const c_char,
                names[first + 1].as_ptr() as *const c_char,
                names[first + 2].as_ptr() as *const c_char,
            ],
        ),
        _ => set_type_err(b"missing required positional argument\0"),
    }
}

#[cold]
#[inline(never)]
unsafe fn too_many_positional(
    fn_name: &'static [u8],
    expected: usize,
    given: ffi::Py_ssize_t,
) -> *mut ffi::PyObject {
    let expected_s = match expected {
        2 => b"2\0".as_ptr() as *const c_char,
        3 => b"3\0".as_ptr() as *const c_char,
        _ => b"?\0".as_ptr() as *const c_char,
    };
    let given_obj = ffi::PyLong_FromSsize_t(given);
    if given_obj.is_null() {
        return ptr::null_mut();
    }
    let given_s = ffi::PyObject_Str(given_obj);
    ffi::Py_DecRef(given_obj);
    if given_s.is_null() {
        return ptr::null_mut();
    }
    let mut n = 0;
    let p = ffi::PyUnicode_AsUTF8AndSize(given_s, &mut n);
    if p.is_null() {
        ffi::Py_DecRef(given_s);
        return ptr::null_mut();
    }
    ffi::PyErr_Format(
        ffi::PyExc_TypeError,
        b"%s() takes %s positional arguments but %s were given\0".as_ptr() as *const c_char,
        fn_name.as_ptr() as *const c_char,
        expected_s,
        p,
    );
    ffi::Py_DecRef(given_s);
    ptr::null_mut()
}

#[cold]
#[inline(never)]
unsafe fn duplicate_arg(fn_name: &'static [u8], name: &'static [u8]) -> *mut ffi::PyObject {
    set_type_err_fmt(
        b"%s() got multiple values for argument '%s'\0",
        &[
            fn_name.as_ptr() as *const c_char,
            name.as_ptr() as *const c_char,
        ],
    )
}

#[cold]
#[inline(never)]
unsafe fn unexpected_keyword(
    fn_name: &'static [u8],
    keyword: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    let mut n = 0;
    let p = ffi::PyUnicode_AsUTF8AndSize(keyword, &mut n);
    if p.is_null() {
        return ptr::null_mut();
    }
    ffi::PyErr_Format(
        ffi::PyExc_TypeError,
        b"%s() got an unexpected keyword argument '%s'\0".as_ptr() as *const c_char,
        fn_name.as_ptr() as *const c_char,
        p,
    );
    ptr::null_mut()
}

#[inline]
unsafe fn keyword_index(keyword: *mut ffi::PyObject, names: &[&'static [u8]]) -> Option<usize> {
    let mut n = 0;
    let p = ffi::PyUnicode_AsUTF8AndSize(keyword, &mut n);
    if p.is_null() {
        return None;
    }
    for (i, name) in names.iter().enumerate() {
        let name = &name[..name.len() - 1];
        if name.len() == n as usize
            && std::slice::from_raw_parts(p as *const u8, n as usize) == name
        {
            return Some(i);
        }
    }
    None
}

#[inline]
unsafe fn parse_fastcall_keywords(
    fn_name: &'static [u8],
    names: &[&'static [u8]],
    args: *const *mut ffi::PyObject,
    nargs: ffi::Py_ssize_t,
    kwnames: *mut ffi::PyObject,
    out: &mut [*mut ffi::PyObject],
) -> bool {
    if nargs as usize > names.len() {
        too_many_positional(fn_name, names.len(), nargs);
        return false;
    }
    for i in 0..nargs as usize {
        out[i] = *args.add(i);
    }
    let nkw = if kwnames.is_null() {
        0
    } else {
        ffi::PyTuple_GET_SIZE(kwnames) as usize
    };
    for k in 0..nkw {
        let keyword = ffi::PyTuple_GET_ITEM(kwnames, k as ffi::Py_ssize_t);
        let Some(index) = keyword_index(keyword, names) else {
            if ffi::PyErr_Occurred().is_null() {
                unexpected_keyword(fn_name, keyword);
            }
            return false;
        };
        if index < nargs as usize || !out[index].is_null() {
            duplicate_arg(fn_name, names[index]);
            return false;
        }
        out[index] = *args.add(nargs as usize + k);
    }
    for (i, value) in out.iter().enumerate() {
        if value.is_null() {
            missing_args(fn_name, names, i);
            return false;
        }
    }
    true
}

/// The PyO3 `&str` extractor produces a `TypeError` whose message contains
/// `"not an instance of 'str'"` — several tests rely on that substring.
#[cold]
#[inline(never)]
unsafe fn set_not_str_type_err_for(o: *mut ffi::PyObject) -> *mut ffi::PyObject {
    let tp = ffi::Py_TYPE(o);
    let is_none = o == ffi::Py_None();
    let name = if is_none {
        b"None\0".as_ptr() as *const c_char
    } else {
        (*tp).tp_name
    };
    if is_none {
        ffi::PyErr_Format(
            ffi::PyExc_TypeError,
            b"'%s' is not an instance of 'str'\0".as_ptr() as *const c_char,
            name,
        );
    } else {
        ffi::PyErr_Format(
            ffi::PyExc_TypeError,
            b"'%s' object is not an instance of 'str'\0".as_ptr() as *const c_char,
            name,
        );
    }
    ptr::null_mut()
}

/// Some tests check the exact fragment `"object cannot be"` for non-integer
/// `max_dist` arguments — that comes from PyO3's default `TryFrom` message
/// (`"<type>' object cannot be interpreted as an integer"`). PyLong_AsLongLong
/// on a non-int already raises with that fragment, so we just propagate.
#[inline(always)]
unsafe fn pylong_as_i64_or_err(o: *mut ffi::PyObject) -> Result<i64, ()> {
    let v = ffi::PyLong_AsLong(o);
    if v == -1 && !ffi::PyErr_Occurred().is_null() {
        return Err(());
    }
    Ok(v as i64)
}

// ---------------------------------------------------------------------------
// String extraction — public API path (PyUnicode_AsUTF8AndSize)
// ---------------------------------------------------------------------------

#[inline(always)]
unsafe fn str_utf8(o: *mut ffi::PyObject) -> Option<&'static [u8]> {
    if ffi::PyUnicode_Check(o) == 0 {
        set_not_str_type_err_for(o);
        return None;
    }
    let mut n: ffi::Py_ssize_t = 0;
    let p = ffi::PyUnicode_AsUTF8AndSize(o, &mut n);
    if p.is_null() {
        return None;
    }
    Some(std::slice::from_raw_parts(p as *const u8, n as usize))
}

// ---------------------------------------------------------------------------
// String extraction — compact-ASCII fast path
// ---------------------------------------------------------------------------
//
// SAFETY / PORTABILITY:
//
// * pyo3-ffi defines `PyASCIIObject` with each version's layout (3.10 and
//   3.11 still carry the `wstr` pointer), so `ascii.offset(1)` is where the
//   compact-ASCII payload starts on every supported version.
// * On GIL-enabled CPython 3.10 - 3.14 the `state` bitfield is
//   interned:2 | kind:3 | compact:1 | ascii:1 | ..., allocated LSB-first by
//   the compilers CPython supports on little-endian targets, so
//   `(state >> 5) & 3 == 3` means compact ASCII.
// * Free-threaded builds reorganize the bitfield, and PyPy/GraalPy use other
//   string layouts, so those builds always use `PyUnicode_AsUTF8AndSize`.
// * `check_ascii_peek` verifies the peek against the public API at module
//   init (for ASCII and non-ASCII samples) and disables it on any mismatch.

#[cfg(all(not(any(PyPy, GraalPy)), not(Py_GIL_DISABLED), not(Py_LIMITED_API),))]
#[inline(always)]
unsafe fn str_ascii_fast(o: *mut ffi::PyObject) -> Option<&'static [u8]> {
    use pyo3::ffi::PyASCIIObject;
    if ffi::PyUnicode_Check(o) == 0 {
        set_not_str_type_err_for(o);
        return None;
    }
    if !ASCII_PEEK_ENABLED.load(Ordering::Relaxed) {
        return str_utf8(o);
    }
    let ascii = o as *mut PyASCIIObject;
    // Direct read of the state bitfield. On <3.14 GIL-enabled and on 3.14
    // GIL-enabled this layout is stable; on free-threaded 3.14+ we go via
    // the safe path (compiled out here).
    let state = (*ascii).state;
    if (state >> 5) & 3 == 3 {
        // compact + ascii both set → payload immediately follows the header.
        let len = (*ascii).length as usize;
        let data = ascii.offset(1) as *const u8;
        return Some(std::slice::from_raw_parts(data, len));
    }
    str_utf8(o)
}

#[cfg(any(PyPy, GraalPy, Py_GIL_DISABLED, Py_LIMITED_API))]
#[inline(always)]
unsafe fn str_ascii_fast(o: *mut ffi::PyObject) -> Option<&'static [u8]> {
    str_utf8(o)
}

#[cfg(all(not(any(PyPy, GraalPy)), not(Py_GIL_DISABLED), not(Py_LIMITED_API),))]
unsafe fn check_ascii_peek() {
    // ASCII samples must take the fast path and match the public API exactly.
    // Non-ASCII samples (1-, 2- and 4-byte kinds) must be declined, which also
    // makes them match: a misread bitfield would return the raw payload
    // instead of the UTF-8 buffer and disable the peek.
    for sample in [
        b"".as_slice(),
        b"abc".as_slice(),
        b"0123456789abcdef".as_slice(),
        "\u{e9}".as_bytes(),
        "abc\u{e9}".as_bytes(),
        "\u{20ac}".as_bytes(),
        "\u{1f600}".as_bytes(),
    ] {
        let obj = ffi::PyUnicode_FromStringAndSize(
            sample.as_ptr() as *const c_char,
            sample.len() as ffi::Py_ssize_t,
        );
        if obj.is_null() {
            ASCII_PEEK_ENABLED.store(false, Ordering::Relaxed);
            return;
        }
        let mut n = 0;
        let utf8 = ffi::PyUnicode_AsUTF8AndSize(obj, &mut n);
        let fast = str_ascii_fast(obj);
        let ok = !utf8.is_null()
            && fast
                .map(|s| s.as_ptr() == utf8 as *const u8 && s.len() == n as usize)
                .unwrap_or(false);
        ffi::Py_DecRef(obj);
        if !ok {
            ASCII_PEEK_ENABLED.store(false, Ordering::Relaxed);
            return;
        }
    }
}

#[cfg(any(PyPy, GraalPy, Py_GIL_DISABLED, Py_LIMITED_API))]
unsafe fn check_ascii_peek() {}

// ---------------------------------------------------------------------------
// Bytes extraction — fast path (exact bytes) and buffer-protocol fallback
// ---------------------------------------------------------------------------

#[inline(always)]
unsafe fn bytes_fast(o: *mut ffi::PyObject) -> Option<&'static [u8]> {
    // Return Some(slice) if `o` is EXACTLY `bytes` (subclasses go via buffer).
    if ffi::PyBytes_CheckExact(o) != 0 {
        let p = ffi::PyBytes_AS_STRING(o) as *const u8;
        let n = ffi::Py_SIZE(o) as usize;
        return Some(std::slice::from_raw_parts(p, n));
    }
    None
}

/// Buffer view guard: acquires a read-only view; releases on Drop.
struct BufView {
    raw: ffi::Py_buffer,
    acquired: bool,
}

impl BufView {
    #[inline(always)]
    fn new() -> Self {
        Self {
            raw: unsafe { std::mem::zeroed() },
            acquired: false,
        }
    }

    /// Acquire a read-only buffer, requiring ND-contiguous + itemsize 1.
    /// On failure raises ValueError (with the message the tests expect) and
    /// returns Err(()). CPython's PyObject_GetBuffer will itself raise on
    /// non-buffer objects; we replace its message with the codified fragment
    /// where the current PyO3 wrapper does so.
    unsafe fn acquire_readonly(&mut self, obj: *mut ffi::PyObject) -> Result<(), ()> {
        // PyBUF_ND | PyBUF_FORMAT — rejects non-contiguous exports.
        let flags = ffi::PyBUF_ND | ffi::PyBUF_FORMAT;
        let rc = ffi::PyObject_GetBuffer(obj, &mut self.raw, flags);
        if rc != 0 {
            // Consume the underlying error to match the PyO3 wrapper's
            // message shape.
            ffi::PyErr_Clear();
            let supports_buffer = ffi::PyObject_CheckBuffer(obj) != 0;
            let msg: &'static [u8] = if supports_buffer {
                b"input must be contiguous\0"
            } else {
                b"error occurred while parsing arguments\0"
            };
            set_value_err(msg);
            return Err(());
        }
        self.acquired = true;

        // itemsize must be 1; if format is provided and it's not "B"/"b"/"c",
        // reject to match the existing wrapper's behavior for numpy uint32.
        if self.raw.itemsize != 1 {
            set_value_err(b"error occurred while parsing arguments\0");
            return Err(());
        }
        if !self.raw.format.is_null() {
            // Match PyO3's `u8` buffer extractor: unsigned byte/char only.
            let c0 = *self.raw.format;
            if c0 != b'B' as c_char && c0 != b'c' as c_char {
                set_value_err(b"error occurred while parsing arguments\0");
                return Err(());
            }
        }
        if self.raw.len < 0 {
            set_value_err(b"invalid buffer view\0");
            return Err(());
        }
        Ok(())
    }

    #[inline(always)]
    unsafe fn as_slice(&self) -> &[u8] {
        if self.raw.len == 0 {
            return &[];
        }
        std::slice::from_raw_parts(self.raw.buf as *const u8, self.raw.len as usize)
    }
}

impl Drop for BufView {
    fn drop(&mut self) {
        if self.acquired {
            unsafe { ffi::PyBuffer_Release(&mut self.raw) };
        }
    }
}

/// Resolve `o` to a byte slice: fast path for exact `bytes`, buffer-protocol
/// otherwise. Also returns whether a buffer view was actually acquired (so
/// the caller can decide whether GIL detachment is safe).
///
/// Returns `Err(())` on error (Python exception already set).
#[inline(always)]
unsafe fn bytes_or_buffer<'a>(
    o: *mut ffi::PyObject,
    view: &'a mut BufView,
) -> Result<(&'a [u8], bool), ()> {
    if let Some(s) = bytes_fast(o) {
        // Lifetime tricks: extend the 'static borrow to 'a; the underlying
        // PyBytes object lives at least until this call returns because the
        // caller holds a borrow (via the fastcall args array).
        return Ok((std::mem::transmute::<&[u8], &'a [u8]>(s), false));
    }
    view.acquire_readonly(o)?;
    Ok((
        std::mem::transmute::<&[u8], &'a [u8]>(view.as_slice()),
        true,
    ))
}

// ---------------------------------------------------------------------------
// GIL release helper: PyEval_SaveThread / RestoreThread
// ---------------------------------------------------------------------------

/// Re-attaches the thread state on drop, including while unwinding, so a
/// panic inside a detached section cannot reach `call_protected` (which sets a
/// Python exception) with the interpreter still detached.
struct Reattach(*mut ffi::PyThreadState);

impl Drop for Reattach {
    fn drop(&mut self) {
        unsafe { ffi::PyEval_RestoreThread(self.0) };
    }
}

#[inline(always)]
unsafe fn detach<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _reattach = Reattach(ffi::PyEval_SaveThread());
    f()
}

// ---------------------------------------------------------------------------
// Result construction helpers
// ---------------------------------------------------------------------------

#[inline(always)]
unsafe fn py_bool(b: bool) -> *mut ffi::PyObject {
    let obj = if b { ffi::Py_True() } else { ffi::Py_False() };
    ffi::Py_IncRef(obj);
    obj
}

#[inline(always)]
unsafe fn py_ulonglong(v: u64) -> *mut ffi::PyObject {
    // Hamming distances fit in a c_long for any practical input; using
    // PyLong_FromLong lets small values (< 257) hit the small-int cache
    // fast path directly, saving an allocation.
    if v <= (std::os::raw::c_long::MAX as u64) {
        ffi::PyLong_FromLong(v as std::os::raw::c_long)
    } else {
        ffi::PyLong_FromUnsignedLongLong(v)
    }
}

#[inline(always)]
unsafe fn py_longlong(v: i64) -> *mut ffi::PyObject {
    ffi::PyLong_FromLongLong(v)
}

#[inline(always)]
unsafe fn py_ssize(v: isize) -> *mut ffi::PyObject {
    ffi::PyLong_FromSsize_t(v as ffi::Py_ssize_t)
}

// ---------------------------------------------------------------------------
// hamming_distance_string — raw
// ---------------------------------------------------------------------------

macro_rules! define_two_str_impl {
    ($name:ident, $extractor:ident) => {
        unsafe extern "C" fn $name(
            _self: *mut ffi::PyObject,
            args: *mut *mut ffi::PyObject,
            nargs: ffi::Py_ssize_t,
        ) -> *mut ffi::PyObject {
            if nargs != 2 {
                return wrong_nargs(nargs, 2);
            }
            let Some(a) = $extractor(*args) else {
                return ptr::null_mut();
            };
            let Some(b) = $extractor(*args.add(1)) else {
                return ptr::null_mut();
            };
            if a.len() != b.len() {
                return set_value_err(b"strings are NOT the same length\0");
            }
            if a.is_empty() {
                return ffi::PyLong_FromLong(0);
            }
            let compute = || hamming_distance_string_dispatch(a, b);
            let r = if a.len() < STRING_GIL_RELEASE_THRESHOLD {
                compute()
            } else {
                detach(compute)
            };
            match r {
                Ok(d) => py_ulonglong(d),
                Err(msg) => set_value_err_str(msg),
            }
        }
    };
}

define_two_str_impl!(raw_hamming_distance_string_ascii, str_ascii_fast);

// ---------------------------------------------------------------------------
// hamming_distance_bytes — raw
// ---------------------------------------------------------------------------

unsafe extern "C" fn raw_hamming_distance_bytes(
    _self: *mut ffi::PyObject,
    args: *mut *mut ffi::PyObject,
    nargs: ffi::Py_ssize_t,
) -> *mut ffi::PyObject {
    if nargs != 2 {
        return wrong_nargs(nargs, 2);
    }
    // Exact-bytes fast path.
    let obj_a = *args;
    let obj_b = *args.add(1);
    let mut va = BufView::new();
    let mut vb = BufView::new();
    let Ok((a, _)) = bytes_or_buffer(obj_a, &mut va) else {
        return ptr::null_mut();
    };
    let Ok((b, _)) = bytes_or_buffer(obj_b, &mut vb) else {
        return ptr::null_mut();
    };
    if a.len() != b.len() {
        return set_value_err(b"bytes are NOT the same length\0");
    }
    if a.is_empty() {
        return ffi::PyLong_FromLong(0);
    }
    let compute = || hamming_distance_bytes_dispatch(a, b, -1);
    let d = if a.len() >= BYTES_GIL_RELEASE_THRESHOLD {
        detach(compute)
    } else {
        compute()
    };
    py_ulonglong(d)
}

// ---------------------------------------------------------------------------
// check_hexstrings_within_dist — raw
// ---------------------------------------------------------------------------

macro_rules! define_check_hex_impl {
    ($name:ident, $extractor:ident) => {
        unsafe extern "C" fn $name(
            _self: *mut ffi::PyObject,
            args: *mut *mut ffi::PyObject,
            nargs: ffi::Py_ssize_t,
        ) -> *mut ffi::PyObject {
            if nargs != 3 {
                return wrong_nargs(nargs, 3);
            }
            let Some(a) = $extractor(*args) else {
                return ptr::null_mut();
            };
            let Some(b) = $extractor(*args.add(1)) else {
                return ptr::null_mut();
            };
            let Ok(max_dist) = pylong_as_i64_or_err(*args.add(2)) else {
                return ptr::null_mut();
            };
            if max_dist < 0 {
                return set_value_err(b"`max_dist` must be >0\0");
            }
            let max_u = max_dist as u64;
            if a.len() != b.len() {
                return set_value_err(b"strings are NOT the same length\0");
            }
            if a == b {
                return py_bool(true);
            }
            if max_u >= (a.len() as u64) * 4 {
                return py_bool(true);
            }
            if a.len() >= 64 {
                let r = if a.len() < STRING_GIL_RELEASE_THRESHOLD {
                    hamming_distance_string_dispatch_with_max(a, b, max_u)
                } else {
                    detach(|| hamming_distance_string_dispatch_with_max(a, b, max_u))
                };
                return match r {
                    Ok(d) => py_bool(d != u64::MAX),
                    Err(msg) => set_value_err_str(msg),
                };
            }
            // Scalar path with early termination (mirrors python.rs verbatim).
            let len = a.len();
            let mut result: u64 = 0;
            let mut i = 0;
            while i + 4 <= len {
                let val1_0 = hex_char_to_nibble(*a.get_unchecked(i));
                let val2_0 = hex_char_to_nibble(*b.get_unchecked(i));
                let val1_1 = hex_char_to_nibble(*a.get_unchecked(i + 1));
                let val2_1 = hex_char_to_nibble(*b.get_unchecked(i + 1));
                let val1_2 = hex_char_to_nibble(*a.get_unchecked(i + 2));
                let val2_2 = hex_char_to_nibble(*b.get_unchecked(i + 2));
                let val1_3 = hex_char_to_nibble(*a.get_unchecked(i + 3));
                let val2_3 = hex_char_to_nibble(*b.get_unchecked(i + 3));
                let invalid =
                    (val1_0 | val2_0 | val1_1 | val2_1 | val1_2 | val2_2 | val1_3 | val2_3) & 0xF0;
                if invalid != 0 {
                    return set_value_err(b"hex string contains invalid char\0");
                }
                result += *LOOKUP.get_unchecked((val1_0 ^ val2_0) as usize) as u64
                    + *LOOKUP.get_unchecked((val1_1 ^ val2_1) as usize) as u64
                    + *LOOKUP.get_unchecked((val1_2 ^ val2_2) as usize) as u64
                    + *LOOKUP.get_unchecked((val1_3 ^ val2_3) as usize) as u64;
                if result > max_u {
                    return py_bool(false);
                }
                i += 4;
            }
            while i < len {
                let val1 = hex_char_to_nibble(*a.get_unchecked(i));
                let val2 = hex_char_to_nibble(*b.get_unchecked(i));
                if (val1 | val2) & 0xF0 != 0 {
                    return set_value_err(b"hex string contains invalid char\0");
                }
                result += *LOOKUP.get_unchecked((val1 ^ val2) as usize) as u64;
                if result > max_u {
                    return py_bool(false);
                }
                i += 1;
            }
            py_bool(true)
        }
    };
}

define_check_hex_impl!(raw_check_hexstrings_within_dist_ascii, str_ascii_fast);

// ---------------------------------------------------------------------------
// check_bytes_within_dist — raw
// ---------------------------------------------------------------------------

unsafe extern "C" fn raw_check_bytes_within_dist(
    _self: *mut ffi::PyObject,
    args: *mut *mut ffi::PyObject,
    nargs: ffi::Py_ssize_t,
) -> *mut ffi::PyObject {
    if nargs != 3 {
        return wrong_nargs(nargs, 3);
    }
    let Ok(max_dist) = pylong_as_i64_or_err(*args.add(2)) else {
        return ptr::null_mut();
    };
    let mut va = BufView::new();
    let mut vb = BufView::new();
    let Ok((a, _)) = bytes_or_buffer(*args, &mut va) else {
        return ptr::null_mut();
    };
    let Ok((b, _)) = bytes_or_buffer(*args.add(1), &mut vb) else {
        return ptr::null_mut();
    };
    if a.is_empty() || b.is_empty() {
        return set_value_err(b"array size must be >0\0");
    }
    if max_dist < 0 {
        return set_value_err(b"`max_dist` must be >=0\0");
    }
    if a.len() != b.len() {
        return set_value_err(b"array sizes need to be the same\0");
    }
    let compute = || hamming_distance_bytes_dispatch(a, b, max_dist);
    let d = if a.len() >= BYTES_GIL_RELEASE_THRESHOLD {
        detach(compute)
    } else {
        compute()
    };
    py_bool(d != u64::MAX)
}

// ---------------------------------------------------------------------------
// check_bytes_arrays_{first,best,all}_within_dist — raw
// ---------------------------------------------------------------------------

/// Common validation used by all three variants.
#[inline(always)]
unsafe fn validate_array_inputs(
    big: &[u8],
    small: &[u8],
    max_dist: i64,
) -> Result<(), *mut ffi::PyObject> {
    if small.is_empty() {
        return Err(set_value_err(b"`elem_to_compare` size must be >0\0"));
    }
    if max_dist < 0 {
        return Err(set_value_err(b"`max_dist` must be >=0\0"));
    }
    if big.len() % small.len() != 0 {
        return Err(set_value_err(
            b"`array_of_elems` size must be multiplier of `elem_to_compare`\0",
        ));
    }
    Ok(())
}

unsafe extern "C" fn raw_check_bytes_arrays_first_within_dist(
    _self: *mut ffi::PyObject,
    args: *mut *mut ffi::PyObject,
    nargs: ffi::Py_ssize_t,
) -> *mut ffi::PyObject {
    if nargs != 3 {
        return wrong_nargs(nargs, 3);
    }
    let Ok(max_dist) = pylong_as_i64_or_err(*args.add(2)) else {
        return ptr::null_mut();
    };
    let mut vb = BufView::new();
    let mut vs = BufView::new();
    let Ok((big, _)) = bytes_or_buffer(*args, &mut vb) else {
        return ptr::null_mut();
    };
    let Ok((small, _)) = bytes_or_buffer(*args.add(1), &mut vs) else {
        return ptr::null_mut();
    };
    if let Err(errptr) = validate_array_inputs(big, small, max_dist) {
        return errptr;
    }
    let compute = || {
        bytes_array_first_within_dist(big, small, max_dist)
            .ok()
            .flatten()
            .map(|i| i as i64)
            .unwrap_or(-1)
    };
    let r = if big.len() < ARRAY_GIL_RELEASE_THRESHOLD {
        compute()
    } else {
        detach(compute)
    };
    py_longlong(r)
}

unsafe extern "C" fn raw_check_bytes_arrays_within_dist_alias(
    slf: *mut ffi::PyObject,
    args: *mut *mut ffi::PyObject,
    nargs: ffi::Py_ssize_t,
) -> *mut ffi::PyObject {
    raw_check_bytes_arrays_first_within_dist(slf, args, nargs)
}

unsafe extern "C" fn raw_check_bytes_arrays_best_within_dist(
    _self: *mut ffi::PyObject,
    args: *mut *mut ffi::PyObject,
    nargs: ffi::Py_ssize_t,
) -> *mut ffi::PyObject {
    if nargs != 3 {
        return wrong_nargs(nargs, 3);
    }
    let Ok(max_dist) = pylong_as_i64_or_err(*args.add(2)) else {
        return ptr::null_mut();
    };
    let mut vb = BufView::new();
    let mut vs = BufView::new();
    let Ok((big, _)) = bytes_or_buffer(*args, &mut vb) else {
        return ptr::null_mut();
    };
    let Ok((small, _)) = bytes_or_buffer(*args.add(1), &mut vs) else {
        return ptr::null_mut();
    };
    if let Err(errptr) = validate_array_inputs(big, small, max_dist) {
        return errptr;
    }
    let compute = || {
        bytes_array_best_within_dist(big, small, max_dist)
            .ok()
            .flatten()
            .map(|(d, i)| (d as i64, i as i64))
            .unwrap_or((-1, -1))
    };
    let (d, i) = if big.len() < ARRAY_GIL_RELEASE_THRESHOLD {
        compute()
    } else {
        detach(compute)
    };
    // Build a 2-tuple with direct PyTuple_New + PyTuple_SET_ITEM.
    let t = ffi::PyTuple_New(2);
    if t.is_null() {
        return ptr::null_mut();
    }
    let d_obj = py_longlong(d);
    if d_obj.is_null() {
        ffi::Py_DecRef(t);
        return ptr::null_mut();
    }
    let i_obj = py_longlong(i);
    if i_obj.is_null() {
        ffi::Py_DecRef(t);
        ffi::Py_DecRef(d_obj);
        return ptr::null_mut();
    }
    ffi::PyTuple_SET_ITEM(t, 0, d_obj);
    ffi::PyTuple_SET_ITEM(t, 1, i_obj);
    t
}

unsafe extern "C" fn raw_check_bytes_arrays_all_within_dist(
    _self: *mut ffi::PyObject,
    args: *mut *mut ffi::PyObject,
    nargs: ffi::Py_ssize_t,
) -> *mut ffi::PyObject {
    if nargs != 3 {
        return wrong_nargs(nargs, 3);
    }
    let Ok(max_dist) = pylong_as_i64_or_err(*args.add(2)) else {
        return ptr::null_mut();
    };
    let mut vb = BufView::new();
    let mut vs = BufView::new();
    let Ok((big, _)) = bytes_or_buffer(*args, &mut vb) else {
        return ptr::null_mut();
    };
    let Ok((small, _)) = bytes_or_buffer(*args.add(1), &mut vs) else {
        return ptr::null_mut();
    };
    if let Err(errptr) = validate_array_inputs(big, small, max_dist) {
        return errptr;
    }
    let compute = || bytes_array_all_within_dist(big, small, max_dist).unwrap_or_default();
    let matches = if big.len() < ARRAY_GIL_RELEASE_THRESHOLD {
        compute()
    } else {
        detach(compute)
    };
    // Build list of (distance, index) tuples with direct C API.
    let n = matches.len() as ffi::Py_ssize_t;
    let list = ffi::PyList_New(n);
    if list.is_null() {
        return ptr::null_mut();
    }
    for (k, &(dist, idx)) in matches.iter().enumerate() {
        let t = ffi::PyTuple_New(2);
        if t.is_null() {
            ffi::Py_DecRef(list);
            return ptr::null_mut();
        }
        let d_obj = py_ulonglong(dist);
        let i_obj = py_ssize(idx as isize);
        if d_obj.is_null() || i_obj.is_null() {
            if !d_obj.is_null() {
                ffi::Py_DecRef(d_obj);
            }
            if !i_obj.is_null() {
                ffi::Py_DecRef(i_obj);
            }
            ffi::Py_DecRef(t);
            ffi::Py_DecRef(list);
            return ptr::null_mut();
        }
        ffi::PyTuple_SET_ITEM(t, 0, d_obj);
        ffi::PyTuple_SET_ITEM(t, 1, i_obj);
        ffi::PyList_SET_ITEM(list, k as ffi::Py_ssize_t, t);
    }
    list
}

// ---------------------------------------------------------------------------
// hamming_distances_bytes — raw (list-returning batch API)
// ---------------------------------------------------------------------------

unsafe extern "C" fn raw_hamming_distances_bytes(
    _self: *mut ffi::PyObject,
    args: *mut *mut ffi::PyObject,
    nargs: ffi::Py_ssize_t,
) -> *mut ffi::PyObject {
    if nargs != 3 {
        return wrong_nargs(nargs, 3);
    }
    let index = ffi::PyNumber_Index(*args.add(2));
    if index.is_null() {
        return ptr::null_mut();
    }
    let element_size = ffi::PyLong_AsUnsignedLongLong(index);
    ffi::Py_DecRef(index);
    if element_size == u64::MAX && !ffi::PyErr_Occurred().is_null() {
        return ptr::null_mut();
    }
    if element_size == 0 {
        return set_value_err(b"`element_size` must be >0\0");
    }
    let mut va = BufView::new();
    let mut vb = BufView::new();
    let Ok((a, a_view)) = bytes_or_buffer(*args, &mut va) else {
        return ptr::null_mut();
    };
    let Ok((b, b_view)) = bytes_or_buffer(*args.add(1), &mut vb) else {
        return ptr::null_mut();
    };
    let element_size = element_size as usize;
    let can_detach = !(a_view || b_view);
    let compute = || crate::bytes_pairwise_distances(a, b, element_size);
    let r = if a.len() < ARRAY_GIL_RELEASE_THRESHOLD || !can_detach {
        compute()
    } else {
        detach(compute)
    };
    let dists = match r {
        Ok(d) => d,
        Err(msg) => return set_value_err_str(msg),
    };
    let n = dists.len() as ffi::Py_ssize_t;
    let list = ffi::PyList_New(n);
    if list.is_null() {
        return ptr::null_mut();
    }
    for (k, &d) in dists.iter().enumerate() {
        let obj = py_ulonglong(d);
        if obj.is_null() {
            ffi::Py_DecRef(list);
            return ptr::null_mut();
        }
        ffi::PyList_SET_ITEM(list, k as ffi::Py_ssize_t, obj);
    }
    list
}

type FastInner = unsafe extern "C" fn(
    *mut ffi::PyObject,
    *mut *mut ffi::PyObject,
    ffi::Py_ssize_t,
) -> *mut ffi::PyObject;

#[inline]
unsafe fn call_protected<F>(f: F) -> *mut ffi::PyObject
where
    F: FnOnce() -> *mut ffi::PyObject,
{
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(_) => {
            ffi::PyErr_SetString(
                ffi::PyExc_SystemError,
                b"Rust panic in hexhamming raw Python function\0".as_ptr() as *const c_char,
            );
            ptr::null_mut()
        }
    }
}

unsafe fn dispatch_keywords(
    slf: *mut ffi::PyObject,
    args: *const *mut ffi::PyObject,
    nargs: ffi::Py_ssize_t,
    kwnames: *mut ffi::PyObject,
    fn_name: &'static [u8],
    names: &[&'static [u8]],
    inner: FastInner,
) -> *mut ffi::PyObject {
    call_protected(|| {
        let mut parsed = [ptr::null_mut(); 3];
        if !parse_fastcall_keywords(
            fn_name,
            names,
            args,
            nargs,
            kwnames,
            &mut parsed[..names.len()],
        ) {
            return ptr::null_mut();
        }
        inner(slf, parsed.as_mut_ptr(), names.len() as ffi::Py_ssize_t)
    })
}

macro_rules! kw_wrapper {
    ($wrapper:ident, $inner:ident, $fn_name:literal, [$($arg:literal),+ $(,)?]) => {
        unsafe extern "C" fn $wrapper(
            slf: *mut ffi::PyObject,
            args: *const *mut ffi::PyObject,
            nargs: ffi::Py_ssize_t,
            kwnames: *mut ffi::PyObject,
        ) -> *mut ffi::PyObject {
            dispatch_keywords(
                slf,
                args,
                nargs,
                kwnames,
                concat!($fn_name, "\0").as_bytes(),
                &[$(concat!($arg, "\0").as_bytes()),+],
                $inner,
            )
        }
    };
}

kw_wrapper!(
    kw_hamming_distance_string,
    raw_hamming_distance_string_ascii,
    "hamming_distance_string",
    ["a", "b"]
);
kw_wrapper!(
    kw_hamming_distance_bytes,
    raw_hamming_distance_bytes,
    "hamming_distance_bytes",
    ["a", "b"]
);
kw_wrapper!(
    kw_check_hexstrings_within_dist,
    raw_check_hexstrings_within_dist_ascii,
    "check_hexstrings_within_dist",
    ["a", "b", "max_dist"]
);
kw_wrapper!(
    kw_check_bytes_within_dist,
    raw_check_bytes_within_dist,
    "check_bytes_within_dist",
    ["a", "b", "max_dist"]
);
kw_wrapper!(
    kw_check_bytes_arrays_within_dist,
    raw_check_bytes_arrays_within_dist_alias,
    "check_bytes_arrays_within_dist",
    ["array_of_elems", "elem_to_compare", "max_dist"]
);
kw_wrapper!(
    kw_check_bytes_arrays_first_within_dist,
    raw_check_bytes_arrays_first_within_dist,
    "check_bytes_arrays_first_within_dist",
    ["array_of_elems", "elem_to_compare", "max_dist"]
);
kw_wrapper!(
    kw_check_bytes_arrays_best_within_dist,
    raw_check_bytes_arrays_best_within_dist,
    "check_bytes_arrays_best_within_dist",
    ["array_of_elems", "elem_to_compare", "max_dist"]
);
kw_wrapper!(
    kw_check_bytes_arrays_all_within_dist,
    raw_check_bytes_arrays_all_within_dist,
    "check_bytes_arrays_all_within_dist",
    ["array_of_elems", "elem_to_compare", "max_dist"]
);
kw_wrapper!(
    kw_hamming_distances_bytes,
    raw_hamming_distances_bytes,
    "hamming_distances_bytes",
    ["a", "b", "element_size"]
);

type Fast = unsafe extern "C" fn(
    *mut ffi::PyObject,
    *const *mut ffi::PyObject,
    ffi::Py_ssize_t,
    *mut ffi::PyObject,
) -> *mut ffi::PyObject;

/// Add a raw METH_FASTCALL | METH_KEYWORDS function to the given module object.
///
/// The `PyMethodDef` is Box-leaked so it lives for the process lifetime; that
/// matches the storage used for statically-defined method tables in CPython.
unsafe fn add_raw(
    module: *mut ffi::PyObject,
    name: &'static [u8],
    f: Fast,
    doc: &'static [u8],
) -> c_int {
    let def = Box::leak(Box::new(ffi::PyMethodDef {
        ml_name: name.as_ptr() as *const c_char,
        ml_meth: ffi::PyMethodDefPointer {
            PyCFunctionFastWithKeywords: f,
        },
        ml_flags: ffi::METH_FASTCALL | ffi::METH_KEYWORDS,
        ml_doc: doc.as_ptr() as *const c_char,
    }));
    // The third argument becomes `__module__`, which must be the module's name
    // (as PyModule_AddFunctions does) for pickling and introspection.
    let module_name = ffi::PyModule_GetNameObject(module);
    if module_name.is_null() {
        return -1;
    }
    let obj = ffi::PyCFunction_NewEx(def, module, module_name);
    ffi::Py_DecRef(module_name);
    if obj.is_null() {
        return -1;
    }
    let name_no_nul_len = name.len() - 1;
    let attr_name = ffi::PyUnicode_FromStringAndSize(
        name.as_ptr() as *const c_char,
        name_no_nul_len as ffi::Py_ssize_t,
    );
    if attr_name.is_null() {
        ffi::Py_DecRef(obj);
        return -1;
    }
    let rc = ffi::PyObject_SetAttr(module, attr_name, obj);
    ffi::Py_DecRef(attr_name);
    ffi::Py_DecRef(obj);
    rc
}

/// Register every raw function in `module_obj`. On error, returns -1 with a
/// Python exception set.
pub(crate) unsafe fn register_all(module_obj: *mut ffi::PyObject) -> c_int {
    check_ascii_peek();
    // Prefer the ASCII fast path for str-taking functions; it falls back to
    // AsUTF8AndSize on Python versions/builds where the layout is unstable.
    macro_rules! add {
        ($name:literal, $f:ident, $doc:expr) => {
            if add_raw(module_obj, concat!($name, "\0").as_bytes(), $f, $doc) < 0 {
                return -1;
            }
        };
    }
    add!(
        "hamming_distance_string",
        kw_hamming_distance_string,
        b"hamming_distance_string($module, a, b)\n--\n\nCalculate the hamming distance of two hexadecimal strings.\n\nEquivalent to `bin(int(a, 16) ^ int(b, 16)).count('1')` but uses SIMD\nwhere available.\0"
    );
    add!(
        "hamming_distance_bytes",
        kw_hamming_distance_bytes,
        b"hamming_distance_bytes($module, a, b)\n--\n\nCalculate the hamming distance of two byte arrays.\n\nAccepts any buffer-protocol object: `bytes`, `bytearray`, `memoryview`,\nNumPy `uint8` arrays.\n\n**WARNING**: mutating a `bytearray` during computation is undefined\nbehavior.\0"
    );
    add!(
        "check_hexstrings_within_dist",
        kw_check_hexstrings_within_dist,
        b"check_hexstrings_within_dist($module, a, b, max_dist)\n--\n\nCheck if two hex strings are within a specified Hamming distance.\n\nFor `len >= 64` uses the SIMD path (full distance then compare).\nFor shorter strings uses scalar with early termination.\0"
    );
    add!(
        "check_bytes_within_dist",
        kw_check_bytes_within_dist,
        b"check_bytes_within_dist($module, a, b, max_dist)\n--\n\nCheck if two byte arrays are within a specified Hamming distance.\nReturns `True` if distance <= max_dist, `False` otherwise.\n\nAccepts any buffer-protocol object.\0"
    );
    add!(
        "check_bytes_arrays_within_dist",
        kw_check_bytes_arrays_within_dist,
        b"check_bytes_arrays_within_dist($module, array_of_elems, elem_to_compare, max_dist)\n--\n\nLegacy alias for `check_bytes_arrays_first_within_dist`.\0"
    );
    add!(
        "check_bytes_arrays_first_within_dist",
        kw_check_bytes_arrays_first_within_dist,
        b"check_bytes_arrays_first_within_dist($module, array_of_elems, elem_to_compare, max_dist)\n--\n\nReturn the index of the first element within a specified Hamming distance,\nor -1 if none found.\0"
    );
    add!(
        "check_bytes_arrays_best_within_dist",
        kw_check_bytes_arrays_best_within_dist,
        b"check_bytes_arrays_best_within_dist($module, array_of_elems, elem_to_compare, max_dist)\n--\n\nFind the element with the smallest Hamming distance.\nReturns `(best_distance, best_index)`, or `(-1, -1)` if none within\nmax_dist.\0"
    );
    add!(
        "check_bytes_arrays_all_within_dist",
        kw_check_bytes_arrays_all_within_dist,
        b"check_bytes_arrays_all_within_dist($module, array_of_elems, elem_to_compare, max_dist)\n--\n\nFind all elements within a specified Hamming distance.\nReturns list of `(distance, index)` tuples.\0"
    );
    add!(
        "hamming_distances_bytes",
        kw_hamming_distances_bytes,
        b"hamming_distances_bytes($module, a, b, element_size)\n--\n\nCompute Hamming distances between corresponding fixed-width records in `a`\nand `b`. Returns a list of `int` distances, one per record.\n\n`a` and `b` must be equal-length buffer-protocol objects whose length is a\nmultiple of `element_size`.\0"
    );
    0
}

// web-sys 0.3 deprecated `set_stroke_style(&JsValue)` in favour of
// `_str` / `_canvas_pattern` / `_gradient` variants, but the deprecated
// fns still work and the new ones are uglier for static strings. Quiet
// the warning until web-sys removes them.
#![allow(deprecated)]

//! `widget-tile-coding` — Chapter 8 tile-coding visualiser.
//!
//! Exports one function to JS: [`start`]. JS calls it with the id of a
//! DOM element; the widget mounts a 2-D canvas + a couple of sliders
//! and lets the reader drag a query point around the unit square,
//! watching the active tile of every overlapping tiling light up in a
//! distinct hue.
//!
//! Architecture: Rust owns the algorithm AND the DOM, same as the
//! bandit template. The widget reuses [`rl_core::TileCoder`] directly
//! — the indices the readout shows are the exact `(action, observation)`
//! indices the simulator's learner would see. The cell rectangles
//! shown on the canvas use the same offset formula
//! [`TileCoder::new`] uses internally
//! (`((t * (2d+1) / T) mod 1) * width`) so the rectangles match the
//! reported indices.
//!
//! This widget runs no learning loop. It's pure visualisation of the
//! coder's output for a single (x, y) point — the textbook's "tile
//! coding generalises across nearby states" claim made tangible.

use rl_core::{ActionTemplateId, Observation, TileCoder};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{
    CanvasRenderingContext2d, Document, Event, HtmlCanvasElement, HtmlElement, HtmlInputElement,
    MouseEvent,
};

/// Canvas dimensions in CSS pixels. The canvas's intrinsic resolution
/// matches; we let the browser scale via CSS for high-DPI without
/// re-doing the draw.
const CANVAS_PX: f64 = 400.0;

/// The fixed observation width — the (x, y) point in the unit square.
const OBS_WIDTH: usize = 2;

/// The IHT size we hand the coder. 4096 is plenty for a 2-D toy: at
/// `num_tilings = 16` and tile_width = 0.05` (the smallest the slider
/// allows) we'd touch ~16 × 400 = 6400 raw (t, cell) pairs across the
/// unit square, so collisions exist but are rare. The actual indices
/// the widget reports are `mod 4096`.
const IHT_SIZE: u32 = 1 << 12;

/// The single action we feed the coder. We only care about the
/// observation indices; action 0 just folds in once.
const ACTION: ActionTemplateId = ActionTemplateId(0);

/// JS entry point. `target` is the id of an existing `<div>` to mount
/// the widget into.
#[wasm_bindgen]
pub fn start(target: &str) -> Result<(), JsValue> {
    let window = web_sys::window().ok_or("no window")?;
    let document = window.document().ok_or("no document")?;
    let host = document
        .get_element_by_id(target)
        .ok_or_else(|| JsValue::from_str(&format!("no element #{target}")))?;
    host.set_inner_html(MARKUP);

    // Bind slider input events. Each redraws with the new parameters
    // but leaves the query point alone.
    bind_slider(&host, "[data-tc-num-tilings]", target, &document)?;
    bind_slider(&host, "[data-tc-tile-width]", target, &document)?;

    // Bind mouse-drag on the canvas. A shared `dragging` flag in JS
    // memory (an attribute on the canvas) lets the mousemove handler
    // know whether to update.
    let canvas_el = host
        .query_selector("[data-tc-canvas]")?
        .ok_or("missing canvas")?;
    let canvas: HtmlCanvasElement = canvas_el.dyn_into()?;

    bind_mouse(&canvas, "mousedown", target, &document, true, true)?;
    bind_mouse(&canvas, "mousemove", target, &document, false, false)?;
    bind_mouse(&canvas, "mouseup", target, &document, true, false)?;
    bind_mouse(&canvas, "mouseleave", target, &document, true, false)?;

    // Initial draw with the default query point (centre of the square).
    redraw(&document, target)?;
    Ok(())
}

/// Attach an `input` event listener that redraws when a slider moves.
fn bind_slider(
    host: &web_sys::Element,
    selector: &str,
    target: &str,
    document: &Document,
) -> Result<(), JsValue> {
    let el = host
        .query_selector(selector)?
        .ok_or_else(|| JsValue::from_str(&format!("missing {selector}")))?;
    let document_for_handler = document.clone();
    let target_for_handler = target.to_string();
    let cb = Closure::<dyn FnMut()>::new(move || {
        if let Err(e) = redraw(&document_for_handler, &target_for_handler) {
            web_sys::console::error_1(&e);
        }
    });
    el.add_event_listener_with_callback("input", cb.as_ref().unchecked_ref())?;
    cb.forget();
    Ok(())
}

/// Attach a mouse event listener that updates the query point.
///
/// - `event_name` is the DOM event ("mousedown", "mousemove", "mouseup",
///   "mouseleave").
/// - `toggles_drag` is `true` for events that flip the dragging flag
///   (down: true, up/leave: false). `false` for `mousemove`, which only
///   updates the query point while dragging.
/// - `start_drag` is the value to set the dragging flag to (only used
///   when `toggles_drag` is true).
fn bind_mouse(
    canvas: &HtmlCanvasElement,
    event_name: &str,
    target: &str,
    document: &Document,
    toggles_drag: bool,
    start_drag: bool,
) -> Result<(), JsValue> {
    let document_for_handler = document.clone();
    let target_for_handler = target.to_string();
    let canvas_for_handler = canvas.clone();
    let event_name_owned = event_name.to_string();
    let cb = Closure::<dyn FnMut(Event)>::new(move |event: Event| {
        let mouse_event = match event.dyn_into::<MouseEvent>() {
            Ok(m) => m,
            Err(_) => return,
        };
        let is_dragging = canvas_for_handler
            .get_attribute("data-tc-dragging")
            .map(|v| v == "1")
            .unwrap_or(false);
        if toggles_drag {
            canvas_for_handler
                .set_attribute(
                    "data-tc-dragging",
                    if start_drag { "1" } else { "0" },
                )
                .ok();
        }
        let should_update = if event_name_owned == "mousedown" {
            true
        } else if event_name_owned == "mousemove" {
            is_dragging
        } else {
            false
        };
        if should_update {
            let x = (mouse_event.offset_x() as f64) / CANVAS_PX;
            let y = (mouse_event.offset_y() as f64) / CANVAS_PX;
            let clamped_x = x.clamp(0.0, 1.0);
            let clamped_y = y.clamp(0.0, 1.0);
            canvas_for_handler
                .set_attribute("data-tc-x", &format!("{clamped_x:.6}"))
                .ok();
            canvas_for_handler
                .set_attribute("data-tc-y", &format!("{clamped_y:.6}"))
                .ok();
            if let Err(e) = redraw(&document_for_handler, &target_for_handler) {
                web_sys::console::error_1(&e);
            }
        }
    });
    canvas.add_event_listener_with_callback(event_name, cb.as_ref().unchecked_ref())?;
    cb.forget();
    Ok(())
}

/// Read parameters + the stored query point and redraw the canvas.
fn redraw(document: &Document, target: &str) -> Result<(), JsValue> {
    let host = document
        .get_element_by_id(target)
        .ok_or_else(|| JsValue::from_str(&format!("no element #{target}")))?;
    let num_tilings = read_number(&host, "[data-tc-num-tilings]")? as u32;
    let num_tilings = num_tilings.clamp(1, 16);
    let tile_width = read_number(&host, "[data-tc-tile-width]")?;
    let tile_width = tile_width.clamp(0.05, 0.5);

    let canvas_el = host.query_selector("[data-tc-canvas]")?.ok_or("missing canvas")?;
    let canvas: HtmlCanvasElement = canvas_el.dyn_into()?;
    let query_x = canvas
        .get_attribute("data-tc-x")
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.5);
    let query_y = canvas
        .get_attribute("data-tc-y")
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.5);

    let coder = TileCoder::uniform(num_tilings, tile_width, OBS_WIDTH, IHT_SIZE);
    let mut obs = Observation::zeros(OBS_WIDTH);
    {
        let slot = obs.as_mut_slice();
        slot[0] = query_x;
        slot[1] = query_y;
    }
    let prepared = coder.prepare(&obs);
    let indices: Vec<u32> = coder.active_indices(&prepared, ACTION).collect();

    // Update slider readouts.
    set_text(&host, "[data-tc-num-tilings-readout]", &format!("{num_tilings}"))?;
    set_text(&host, "[data-tc-tile-width-readout]", &format!("{tile_width:.2}"))?;

    draw(&canvas, num_tilings, tile_width, query_x, query_y)?;
    update_readout(&host, query_x, query_y, &indices)?;
    Ok(())
}

fn read_number(host: &web_sys::Element, selector: &str) -> Result<f32, JsValue> {
    let el = host
        .query_selector(selector)?
        .ok_or_else(|| JsValue::from_str(&format!("missing {selector}")))?;
    let input: HtmlInputElement = el.dyn_into()?;
    input
        .value()
        .parse()
        .map_err(|_| JsValue::from_str("invalid number"))
}

fn set_text(host: &web_sys::Element, selector: &str, value: &str) -> Result<(), JsValue> {
    let el = host
        .query_selector(selector)?
        .ok_or_else(|| JsValue::from_str(&format!("missing {selector}")))?;
    let html: HtmlElement = el.dyn_into()?;
    html.set_text_content(Some(value));
    Ok(())
}

/// The per-tiling, per-dimension offset, matching `TileCoder::new`'s
/// formula exactly:
/// ```text
/// offset[t][d] = ((t * (2d+1) / T) mod 1) * tile_width
/// ```
/// We duplicate the formula here (rather than expose offsets from the
/// coder) because the coder's `offsets` field is private — and the
/// formula is short enough that a re-derive is cleaner than a coder
/// API change.
fn tiling_offset(t: u32, d: u32, num_tilings: u32, tile_width: f32) -> f32 {
    let frac =
        ((t as f64) * ((2 * d + 1) as f64) / (num_tilings as f64)).fract();
    (frac as f32) * tile_width
}

fn draw(
    canvas: &HtmlCanvasElement,
    num_tilings: u32,
    tile_width: f32,
    query_x: f32,
    query_y: f32,
) -> Result<(), JsValue> {
    let ctx: CanvasRenderingContext2d = canvas
        .get_context("2d")?
        .ok_or("no 2d context")?
        .dyn_into()?;
    let w = canvas.width() as f64;
    let h = canvas.height() as f64;
    ctx.clear_rect(0.0, 0.0, w, h);

    // Background — unit square with a subtle grid for orientation.
    ctx.set_fill_style(&JsValue::from_str("#1c1c1c"));
    ctx.fill_rect(0.0, 0.0, w, h);
    ctx.set_stroke_style(&JsValue::from_str("#2c2c2c"));
    ctx.set_line_width(1.0);
    for i in 1..10 {
        let p = (i as f64) * w / 10.0;
        ctx.begin_path();
        ctx.move_to(p, 0.0);
        ctx.line_to(p, h);
        ctx.stroke();
        ctx.begin_path();
        ctx.move_to(0.0, p);
        ctx.line_to(w, p);
        ctx.stroke();
    }

    // For each tiling, draw the cell that contains the query point as
    // a translucent fill + a coloured outline.
    for t in 0..num_tilings {
        // Hue evenly spread across the tilings. HSL goes 0–360°.
        let hue = (t as f32) / (num_tilings as f32) * 360.0;
        let fill = format!("hsla({hue:.0}, 70%, 50%, 0.20)");
        let stroke = format!("hsla({hue:.0}, 70%, 60%, 0.85)");

        // Replicate `fold_block`'s coord formula per dimension so the
        // rectangle we draw is exactly the tile the coder identifies.
        let ox = tiling_offset(t, 0, num_tilings, tile_width);
        let oy = tiling_offset(t, 1, num_tilings, tile_width);
        let cell_x = ((query_x + ox) / tile_width).floor();
        let cell_y = ((query_y + oy) / tile_width).floor();
        // World-coordinate lower-left of the cell (may be < 0 because
        // the asymmetric offset can push tile 0's left edge into
        // negatives; the visible rectangle is clipped to the unit
        // square below).
        let x_lo = (cell_x * tile_width - ox) as f64;
        let y_lo = (cell_y * tile_width - oy) as f64;
        let x_hi = x_lo + tile_width as f64;
        let y_hi = y_lo + tile_width as f64;

        // Clip the rectangle to the unit square so it doesn't bleed
        // off the canvas. The coder's tile extends outside [0, 1]² in
        // principle — we just don't draw that part.
        let cx_lo = x_lo.max(0.0);
        let cy_lo = y_lo.max(0.0);
        let cx_hi = x_hi.min(1.0);
        let cy_hi = y_hi.min(1.0);
        if cx_hi <= cx_lo || cy_hi <= cy_lo {
            continue;
        }

        let px = cx_lo * w;
        let py = cy_lo * h;
        let pw = (cx_hi - cx_lo) * w;
        let ph = (cy_hi - cy_lo) * h;

        ctx.set_fill_style(&JsValue::from_str(&fill));
        ctx.fill_rect(px, py, pw, ph);
        ctx.set_stroke_style(&JsValue::from_str(&stroke));
        ctx.set_line_width(1.5);
        ctx.stroke_rect(px, py, pw, ph);
    }

    // The query point itself — a small white-ringed dot on top.
    let qx_px = (query_x as f64) * w;
    let qy_px = (query_y as f64) * h;
    ctx.set_fill_style(&JsValue::from_str("#ffffff"));
    ctx.begin_path();
    ctx.arc(qx_px, qy_px, 5.0, 0.0, std::f64::consts::TAU)?;
    ctx.fill();
    ctx.set_stroke_style(&JsValue::from_str("#111"));
    ctx.set_line_width(1.5);
    ctx.begin_path();
    ctx.arc(qx_px, qy_px, 5.0, 0.0, std::f64::consts::TAU)?;
    ctx.stroke();

    // Title strip.
    ctx.set_fill_style(&JsValue::from_str("#ddd"));
    ctx.set_font("12px sans-serif");
    ctx.fill_text("Drag the white dot. Each colour is one tiling's active tile.", 8.0, 14.0)
        .ok();

    Ok(())
}

fn update_readout(
    host: &web_sys::Element,
    query_x: f32,
    query_y: f32,
    indices: &[u32],
) -> Result<(), JsValue> {
    let readout = host
        .query_selector("[data-tc-readout]")?
        .ok_or("missing readout")?;
    let html: HtmlElement = readout.dyn_into()?;
    let mut indices_str = String::new();
    for (i, idx) in indices.iter().enumerate() {
        if i > 0 {
            indices_str.push_str(", ");
        }
        indices_str.push_str(&format!("{idx}"));
    }
    html.set_inner_html(&format!(
        "query point: <b>({query_x:.3}, {query_y:.3})</b> &nbsp;·&nbsp; \
         active tile indices ({n} total): <code>[{indices_str}]</code>",
        n = indices.len(),
    ));
    Ok(())
}

const MARKUP: &str = r#"
<div class="tc-widget">
  <div class="tc-controls">
    <label>num_tilings
      <input type="range" min="1" max="16" step="1" value="8" data-tc-num-tilings>
      <span data-tc-num-tilings-readout class="tc-value">8</span>
    </label>
    <label>tile_width
      <input type="range" min="0.05" max="0.5" step="0.01" value="0.25" data-tc-tile-width>
      <span data-tc-tile-width-readout class="tc-value">0.25</span>
    </label>
  </div>
  <canvas width="400" height="400"
          data-tc-canvas
          data-tc-x="0.5" data-tc-y="0.5"
          data-tc-dragging="0"></canvas>
  <div data-tc-readout class="tc-readout"></div>
</div>
<style>
.tc-widget { font-family: sans-serif; margin: 1em 0; }
.tc-controls { display: flex; flex-wrap: wrap; gap: 1.2em; align-items: center; margin-bottom: 0.5em; }
.tc-controls label { display: flex; align-items: center; gap: 0.4em; font-size: 0.9em; }
.tc-controls input[type=range] { width: 140px; }
.tc-controls .tc-value { display: inline-block; min-width: 2.5em; text-align: right; font-variant-numeric: tabular-nums; color: #ccc; }
.tc-widget canvas { display: block; width: 400px; max-width: 100%; height: auto; background: #1c1c1c; border-radius: 4px; cursor: crosshair; }
.tc-readout { margin-top: 0.5em; font-size: 0.85em; color: #ccc; word-break: break-word; }
.tc-readout code { font-size: 0.85em; color: #9cdcfe; }
</style>
"#;

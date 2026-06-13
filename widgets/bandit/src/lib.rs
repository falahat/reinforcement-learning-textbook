// web-sys 0.3 deprecated `set_stroke_style(&JsValue)` in favour of
// `_str` / `_canvas_pattern` / `_gradient` variants, but the deprecated
// fns still work and the new ones are uglier for static strings. Quiet
// the warning until web-sys removes them.
#![allow(deprecated)]

//! `widget-bandit` — Chapter 12 ε-greedy bandit visualiser.
//!
//! Exports one function to JS: [`start`]. JS calls it with the id of a
//! DOM element; the widget mounts a small UI (sliders for ε, α, and the
//! arm means + a "Run episode" button + a `<canvas>`) inside that
//! element and drives the canonical
//! [`playground::run_episode`](::playground::run_episode) under the hood.
//!
//! Architecture: Rust owns the algorithm (`run_episode` over a
//! `MultiArmedBandit` + `TileCodedValue`), and the DOM. JS only loads
//! the WASM and triggers `start`. No JS-side ε-greedy reimplementation
//! — the same code the production simulator uses also drives the
//! widget.

use playground::{run_episode, EpisodeConfig, MultiArmedBandit};
use rl_core::{ActionTemplateId, TileCoder, TileCodedValue};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{
    CanvasRenderingContext2d, Document, HtmlCanvasElement, HtmlElement, HtmlInputElement,
};

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

    // Bind the run button to a fresh episode.
    let document_for_handler = document.clone();
    let target_for_handler = target.to_string();
    let run_button = host
        .query_selector("[data-bandit-run]")?
        .ok_or("missing run button")?;
    let click = Closure::<dyn FnMut()>::new(move || {
        if let Err(e) = run_one(&document_for_handler, &target_for_handler) {
            web_sys::console::error_1(&e);
        }
    });
    run_button
        .add_event_listener_with_callback("click", click.as_ref().unchecked_ref())?;
    click.forget();

    // Run once on load so the user sees a populated chart immediately.
    run_one(&document, target)?;
    Ok(())
}

/// Read the current slider state, run one episode, draw the result.
fn run_one(document: &Document, target: &str) -> Result<(), JsValue> {
    let host = document
        .get_element_by_id(target)
        .ok_or_else(|| JsValue::from_str(&format!("no element #{target}")))?;
    let epsilon = read_number(&host, "[data-bandit-epsilon]")?;
    let alpha = read_number(&host, "[data-bandit-alpha]")?;
    let steps = read_number(&host, "[data-bandit-steps]")? as usize;
    let arm_means = vec![0.10, 0.30, 0.50, 0.70, 0.90];

    let mut env = MultiArmedBandit::new(arm_means.clone(), 0.05);
    let mut value = TileCodedValue::new(TileCoder::uniform(8, 0.25, 1, 1 << 14));
    let cfg = EpisodeConfig {
        max_steps: steps,
        epsilon,
        alpha,
        gamma: 0.0, // bandit has no successor state
        seed: 0,
    };
    let stats = run_episode(&mut env, &mut value, |a| ActionTemplateId(a as u32), &cfg);

    // Per-block best-arm-pull fraction — the canonical bandit learning
    // curve from Sutton & Barto Ch 2.
    let best_arm = env.best_arm();
    let block_size = 50usize.max(steps / 40);
    let mut curve = Vec::new();
    for block in stats.action_history.chunks(block_size) {
        let best = block.iter().filter(|&&a| a == best_arm).count() as f32;
        curve.push(best / block.len() as f32);
    }

    draw_curve(&host, &curve, best_arm)?;
    update_readout(&host, stats.total_reward, &curve, best_arm)?;
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

fn draw_curve(host: &web_sys::Element, curve: &[f32], best_arm: usize) -> Result<(), JsValue> {
    let canvas_el = host
        .query_selector("[data-bandit-canvas]")?
        .ok_or("missing canvas")?;
    let canvas: HtmlCanvasElement = canvas_el.dyn_into()?;
    let ctx: CanvasRenderingContext2d = canvas
        .get_context("2d")?
        .ok_or("no 2d context")?
        .dyn_into()?;
    let w = canvas.width() as f64;
    let h = canvas.height() as f64;
    ctx.clear_rect(0.0, 0.0, w, h);

    // Background grid.
    ctx.set_stroke_style(&JsValue::from_str("#444"));
    ctx.set_line_width(1.0);
    for i in 0..=4 {
        let y = h - (i as f64) * (h / 4.0);
        ctx.begin_path();
        ctx.move_to(0.0, y);
        ctx.line_to(w, y);
        ctx.stroke();
    }
    // Axis labels.
    ctx.set_fill_style(&JsValue::from_str("#aaa"));
    ctx.set_font("11px sans-serif");
    for i in 0..=4 {
        let y = h - (i as f64) * (h / 4.0);
        let label = format!("{:.2}", i as f32 / 4.0);
        ctx.fill_text(&label, 4.0, y - 2.0).ok();
    }
    // Reference line at random-pick fraction (1/K).
    let random_frac = 1.0 / 5.0; // K=5 arms
    let y_random = h - (random_frac as f64) * h;
    ctx.set_stroke_style(&JsValue::from_str("#888"));
    ctx.set_line_width(1.0);
    ctx.begin_path();
    ctx.move_to(0.0, y_random);
    ctx.line_to(w, y_random);
    ctx.stroke();

    // Learning curve.
    if curve.is_empty() {
        return Ok(());
    }
    ctx.set_stroke_style(&JsValue::from_str("#4caf50"));
    ctx.set_line_width(2.0);
    ctx.begin_path();
    for (i, &frac) in curve.iter().enumerate() {
        let x = (i as f64) * (w / (curve.len().max(1) as f64));
        let y = h - (frac as f64) * h;
        if i == 0 {
            ctx.move_to(x, y);
        } else {
            ctx.line_to(x, y);
        }
    }
    ctx.stroke();

    // Title text.
    ctx.set_fill_style(&JsValue::from_str("#ddd"));
    ctx.set_font("12px sans-serif");
    ctx.fill_text(
        &format!("Pulls of best arm (#{best_arm}) per block, target = 1 − ε"),
        50.0,
        14.0,
    )
    .ok();
    Ok(())
}

fn update_readout(
    host: &web_sys::Element,
    total_reward: f32,
    curve: &[f32],
    best_arm: usize,
) -> Result<(), JsValue> {
    let final_frac = curve.last().copied().unwrap_or(0.0);
    let initial_frac = curve.first().copied().unwrap_or(0.0);
    let readout = host
        .query_selector("[data-bandit-readout]")?
        .ok_or("missing readout")?;
    let html: HtmlElement = readout.dyn_into()?;
    html.set_inner_html(&format!(
        "best arm: <b>#{best_arm}</b> &nbsp;·&nbsp; \
         initial best-arm fraction: <b>{:.0}%</b> &nbsp;·&nbsp; \
         final: <b>{:.0}%</b> &nbsp;·&nbsp; \
         total reward: <b>{:.1}</b>",
        initial_frac * 100.0,
        final_frac * 100.0,
        total_reward,
    ));
    Ok(())
}

const MARKUP: &str = r#"
<div class="bandit-widget">
  <div class="bandit-controls">
    <label>ε <input type="range" min="0" max="1" step="0.01" value="0.10" data-bandit-epsilon></label>
    <label>α <input type="range" min="0.01" max="0.5" step="0.01" value="0.10" data-bandit-alpha></label>
    <label>steps <input type="number" min="100" max="5000" step="100" value="2000" data-bandit-steps></label>
    <button data-bandit-run>Run episode</button>
  </div>
  <canvas width="640" height="200" data-bandit-canvas></canvas>
  <div data-bandit-readout class="bandit-readout"></div>
</div>
<style>
.bandit-widget { font-family: sans-serif; margin: 1em 0; }
.bandit-controls { display: flex; flex-wrap: wrap; gap: 0.8em; align-items: center; margin-bottom: 0.5em; }
.bandit-controls label { display: flex; align-items: center; gap: 0.3em; font-size: 0.9em; }
.bandit-controls input[type=range] { width: 100px; }
.bandit-controls input[type=number] { width: 80px; }
.bandit-controls button { padding: 0.3em 0.8em; cursor: pointer; }
.bandit-widget canvas { display: block; width: 100%; max-width: 640px; background: #222; border-radius: 4px; }
.bandit-readout { margin-top: 0.5em; font-size: 0.9em; color: #ccc; }
</style>
"#;

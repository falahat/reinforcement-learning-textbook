// web-sys 0.3 deprecated `set_stroke_style(&JsValue)` in favour of
// `_str` / `_canvas_pattern` / `_gradient` variants, but the deprecated
// fns still work and the new ones are uglier for static strings. Quiet
// the warning until web-sys removes them.
#![allow(deprecated)]

//! `widget-gridworld` — Chapter 6 Q-learning on a 5×5 GridWorld.
//!
//! Exports one function to JS: [`start`]. JS calls it with the id of a
//! DOM element; the widget mounts a small UI (sliders for ε, α, γ, and
//! episode count + a "Train" button + a `<canvas>`) inside that element
//! and drives the canonical
//! [`playground::run_episode`](::playground::run_episode) under the hood,
//! once per episode, accumulating Q-values across episodes in a single
//! shared [`TileCodedValue`].
//!
//! After training, the canvas renders the 5×5 grid as a heatmap of
//! `max_a Q(s, a)` — cells the learner believes lead to the goal show
//! bright, untouched cells stay dim. Greedy-policy arrows overlay each
//! visited cell.
//!
//! Architecture: Rust owns the algorithm (`run_episode` over a
//! `GridWorld` + `TileCodedValue`) and the DOM. JS only loads the WASM
//! and triggers `start`. No JS-side Q-learning reimplementation — the
//! same code the focused gridworld test runs also drives the widget.

use playground::{run_episode, EpisodeConfig, GridAction, GridWorld};
use rl_core::{ActionTemplateId, Observation, TileCoder, TileCodedValue, ValueFunction};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{
    CanvasRenderingContext2d, Document, HtmlCanvasElement, HtmlElement, HtmlInputElement,
};

/// Grid side length. The proposal asks for a 5×5 gridworld with the
/// goal in the opposite corner.
const GRID: u32 = 5;

/// Map the four cardinal `GridAction`s to flat `ActionTemplateId`s the
/// tile-coded value function indexes weights by. Identical to the
/// mapping used in the `gridworld_learner_eventually_reaches_the_goal`
/// focused test.
fn action_key(a: GridAction) -> ActionTemplateId {
    match a {
        GridAction::North => ActionTemplateId(0),
        GridAction::South => ActionTemplateId(1),
        GridAction::East => ActionTemplateId(2),
        GridAction::West => ActionTemplateId(3),
    }
}

/// All four actions, in the same order the env's `action_space()`
/// returns them — used when sweeping over actions to build the greedy
/// argmax / max-Q at every cell for the heatmap.
const ACTIONS: [GridAction; 4] = [
    GridAction::North,
    GridAction::South,
    GridAction::East,
    GridAction::West,
];

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

    // Bind the train button to a fresh training run.
    let document_for_handler = document.clone();
    let target_for_handler = target.to_string();
    let train_button = host
        .query_selector("[data-gridworld-train]")?
        .ok_or("missing train button")?;
    let click = Closure::<dyn FnMut()>::new(move || {
        if let Err(e) = train(&document_for_handler, &target_for_handler) {
            web_sys::console::error_1(&e);
        }
    });
    train_button
        .add_event_listener_with_callback("click", click.as_ref().unchecked_ref())?;
    click.forget();

    // Run one training pass on load so the user sees a populated heatmap
    // immediately — same UX as the bandit widget.
    train(&document, target)?;
    Ok(())
}

/// Read the current slider state, run N episodes against a freshly-init
/// `TileCodedValue`, draw the resulting heatmap + arrow overlay, and
/// update the readout div.
fn train(document: &Document, target: &str) -> Result<(), JsValue> {
    let host = document
        .get_element_by_id(target)
        .ok_or_else(|| JsValue::from_str(&format!("no element #{target}")))?;
    let epsilon = read_number(&host, "[data-gridworld-epsilon]")?;
    let alpha = read_number(&host, "[data-gridworld-alpha]")?;
    let gamma = read_number(&host, "[data-gridworld-gamma]")?;
    let episodes = read_number(&host, "[data-gridworld-episodes]")? as usize;

    // The width-2 observation `GridWorld` emits is `(x/(W-1), y/(H-1))`
    // in [0, 1]. Tile width 0.25 with 8 tilings is the same generaliser
    // the focused gridworld test uses — small enough that the 5×5 cells
    // get distinct tile activations, big enough that nearby cells share
    // weight.
    let mut value = TileCodedValue::new(TileCoder::uniform(8, 0.25, 2, 1 << 14));
    let mut steps_per_ep: Vec<usize> = Vec::with_capacity(episodes);
    let mut total_reward = 0.0_f32;
    for ep in 0..episodes {
        let mut env = GridWorld::new(GRID, GRID, (0, 0), (GRID - 1, GRID - 1));
        let cfg = EpisodeConfig {
            // 500 steps is enough room for a random-walking first episode
            // on a 5×5 grid to wander into the goal eventually; later
            // episodes converge to <20.
            max_steps: 500,
            epsilon,
            alpha,
            gamma,
            seed: ep as u64,
        };
        let stats = run_episode(&mut env, &mut value, action_key, &cfg);
        steps_per_ep.push(stats.action_history.len());
        total_reward += stats.total_reward;
    }

    draw_heatmap(&host, &value)?;
    update_readout(&host, &steps_per_ep, total_reward)?;
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

/// Build the same width-2 observation `GridWorld::step` would emit if
/// the agent were at `(x, y)`. Lets us query `value.value(obs, a)` at
/// every cell for the heatmap without stepping the env.
fn obs_at(x: u32, y: u32) -> Observation {
    let mut o = Observation::zeros(2);
    let v = o.as_mut_slice();
    let denom = (GRID - 1).max(1) as f32;
    v[0] = x as f32 / denom;
    v[1] = y as f32 / denom;
    o
}

/// `max_a Q((x, y), a)` over the four cardinal actions. The same max
/// the Q-learning update bootstraps off, evaluated post-hoc for
/// visualisation.
fn cell_value(value: &TileCodedValue, x: u32, y: u32) -> f32 {
    let obs = obs_at(x, y);
    ACTIONS
        .iter()
        .map(|&a| value.value(&obs, action_key(a)))
        .fold(f32::NEG_INFINITY, f32::max)
}

/// `argmax_a Q((x, y), a)` — the greedy action at `(x, y)`. Drawn as
/// an arrow over each cell so the reader can see the learned policy.
fn cell_argmax(value: &TileCodedValue, x: u32, y: u32) -> GridAction {
    let obs = obs_at(x, y);
    let mut best = GridAction::North;
    let mut best_v = f32::NEG_INFINITY;
    for &a in &ACTIONS {
        let v = value.value(&obs, action_key(a));
        if v > best_v {
            best_v = v;
            best = a;
        }
    }
    best
}

fn draw_heatmap(host: &web_sys::Element, value: &TileCodedValue) -> Result<(), JsValue> {
    let canvas_el = host
        .query_selector("[data-gridworld-canvas]")?
        .ok_or("missing canvas")?;
    let canvas: HtmlCanvasElement = canvas_el.dyn_into()?;
    let ctx: CanvasRenderingContext2d = canvas
        .get_context("2d")?
        .ok_or("no 2d context")?
        .dyn_into()?;
    let w = canvas.width() as f64;
    let h = canvas.height() as f64;
    ctx.clear_rect(0.0, 0.0, w, h);

    // Pre-compute per-cell max-Q. Normalise to [0, 1] for colouring.
    // Cells that never got any positive weight (untouched corners with
    // negative tile sums) clamp at 0 — they stay dark.
    let mut grid_vals = [[0.0_f32; GRID as usize]; GRID as usize];
    let mut max_seen = f32::NEG_INFINITY;
    let mut min_seen = f32::INFINITY;
    for y in 0..GRID {
        for x in 0..GRID {
            let v = cell_value(value, x, y).max(0.0);
            grid_vals[y as usize][x as usize] = v;
            if v > max_seen {
                max_seen = v;
            }
            if v < min_seen {
                min_seen = v;
            }
        }
    }
    let range = (max_seen - min_seen).max(1.0e-6);

    let cell_w = w / GRID as f64;
    let cell_h = h / GRID as f64;
    for y in 0..GRID {
        for x in 0..GRID {
            let v = grid_vals[y as usize][x as usize];
            let t = ((v - min_seen) / range).clamp(0.0, 1.0);
            // Plasma-ish ramp: dark navy → orange → pale yellow as t↑.
            // Pure colour interpolation in sRGB; good enough for a small
            // 5×5 grid where exact perceptual uniformity doesn't matter.
            let r = (30.0 + 220.0 * t) as u8;
            let g = (20.0 + 180.0 * t.powf(1.5)) as u8;
            let b = (60.0 + 60.0 * (1.0 - t)) as u8;
            let fill = format!("#{:02x}{:02x}{:02x}", r, g, b);
            ctx.set_fill_style(&JsValue::from_str(&fill));
            // Canvas y grows downward; the GridWorld y axis grows upward
            // (north = +y). Flip so y=0 (start) renders at the bottom.
            let canvas_y = h - (y as f64 + 1.0) * cell_h;
            ctx.fill_rect(x as f64 * cell_w, canvas_y, cell_w, cell_h);

            // Numeric Q-value, small.
            ctx.set_fill_style(&JsValue::from_str("#fff"));
            ctx.set_font("11px sans-serif");
            ctx.fill_text(
                &format!("{:.2}", v),
                x as f64 * cell_w + 4.0,
                canvas_y + 14.0,
            )
            .ok();

            // Greedy-action arrow centred in the cell (overlay).
            if v > 0.001 {
                let cx = x as f64 * cell_w + cell_w / 2.0;
                let cy = canvas_y + cell_h / 2.0;
                let arrow_len = cell_w.min(cell_h) * 0.25;
                let (dx, dy) = match cell_argmax(value, x, y) {
                    // North = +y in env space = upward in canvas space
                    // post-flip = -dy on canvas.
                    GridAction::North => (0.0, -arrow_len),
                    GridAction::South => (0.0, arrow_len),
                    GridAction::East => (arrow_len, 0.0),
                    GridAction::West => (-arrow_len, 0.0),
                };
                ctx.set_stroke_style(&JsValue::from_str("#fff"));
                ctx.set_line_width(2.0);
                ctx.begin_path();
                ctx.move_to(cx - dx, cy - dy);
                ctx.line_to(cx + dx, cy + dy);
                ctx.stroke();
                // Arrowhead — two short strokes back from the tip.
                let head = arrow_len * 0.5;
                let (hx1, hy1, hx2, hy2) = if dx.abs() > dy.abs() {
                    (-dx.signum() * head, head, -dx.signum() * head, -head)
                } else {
                    (head, -dy.signum() * head, -head, -dy.signum() * head)
                };
                ctx.begin_path();
                ctx.move_to(cx + dx, cy + dy);
                ctx.line_to(cx + dx + hx1, cy + dy + hy1);
                ctx.move_to(cx + dx, cy + dy);
                ctx.line_to(cx + dx + hx2, cy + dy + hy2);
                ctx.stroke();
            }
        }
    }

    // Grid lines.
    ctx.set_stroke_style(&JsValue::from_str("#000"));
    ctx.set_line_width(1.0);
    for i in 0..=GRID {
        let x = (i as f64) * cell_w;
        ctx.begin_path();
        ctx.move_to(x, 0.0);
        ctx.line_to(x, h);
        ctx.stroke();
        let y = (i as f64) * cell_h;
        ctx.begin_path();
        ctx.move_to(0.0, y);
        ctx.line_to(w, y);
        ctx.stroke();
    }

    // Label start (bottom-left after flip) and goal (top-right).
    ctx.set_fill_style(&JsValue::from_str("#fff"));
    ctx.set_font("bold 12px sans-serif");
    ctx.fill_text("S", 4.0, h - 4.0).ok();
    ctx.fill_text("G", w - 14.0, 14.0).ok();
    Ok(())
}

fn update_readout(
    host: &web_sys::Element,
    steps_per_ep: &[usize],
    total_reward: f32,
) -> Result<(), JsValue> {
    // Convergence heuristic: first episode whose step count is within
    // 1.5× of the average of the last 5. Reports "—" if it never
    // converges within the training window.
    let n = steps_per_ep.len();
    let last_window = 5.min(n);
    let last_avg = if last_window > 0 {
        steps_per_ep[n - last_window..]
            .iter()
            .map(|&s| s as f32)
            .sum::<f32>()
            / last_window as f32
    } else {
        0.0
    };
    let converge_threshold = last_avg * 1.5;
    let converged_at = steps_per_ep
        .iter()
        .position(|&s| (s as f32) <= converge_threshold)
        .map(|i| i + 1);
    let first_steps = steps_per_ep.first().copied().unwrap_or(0);

    let readout = host
        .query_selector("[data-gridworld-readout]")?
        .ok_or("missing readout")?;
    let html: HtmlElement = readout.dyn_into()?;
    let converge_str = match converged_at {
        Some(i) => format!("episode <b>{i}</b>"),
        None => "<b>—</b>".to_string(),
    };
    html.set_inner_html(&format!(
        "first episode steps: <b>{first_steps}</b> &nbsp;·&nbsp; \
         last-5 avg steps: <b>{:.1}</b> &nbsp;·&nbsp; \
         converged at: {converge_str} &nbsp;·&nbsp; \
         total reward: <b>{:.1}</b>",
        last_avg, total_reward,
    ));
    Ok(())
}

const MARKUP: &str = r#"
<div class="gridworld-widget">
  <div class="gridworld-controls">
    <label>ε <input type="range" min="0.01" max="0.5" step="0.01" value="0.20" data-gridworld-epsilon></label>
    <label>α <input type="range" min="0.05" max="0.5" step="0.01" value="0.20" data-gridworld-alpha></label>
    <label>γ <input type="range" min="0.80" max="0.99" step="0.01" value="0.95" data-gridworld-gamma></label>
    <label>episodes <input type="number" min="10" max="100" step="5" value="30" data-gridworld-episodes></label>
    <button data-gridworld-train>Train</button>
  </div>
  <canvas width="320" height="320" data-gridworld-canvas></canvas>
  <div data-gridworld-readout class="gridworld-readout"></div>
</div>
<style>
.gridworld-widget { font-family: sans-serif; margin: 1em 0; }
.gridworld-controls { display: flex; flex-wrap: wrap; gap: 0.8em; align-items: center; margin-bottom: 0.5em; }
.gridworld-controls label { display: flex; align-items: center; gap: 0.3em; font-size: 0.9em; }
.gridworld-controls input[type=range] { width: 100px; }
.gridworld-controls input[type=number] { width: 80px; }
.gridworld-controls button { padding: 0.3em 0.8em; cursor: pointer; }
.gridworld-widget canvas { display: block; width: 100%; max-width: 320px; background: #222; border-radius: 4px; image-rendering: pixelated; }
.gridworld-readout { margin-top: 0.5em; font-size: 0.9em; color: #ccc; }
</style>
"#;

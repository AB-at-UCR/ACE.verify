import random
import numpy as np

def generate_timeline(duration_in_sec=30, n_segs=60):
    base = np.random.beta(2, 5, n_segs)
    spike_idxs = random.sample(range(n_segs), k=min(8, n_segs // 4))
    for i in spike_idxs:
        base[i] = random.uniform(0.6, 0.98)
    return base.tolist()

def score_to_color(score):
    if score > 0.65:
        return "rgb(232,64,64)"
    elif score > 0.35:
        return "rgb(240,160,48)"
    else:
        return "rgb(46,204,113)"

def render_timeline_html(scores, duration_in_sec):
    n = len(scores)
    bars = ""
    for i, s in enumerate(scores):
        h = max(8, int(s * 56))
        color = score_to_color(s)
        tooltip = f"t={i*duration_in_sec//n}s  score={s:.2f}"
        bars += f'<div class="timeline-segment" style="height:{h}px;background:{color};opacity:0.85;" title="{tooltip}"></div>'

    ticks = ""
    tick_count = 7
    for i in range(tick_count):
        t = int(i * duration_in_sec / (tick_count - 1))
        ticks += f"<span>{t}s</span>"

    return f"""
    <div class="timeline-wrap">
      <h4>🕐 Temporal Fakeness Timeline</h4>
      <div class="timeline-bar-container">{bars}</div>
      <div class="timeline-ticks">{ticks}</div>
      <div class="timeline-legend">
        <span><span class="legend-dot" style="background:#E84040"></span>High risk (&gt;65%)</span>
        <span><span class="legend-dot" style="background:#F0A030"></span>Uncertain (35-65%)</span>
        <span><span class="legend-dot" style="background:#2ECC71"></span>Authentic (&lt;35%)</span>
      </div>
    </div>
    """
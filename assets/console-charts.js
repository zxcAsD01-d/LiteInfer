(function (root) {
  "use strict";
  const U = root.LiteInferUI;
  const modes = {
    requests: { left: "请求数", right: "平均耗时", count: (p) => p.requests, secondary: (p) => p.avg_duration_ms,
      leftFormat: (n) => U.number(n), rightFormat: (n) => U.number(n / 1000, 1) + " s", floor: 4 },
    output: { left: "输出吞吐", right: "出字速度", count: (p) => p.output_tokens_per_second,
      secondary: (p) => p.avg_output_tokens_per_second, leftFormat: speed, rightFormat: speed, floor: .01 },
  };
  function speed(n) { return n == null ? "—" : U.number(n, n < 1 ? 2 : 1); }
  function sparkline(target, values, blue = false) {
    target.replaceChildren();
    if (!values.some((v) => v > 0)) return;
    target.classList.toggle("blue-spark", blue);
    const maximum = Math.max(...values, 1);
    const points = values.map((v, i) => `${i / Math.max(values.length - 1, 1) * 72},${27 - v / maximum * 23}`);
    const svg = U.svg("svg", { viewBox: "0 0 72 29", "aria-hidden": "true" });
    svg.append(U.svg("path", { d: `M${points.join(" L")}`, class: "spark-path" })); target.append(svg);
  }
  function label(svg, text, attrs) {
    const item = U.svg("text", { class: "chart-text", ...attrs }); item.textContent = text; svg.append(item);
  }
  function chartFrame(target, mode) {
    const width = Math.max(300, target.clientWidth);
    const height = target.clientHeight || 237;
    const svg = U.svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img",
      "aria-label": mode.left + "与" + mode.right + "趋势" });
    const defs = U.svg("defs");
    const gradient = U.svg("linearGradient", { id: "traffic-gradient", x1: 0, y1: 0, x2: 0, y2: 1 });
    gradient.append(U.svg("stop", { offset: "0%", "stop-color": "#64b59d", "stop-opacity": .19 }),
      U.svg("stop", { offset: "100%", "stop-color": "#64b59d", "stop-opacity": .01 }));
    defs.append(gradient); svg.append(defs);
    return { svg, width, height, left: 37, right: 41, top: 23, bottom: 32 };
  }
  function axes(frame, report, mode) {
    const { svg, left, top, width, height, right, bottom } = frame;
    const w = width - left - right, h = height - top - bottom;
    const max = Math.max(mode.floor, ...report.series.map(mode.count));
    const maxMs = Math.max(1, ...report.series.map((p) => mode.secondary(p) || 0));
    for (let i = 0; i <= 4; i++) {
      const y = top + h * i / 4;
      svg.append(U.svg("line", { x1: left, x2: width - right, y1: y, y2: y, class: "chart-grid" }));
      label(svg, mode.leftFormat(max * (1 - i / 4)), { x: left - 10, y: y + 3, "text-anchor": "end" });
      label(svg, mode.rightFormat(maxMs * (1 - i / 4)), { x: width - right + 10, y: y + 3 });
    }
    for (const index of [0, 6, 12, 18, 24, 29]) {
      const x = left + w * index / 29;
      label(svg, U.clock(report.series[index].start).slice(0, 5), { x, y: height - 9, "text-anchor": "middle" });
    }
    label(svg, mode.left, { x: left, y: 10 });
    label(svg, mode.right, { x: width - right, y: 10, "text-anchor": "end" });
    return { w, h, max, maxMs };
  }
  function paths(frame, dimensions, report, mode) {
    const { svg, left, top } = frame, { w, h, max, maxMs } = dimensions;
    const points = report.series.map((p, i) => [left + w * i / 29, top + h - mode.count(p) / max * h]);
    const line = points.map(([x, y]) => `${x},${y}`).join(" L");
    svg.append(U.svg("path", { d: `M${left},${top + h} L${line} L${left + w},${top + h} Z`, class: "chart-area" }));
    svg.append(U.svg("path", { d: `M${line}`, class: "chart-volume" }));
    let latency = "";
    report.series.forEach((p, i) => {
      const value = mode.secondary(p);
      if (value == null) { latency += " "; return; }
      const previous = i && mode.secondary(report.series[i - 1]);
      const x = left + w * i / 29, y = top + h - value / maxMs * h;
      latency += (i && previous != null ? "L" : "M") + x + "," + y + " ";
      svg.append(U.svg("circle", { cx: x, cy: y, r: 2, class: "chart-secondary-point" }));
    });
    svg.append(U.svg("path", { d: latency.trim(), class: "chart-duration" }));
    return points;
  }
  function tooltips(frame, dimensions, report, points, mode) {
    const { svg, left, top, width, height } = frame, { w, h } = dimensions;
    const popup = U.svg("g", { visibility: "hidden", "pointer-events": "none" });
    const dot = U.svg("circle", { r: 4, class: "chart-focus" });
    const background = U.svg("rect", { width: 159, height: 47, rx: 6, class: "chart-tooltip" });
    const title = U.svg("text", { class: "chart-tooltip-text" });
    const detail = U.svg("text", { class: "chart-tooltip-text" });
    popup.append(dot, background, title, detail);
    report.series.forEach((p, i) => {
      const hit = U.svg("rect", { x: left + w * (i - .5) / 29, y: top, width: w / 29 + 1, height: h, class: "chart-hit" });
      hit.addEventListener("pointerenter", () => {
        const [x, y] = points[i], px = Math.max(2, Math.min(width - 161, x - 79)), py = Math.max(15, Math.min(height - 80, y - 59));
        dot.setAttribute("cx", x); dot.setAttribute("cy", y);
        background.setAttribute("x", px); background.setAttribute("y", py);
        title.setAttribute("x", px + 10); title.setAttribute("y", py + 18); title.textContent = U.clock(p.start) + " · " + mode.leftFormat(mode.count(p));
        detail.setAttribute("x", px + 10); detail.setAttribute("y", py + 35); detail.textContent = mode.right + " " + (mode.secondary(p) == null ? "—" : mode.rightFormat(mode.secondary(p)));
        popup.setAttribute("visibility", "visible");
      });
      svg.append(hit);
    });
    svg.append(popup); svg.addEventListener("pointerleave", () => popup.setAttribute("visibility", "hidden"));
  }
  function traffic(report) {
    const target = U.el("traffic-chart");
    const output = U.el("chart-mode").value === "output";
    const mode = output ? modes.output : modes.requests;
    U.text("volume-legend", output ? "输出吞吐" : "请求量");
    U.text("secondary-legend", output ? "出字速度" : "平均耗时");
    U.text("chart-unit", output ? "单位：tok/s" : "端到端耗时");
    if (!report?.series?.length || !report.summary.requests) {
      const empty = U.node("div", "empty-chart"); empty.append(U.icon("pulse"), U.node("strong", "", "所选时段暂无请求"),
        U.node("span", "", "开始调用推理 API，或切换更长的统计时段")); target.replaceChildren(empty); return;
    }
    const frame = chartFrame(target, mode), dimensions = axes(frame, report, mode);
    const points = paths(frame, dimensions, report, mode); tooltips(frame, dimensions, report, points, mode);
    target.replaceChildren(frame.svg);
  }
  root.LiteInferCharts = Object.freeze({ sparkline, traffic });
})(window);

import { useState, useMemo, useEffect, useRef } from "react";

// Generate concentric circles dataset
function generateData(n = 80) {
  const points = [];
  for (let i = 0; i < n; i++) {
    const label = i < n / 2 ? 0 : 1;
    const r = label === 0 ? 0.3 + Math.random() * 0.4 : 1.0 + Math.random() * 0.5;
    const angle = Math.random() * Math.PI * 2;
    const x = r * Math.cos(angle) + (Math.random() - 0.5) * 0.15;
    const y = r * Math.sin(angle) + (Math.random() - 0.5) * 0.15;
    points.push({ x, y, label });
  }
  return points;
}

// ReLU activation
const relu = (x) => Math.max(0, x);
// Sigmoid
const sigmoid = (x) => 1 / (1 + Math.exp(-x));

// Pre-trained weights that nicely separate concentric circles
const W1 = [
  [2.5, 0.8],
  [0.6, 2.8],
  [-1.8, 1.5],
];
const b1 = [-1.2, -1.0, 0.3];

const W2 = [
  [1.8, -1.2, 2.0],
  [-1.5, 2.2, -0.8],
];
const b2 = [-0.5, 0.3];

const W3 = [[2.5, -2.0]];
const b3 = [-0.3];

function forwardLayer(input, W, b, activation) {
  return W.map((row, i) => {
    let sum = b[i];
    for (let j = 0; j < row.length; j++) sum += row[j] * input[j];
    return activation(sum);
  });
}

function transformData(points) {
  return points.map((p) => {
    const input = [p.x, p.y];
    const h1 = forwardLayer(input, W1, b1, relu);
    const h2 = forwardLayer(h1, W2, b2, relu);
    const out = forwardLayer(h2, W3, b3, sigmoid);
    return { input, h1, h2, out, label: p.label };
  });
}

// Simple 3D renderer using canvas
function Canvas3D({ data, axes, activeLayer, dimCount, title, subtitle }) {
  const canvasRef = useRef(null);
  const [rotation, setRotation] = useState({ x: -25, y: 35 });
  const [isDragging, setIsDragging] = useState(false);
  const lastPos = useRef({ x: 0, y: 0 });

  const getValues = (d) => {
    if (activeLayer === 0) return [d.input[0], d.input[1], 0];
    if (activeLayer === 1) return [d.h1[0], d.h1[1], d.h1[2] || 0];
    if (activeLayer === 2) return [d.h2[0], d.h2[1], 0];
    return [d.out[0], 0, 0];
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;

    ctx.clearRect(0, 0, W, H);

    const radX = (rotation.x * Math.PI) / 180;
    const radY = (rotation.y * Math.PI) / 180;

    const cosX = Math.cos(radX),
      sinX = Math.sin(radX);
    const cosY = Math.cos(radY),
      sinY = Math.sin(radY);

    function project(x, y, z) {
      const x1 = x * cosY - z * sinY;
      const z1 = x * sinY + z * cosY;
      const y1 = y * cosX - z1 * sinX;
      const z2 = y * sinX + z1 * cosX;
      const scale = 120;
      return {
        px: W / 2 + x1 * scale,
        py: H / 2 - y1 * scale,
        depth: z2,
      };
    }

    // Find ranges
    let allVals = data.map(getValues);
    let ranges = [0, 1, 2].map((i) => {
      let vals = allVals.map((v) => v[i]);
      let min = Math.min(...vals);
      let max = Math.max(...vals);
      let range = max - min || 1;
      return { min, max, range };
    });

    // Normalize to [-1.5, 1.5]
    function normalize(vals) {
      return vals.map((v, i) => ((v - ranges[i].min) / ranges[i].range) * 3 - 1.5);
    }

    // Draw grid
    ctx.strokeStyle = "rgba(120,120,140,0.15)";
    ctx.lineWidth = 0.5;

    for (let i = -1.5; i <= 1.5; i += 0.5) {
      // XY plane grid
      let a = project(i, -1.5, 0);
      let b2p = project(i, 1.5, 0);
      ctx.beginPath();
      ctx.moveTo(a.px, a.py);
      ctx.lineTo(b2p.px, b2p.py);
      ctx.stroke();

      a = project(-1.5, i, 0);
      b2p = project(1.5, i, 0);
      ctx.beginPath();
      ctx.moveTo(a.px, a.py);
      ctx.lineTo(b2p.px, b2p.py);
      ctx.stroke();
    }

    // Draw axes
    const axisLen = 1.8;
    const axisColors = ["#e74c3c", "#2ecc71", "#3498db"];
    const axisEnds = [
      [axisLen, 0, 0],
      [0, axisLen, 0],
      [0, 0, axisLen],
    ];

    const axisCount = dimCount;
    for (let i = 0; i < axisCount; i++) {
      const origin = project(0, 0, 0);
      const end = project(...axisEnds[i]);
      ctx.strokeStyle = axisColors[i];
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(origin.px, origin.py);
      ctx.lineTo(end.px, end.py);
      ctx.stroke();

      // Label
      ctx.fillStyle = axisColors[i];
      ctx.font = "bold 13px 'DM Mono', monospace";
      ctx.fillText(axes[i] || "", end.px + 5, end.py - 5);
    }

    // Draw data points sorted by depth
    let projected = data.map((d, idx) => {
      const vals = normalize(getValues(d));
      const p = project(vals[0], vals[1], dimCount >= 3 ? vals[2] : 0);
      return { ...p, label: d.label, idx };
    });

    projected.sort((a, b) => a.depth - b.depth);

    projected.forEach((p) => {
      const alpha = 0.5 + 0.4 * ((p.depth + 2) / 4);
      const r = 5 + 2 * ((p.depth + 2) / 4);
      ctx.beginPath();
      ctx.arc(p.px, p.py, r, 0, Math.PI * 2);

      if (p.label === 0) {
        ctx.fillStyle = `rgba(255, 107, 107, ${alpha})`;
        ctx.strokeStyle = `rgba(200, 60, 60, ${alpha})`;
      } else {
        ctx.fillStyle = `rgba(100, 180, 255, ${alpha})`;
        ctx.strokeStyle = `rgba(40, 120, 200, ${alpha})`;
      }
      ctx.lineWidth = 1.5;
      ctx.fill();
      ctx.stroke();
    });
  }, [data, rotation, activeLayer, dimCount, axes]);

  const handleMouseDown = (e) => {
    setIsDragging(true);
    lastPos.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseMove = (e) => {
    if (!isDragging) return;
    const dx = e.clientX - lastPos.current.x;
    const dy = e.clientY - lastPos.current.y;
    setRotation((r) => ({
      x: Math.max(-89, Math.min(89, r.x + dy * 0.5)),
      y: r.y + dx * 0.5,
    }));
    lastPos.current = { x: e.clientX, y: e.clientY };
  };

  const handleTouchStart = (e) => {
    if (e.touches.length === 1) {
      setIsDragging(true);
      lastPos.current = { x: e.touches[0].clientX, y: e.touches[0].clientY };
    }
  };

  const handleTouchMove = (e) => {
    if (!isDragging || e.touches.length !== 1) return;
    e.preventDefault();
    const dx = e.touches[0].clientX - lastPos.current.x;
    const dy = e.touches[0].clientY - lastPos.current.y;
    setRotation((r) => ({
      x: Math.max(-89, Math.min(89, r.x + dy * 0.5)),
      y: r.y + dx * 0.5,
    }));
    lastPos.current = { x: e.touches[0].clientX, y: e.touches[0].clientY };
  };

  return (
    <div style={{ position: "relative" }}>
      <div style={{ textAlign: "center", marginBottom: 6 }}>
        <div
          style={{
            fontSize: 15,
            fontWeight: 700,
            color: "#e0e0e0",
            fontFamily: "'DM Mono', monospace",
            letterSpacing: 1,
          }}
        >
          {title}
        </div>
        <div
          style={{
            fontSize: 12,
            color: "#888",
            fontFamily: "'DM Sans', sans-serif",
            marginTop: 2,
          }}
        >
          {subtitle}
        </div>
      </div>
      <canvas
        ref={canvasRef}
        width={460}
        height={380}
        style={{
          cursor: isDragging ? "grabbing" : "grab",
          borderRadius: 12,
          background: "rgba(15,15,25,0.6)",
          border: "1px solid rgba(255,255,255,0.06)",
          display: "block",
          margin: "0 auto",
          maxWidth: "100%",
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={() => setIsDragging(false)}
        onMouseLeave={() => setIsDragging(false)}
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={() => setIsDragging(false)}
      />
      <div
        style={{
          textAlign: "center",
          fontSize: 11,
          color: "#555",
          marginTop: 4,
          fontFamily: "'DM Sans', sans-serif",
        }}
      >
        拖拽旋转坐标系
      </div>
    </div>
  );
}

// Network architecture diagram
function NetworkDiagram({ activeLayer }) {
  const layers = [
    { name: "输入层", neurons: 2, label: "x₁, x₂" },
    { name: "隐藏层1", neurons: 3, label: "h₁,h₂,h₃" },
    { name: "隐藏层2", neurons: 2, label: "z₁, z₂" },
    { name: "输出层", neurons: 1, label: "ŷ" },
  ];

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        gap: 0,
        padding: "16px 0",
        flexWrap: "wrap",
      }}
    >
      {layers.map((layer, li) => (
        <div key={li} style={{ display: "flex", alignItems: "center" }}>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 6,
              padding: "10px 16px",
              borderRadius: 10,
              background:
                li === activeLayer
                  ? "rgba(100,180,255,0.12)"
                  : "rgba(255,255,255,0.02)",
              border:
                li === activeLayer
                  ? "1.5px solid rgba(100,180,255,0.4)"
                  : "1.5px solid rgba(255,255,255,0.05)",
              transition: "all 0.3s",
              minWidth: 70,
            }}
          >
            <div
              style={{
                fontSize: 11,
                color: li === activeLayer ? "#64b4ff" : "#666",
                fontFamily: "'DM Sans', sans-serif",
                fontWeight: 600,
              }}
            >
              {layer.name}
            </div>
            <div style={{ display: "flex", gap: 5 }}>
              {Array.from({ length: layer.neurons }).map((_, ni) => (
                <div
                  key={ni}
                  style={{
                    width: 14,
                    height: 14,
                    borderRadius: "50%",
                    background:
                      li === activeLayer
                        ? "rgba(100,180,255,0.6)"
                        : "rgba(255,255,255,0.15)",
                    border: "1px solid rgba(255,255,255,0.2)",
                    transition: "all 0.3s",
                  }}
                />
              ))}
            </div>
            <div
              style={{
                fontSize: 11,
                color: "#777",
                fontFamily: "'DM Mono', monospace",
              }}
            >
              {layer.label}
            </div>
          </div>
          {li < layers.length - 1 && (
            <div
              style={{
                padding: "0 8px",
                color:
                  li === activeLayer || li + 1 === activeLayer
                    ? "#64b4ff"
                    : "#333",
                fontSize: 18,
                transition: "color 0.3s",
              }}
            >
              →
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

const layerInfo = [
  {
    title: "输入层：原始特征空间",
    subtitle: "2个原始特征 → 2维坐标系",
    axes: ["x₁", "x₂"],
    dims: 2,
    desc: "红蓝两类样本在原始特征空间中形成同心圆——一根木棍（直线）无法将它们分开。这就是为什么需要深度学习。",
    matrix: "每个样本是 [x₁, x₂]",
  },
  {
    title: "隐藏层1：第一次坐标改写",
    subtitle: "3个神经元 → 3维坐标系",
    axes: ["h₁", "h₂", "h₃"],
    dims: 3,
    desc: "3个神经元就是3个"手电筒"，从不同角度照射原始数据，把2维的点投影成3个标量再经过ReLU弯曲，拼成3维坐标。原来缠绕在一起的同心圆，在三维空间中被"拎"开了。",
    matrix: "W₁[3×2] · [x₁,x₂]ᵀ + b₁ → ReLU → [h₁,h₂,h₃]",
  },
  {
    title: "隐藏层2：第二次坐标改写",
    subtitle: "2个神经元 → 2维坐标系",
    axes: ["z₁", "z₂"],
    dims: 2,
    desc: "2个神经元再次改写坐标，把3维降回2维。此时红蓝两类已经被推到了可以被一根木棍分开的位置——前面两个房间的"翻译"工作做完了。",
    matrix: "W₂[2×3] · [h₁,h₂,h₃]ᵀ + b₂ → ReLU → [z₁,z₂]",
  },
  {
    title: "输出层：最终预测",
    subtitle: "1个神经元 → 1维坐标",
    axes: ["ŷ"],
    dims: 1,
    desc: "最后一层只有1个神经元，做的事和机器学习的逻辑回归一模一样——拿到整理好的特征，用sigmoid输出一个概率值。红色聚在0附近，蓝色聚在1附近，弹簧（Loss）在这里连接预测值和真实标签。",
    matrix: "W₃[1×2] · [z₁,z₂]ᵀ + b₃ → σ → ŷ",
  },
];

export default function NeuralNetViz() {
  const [activeLayer, setActiveLayer] = useState(0);
  const [data] = useState(() => {
    const raw = generateData(80);
    return transformData(raw);
  });

  const info = layerInfo[activeLayer];

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "linear-gradient(160deg, #0a0a14 0%, #0f1029 50%, #0a0a14 100%)",
        color: "#e0e0e0",
        fontFamily: "'DM Sans', sans-serif",
        padding: "24px 16px",
      }}
    >
      <link
        href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&display=swap"
        rel="stylesheet"
      />

      <div style={{ maxWidth: 520, margin: "0 auto" }}>
        {/* Title */}
        <div style={{ textAlign: "center", marginBottom: 20 }}>
          <h1
            style={{
              fontSize: 20,
              fontWeight: 700,
              margin: 0,
              letterSpacing: 1,
              fontFamily: "'DM Mono', monospace",
              background: "linear-gradient(135deg, #64b4ff, #a78bfa)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
            }}
          >
            神经网络层间数据变换
          </h1>
          <p style={{ color: "#666", fontSize: 13, margin: "6px 0 0" }}>
            观察同一批样本在每一层坐标系中的位置如何变化
          </p>
        </div>

        {/* Network diagram */}
        <NetworkDiagram activeLayer={activeLayer} />

        {/* Layer selector */}
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            gap: 6,
            margin: "16px 0",
            flexWrap: "wrap",
          }}
        >
          {layerInfo.map((l, i) => (
            <button
              key={i}
              onClick={() => setActiveLayer(i)}
              style={{
                padding: "8px 14px",
                borderRadius: 8,
                border:
                  i === activeLayer
                    ? "1.5px solid rgba(100,180,255,0.5)"
                    : "1px solid rgba(255,255,255,0.08)",
                background:
                  i === activeLayer
                    ? "rgba(100,180,255,0.12)"
                    : "rgba(255,255,255,0.03)",
                color: i === activeLayer ? "#64b4ff" : "#888",
                fontSize: 12,
                fontWeight: 600,
                cursor: "pointer",
                fontFamily: "'DM Sans', sans-serif",
                transition: "all 0.2s",
              }}
            >
              {i === 0
                ? "输入层"
                : i === layerInfo.length - 1
                ? "输出层"
                : `隐藏层${i}`}
            </button>
          ))}
        </div>

        {/* 3D Canvas */}
        <Canvas3D
          data={data}
          axes={info.axes}
          activeLayer={activeLayer}
          dimCount={info.dims}
          title={info.title}
          subtitle={info.subtitle}
        />

        {/* Matrix notation */}
        <div
          style={{
            margin: "16px auto",
            padding: "10px 16px",
            background: "rgba(167, 139, 250, 0.06)",
            border: "1px solid rgba(167, 139, 250, 0.15)",
            borderRadius: 8,
            textAlign: "center",
            fontFamily: "'DM Mono', monospace",
            fontSize: 13,
            color: "#a78bfa",
            maxWidth: 460,
          }}
        >
          {info.matrix}
        </div>

        {/* Description */}
        <div
          style={{
            margin: "0 auto",
            padding: "14px 18px",
            background: "rgba(255,255,255,0.02)",
            border: "1px solid rgba(255,255,255,0.06)",
            borderRadius: 10,
            fontSize: 14,
            lineHeight: 1.7,
            color: "#bbb",
            maxWidth: 460,
          }}
        >
          {info.desc}
        </div>

        {/* Legend */}
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            gap: 24,
            margin: "16px 0",
            fontSize: 12,
            color: "#888",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div
              style={{
                width: 10,
                height: 10,
                borderRadius: "50%",
                background: "rgba(255,107,107,0.8)",
              }}
            />
            类别 0（内圈）
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div
              style={{
                width: 10,
                height: 10,
                borderRadius: "50%",
                background: "rgba(100,180,255,0.8)",
              }}
            />
            类别 1（外圈）
          </div>
        </div>

        {/* Insight box */}
        <div
          style={{
            margin: "12px auto",
            padding: "14px 18px",
            background: "rgba(100,180,255,0.04)",
            border: "1px solid rgba(100,180,255,0.1)",
            borderRadius: 10,
            fontSize: 13,
            lineHeight: 1.7,
            color: "#8ab4d8",
            maxWidth: 460,
          }}
        >
          <div style={{ fontWeight: 700, marginBottom: 4, color: "#64b4ff" }}>
            房间类比对应
          </div>
          {activeLayer === 0
            ? "这就是房间1的软木板——图钉钉死在原始特征坐标上，不会动。红蓝两类缠绕在一起，一根木棍（直线）无法分开。"
            : activeLayer === 1
            ? "这就是房间2的软木板——大头针的位置由房间1的3个\"手电筒\"决定。注意红蓝两类在三维空间中开始分离了——这就是\"坐标改写\"的力量。拖拽旋转看看！"
            : activeLayer === 2
            ? "这就是房间3之前的准备——大头针被进一步重新排列。红蓝两类已经几乎可以用一根木棍分开了。前面两个房间的\"翻译\"工作到此结束。"
            : "最后一个房间。只剩一个维度——概率值。红色样本被推向0，蓝色被推向1。弹簧（Loss）就连接在这里。这和机器学习中的逻辑回归一模一样。"}
        </div>
      </div>
    </div>
  );
}

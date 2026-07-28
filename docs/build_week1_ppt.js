const pptxgen = require('pptxgenjs');

const pptx = new pptxgen();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = 'AVagent Project';
pptx.company = 'AVagent';
pptx.subject = '音视频生成评测 Agent 第一周进展';
pptx.title = '音视频生成评测 Agent｜第一周进展汇报';
pptx.lang = 'zh-CN';
pptx.theme = {
  headFontFace: 'Microsoft YaHei',
  bodyFontFace: 'Microsoft YaHei',
  lang: 'zh-CN',
};
pptx.defineLayout({ name: 'CUSTOM_WIDE', width: 13.333, height: 7.5 });
pptx.layout = 'CUSTOM_WIDE';

const C = {
  ink: '101827',
  ink2: '18243A',
  bg: 'F5F7FA',
  white: 'FFFFFF',
  text: '24324A',
  muted: '6C7A91',
  line: 'DDE4EC',
  teal: '25C2A0',
  tealDark: '0C8F7A',
  tealPale: 'DDF7F1',
  orange: 'FF6B4A',
  orangePale: 'FFF0EB',
  amber: 'F4B740',
  amberPale: 'FFF6D9',
  blue: '4D7CFE',
  bluePale: 'EAF0FF',
  violet: '8B6CF6',
  violetPale: 'EFEAFF',
  green: '3DBA78',
};

const F = 'Microsoft YaHei';
const S = pptx.ShapeType;
const shadow = () => ({ type: 'outer', color: '172033', blur: 2, angle: 45, distance: 1, opacity: 0.10 });

function addText(slide, text, x, y, w, h, opts = {}) {
  slide.addText(text, {
    x, y, w, h,
    fontFace: F,
    fontSize: opts.fontSize || 16,
    color: opts.color || C.text,
    bold: opts.bold || false,
    margin: opts.margin === undefined ? 0 : opts.margin,
    valign: opts.valign || 'mid',
    align: opts.align || 'left',
    breakLine: false,
    fit: 'shrink',
    ...opts,
  });
}

function addHeader(slide, title, kicker, page) {
  slide.background = { color: C.bg };
  addText(slide, kicker.toUpperCase(), 0.62, 0.35, 3.3, 0.24, {
    fontSize: 10, color: C.tealDark, bold: true, charSpacing: 1.4,
  });
  addText(slide, title, 0.62, 0.68, 11.5, 0.62, {
    fontSize: 30, color: C.ink, bold: true,
  });
  addText(slide, String(page).padStart(2, '0'), 12.15, 0.38, 0.55, 0.28, {
    fontSize: 11, color: C.muted, align: 'right', bold: true,
  });
}

function addFooter(slide, source = '') {
  addText(slide, 'AV GENERATION EVALUATION AGENT · WEEK 1', 0.62, 7.12, 4.4, 0.18, {
    fontSize: 8.5, color: '8895A8', charSpacing: 0.7,
  });
  if (source) {
    addText(slide, source, 6.2, 7.08, 6.45, 0.24, {
      fontSize: 8.2, color: '8895A8', align: 'right',
    });
  }
}

function addCard(slide, x, y, w, h, fill = C.white, line = C.line, radius = true) {
  slide.addShape(radius ? S.roundRect : S.rect, {
    x, y, w, h,
    rectRadius: 0.08,
    fill: { color: fill },
    line: { color: line, width: 0.8 },
    shadow: shadow(),
  });
}

function addPill(slide, text, x, y, w, fill, color, opts = {}) {
  slide.addShape(S.roundRect, {
    x, y, w, h: opts.h || 0.33,
    rectRadius: 0.08,
    fill: { color: fill },
    line: { color: fill },
  });
  addText(slide, text, x, y + 0.01, w, (opts.h || 0.33) - 0.02, {
    fontSize: opts.fontSize || 10.5, color, bold: true, align: 'center',
  });
}

function addNumberCircle(slide, n, x, y, fill = C.teal, color = C.ink) {
  slide.addShape(S.ellipse, {
    x, y, w: 0.46, h: 0.46,
    fill: { color: fill }, line: { color: fill },
  });
  addText(slide, String(n), x, y, 0.46, 0.46, { fontSize: 13, color, bold: true, align: 'center' });
}

function addArrow(slide, x1, y1, x2, y2, color = C.teal, width = 2) {
  slide.addShape(S.line, {
    x: x1, y: y1, w: x2 - x1, h: y2 - y1,
    line: { color, width, endArrowType: 'triangle' },
  });
}

function addNotes(slide, text) {
  slide.addNotes(text);
}

function addFrameVisual(slide, x, y, w, h, label, time, accent = C.orange) {
  slide.addShape(S.roundRect, {
    x, y, w, h, rectRadius: 0.08,
    fill: { color: C.ink2 }, line: { color: '3A4962', width: 1.2 },
  });
  slide.addShape(S.rect, {
    x: x + 0.2, y: y + 0.22, w: w - 0.4, h: h - 0.48,
    fill: { color: '23344F' }, line: { color: '23344F' },
  });
  slide.addShape(S.ellipse, {
    x: x + 0.66, y: y + 0.64, w: 0.62, h: 0.82,
    fill: { color: '5B6D89' }, line: { color: '5B6D89' },
  });
  slide.addShape(S.roundRect, {
    x: x + 0.44, y: y + 1.35, w: 1.2, h: 0.92, rectRadius: 0.06,
    fill: { color: '53657F' }, line: { color: '53657F' },
  });
  slide.addShape(S.rect, {
    x: x + w - 1.7, y: y + 0.42, w: 1.16, h: 0.92,
    fill: { color: '314761' }, line: { color: '314761' },
  });
  slide.addShape(S.rect, {
    x: x + 0.47, y: y + 0.55, w: 1.35, h: 1.92,
    fill: { color: 'FFFFFF', transparency: 100 }, line: { color: accent, width: 2.5 },
  });
  addPill(slide, label, x + 0.47, y + h - 0.48, Math.min(2.1, w - 1.1), accent, C.white, { h: 0.3, fontSize: 9.5 });
  addText(slide, time, x + w - 1.1, y + h - 0.42, 0.78, 0.2, { fontSize: 9, color: C.white, align: 'right' });
}

// Slide 1: Cover
{
  const slide = pptx.addSlide();
  slide.background = { color: C.ink };
  slide.addShape(S.rect, { x: 0, y: 0, w: 13.333, h: 7.5, fill: { color: C.ink }, line: { color: C.ink } });
  slide.addShape(S.ellipse, { x: 9.5, y: -1.0, w: 5.0, h: 5.0, fill: { color: C.teal, transparency: 82 }, line: { color: C.teal, transparency: 100 } });
  slide.addShape(S.ellipse, { x: -1.4, y: 4.7, w: 4.2, h: 4.2, fill: { color: C.orange, transparency: 88 }, line: { color: C.orange, transparency: 100 } });
  addPill(slide, 'WEEK 01 · PROGRESS REVIEW', 0.72, 0.62, 2.55, C.teal, C.ink, { h: 0.34, fontSize: 10 });
  addText(slide, '音视频生成评测 Agent', 0.72, 1.35, 7.3, 0.92, { fontSize: 38, color: C.white, bold: true });
  addText(slide, '第一周进展汇报', 0.72, 2.23, 6.6, 0.72, { fontSize: 28, color: 'C9D5E8', bold: true });
  addText(slide, '从调研地图到“可提示、可定位、维度化”的技术路线', 0.75, 3.2, 6.9, 0.42, { fontSize: 17, color: 'AAB8CC' });
  addText(slide, '2026.07.21', 0.75, 6.6, 2.2, 0.3, { fontSize: 12, color: '90A0B7', bold: true, charSpacing: 1.3 });
  addFrameVisual(slide, 8.2, 1.45, 4.25, 3.35, '结构畸变', '[T=2.40s]', C.orange);
  for (let i = 0; i < 4; i += 1) {
    slide.addShape(S.line, { x: 8.45 + i * 0.9, y: 5.35, w: 0.62, h: 0, line: { color: i === 2 ? C.orange : '53647E', width: i === 2 ? 3 : 1.5 } });
    addText(slide, `0${i + 1}`, 8.42 + i * 0.9, 5.52, 0.68, 0.18, { fontSize: 8, color: '8795AA', align: 'center' });
  }
  addNotes(slide, '开场：本周工作的重点是领域学习与问题收敛。今天不重复项目背景，直接汇报本周建立了什么认知、形成了哪些技术判断，以及第二周准备如何验证。');
}

// Slide 2: Goals and completion
{
  const slide = pptx.addSlide();
  addHeader(slide, '本周目标与完成情况', '01 · WEEKLY OUTPUT', 2);
  const cards = [
    { x: 0.72, color: C.teal, pale: C.tealPale, tag: '已完成', n: '01', title: '领域学习', body: '系统梳理 Agentic Evaluation、缺陷接地、内在保真与音视频联合评测。' },
    { x: 4.47, color: C.blue, pale: C.bluePale, tag: '已完成', n: '02', title: '方法梳理', body: '形成代表工作对照表，明确不同 Benchmark 的输入、输出与评测维度。' },
    { x: 8.22, color: C.violet, pale: C.violetPale, tag: '形成判断', n: '03', title: '项目收敛', body: '收敛出四维问题体系、接地输出协议与 Agent + 工具的候选路线。' },
  ];
  cards.forEach((c) => {
    addCard(slide, c.x, 1.64, 3.35, 3.8, C.white, C.line);
    slide.addShape(S.ellipse, { x: c.x + 0.32, y: 1.96, w: 0.72, h: 0.72, fill: { color: c.pale }, line: { color: c.pale } });
    addText(slide, c.n, c.x + 0.32, 1.96, 0.72, 0.72, { fontSize: 17, color: c.color, bold: true, align: 'center' });
    addPill(slide, c.tag, c.x + 2.15, 1.98, 0.84, c.pale, c.color, { h: 0.3, fontSize: 9.5 });
    addText(slide, c.title, c.x + 0.34, 2.95, 2.65, 0.42, { fontSize: 22, color: C.ink, bold: true });
    addText(slide, c.body, c.x + 0.34, 3.55, 2.64, 1.15, { fontSize: 14.5, color: C.muted, valign: 'top', breakLine: false });
  });
  addCard(slide, 0.72, 5.78, 10.85, 0.76, C.ink2, C.ink2, true);
  addText(slide, '本周核心产物', 1.04, 5.96, 1.25, 0.28, { fontSize: 12, color: C.teal, bold: true });
  addText(slide, '《音视频生成评测 Agent 研究调研》', 2.42, 5.89, 4.2, 0.39, { fontSize: 18, color: C.white, bold: true });
  addPill(slide, '待实验验证：模型复现 / 基线结果 / bbox 样例', 7.42, 5.98, 3.72, C.orange, C.white, { h: 0.3, fontSize: 9.6 });
  addFooter(slide, '依据：avgen_eval_survey.docx');
  addNotes(slide, '本页先把完成边界说清楚：本周完成的是调研、梳理和路线判断。模型复现、基线结果和真实 bbox 样例还没有形成可验证产物，后面不会把它们当成本周成果。');
}

// Slide 3: Research questions
{
  const slide = pptx.addSlide();
  addHeader(slide, '本周调研范围与问题拆解', '02 · RESEARCH SCOPE', 3);
  addCard(slide, 4.88, 2.7, 3.55, 1.34, C.ink2, C.ink2);
  addText(slide, '总体任务', 5.18, 2.91, 0.9, 0.27, { fontSize: 11, color: C.teal, bold: true });
  addText(slide, '细粒度音视频生成评测 Agent', 5.18, 3.24, 2.95, 0.42, { fontSize: 18.5, color: C.white, bold: true, align: 'center' });
  const qs = [
    { x: 0.72, y: 1.62, n: 'Q1', title: 'Agent 如何组织？', body: '规划、判断、工具调用与迭代如何形成 loop', color: C.teal, pale: C.tealPale },
    { x: 9.18, y: 1.62, n: 'Q2', title: '缺陷如何分类？', body: '从 L1 到 L3 建立可执行的问题体系', color: C.blue, pale: C.bluePale },
    { x: 0.72, y: 4.62, n: 'Q3', title: '问题如何定位？', body: '类别 + 时间区间 + bbox / mask', color: C.orange, pale: C.orangePale },
    { x: 9.18, y: 4.62, n: 'Q4', title: '音视频增加什么？', body: '事件同步、唇音同步与联合物理一致性', color: C.violet, pale: C.violetPale },
  ];
  qs.forEach((q) => {
    addCard(slide, q.x, q.y, 3.35, 1.52, C.white, C.line);
    addPill(slide, q.n, q.x + 0.28, q.y + 0.25, 0.58, q.pale, q.color, { h: 0.3, fontSize: 10 });
    addText(slide, q.title, q.x + 1.0, q.y + 0.2, 2.0, 0.42, { fontSize: 17, color: C.ink, bold: true });
    addText(slide, q.body, q.x + 0.3, q.y + 0.78, 2.74, 0.44, { fontSize: 12.8, color: C.muted, valign: 'top' });
  });
  slide.addShape(S.line, { x: 4.12, y: 2.5, w: 0.78, h: 0.55, line: { color: 'A8B6C9', width: 1.3 } });
  slide.addShape(S.line, { x: 8.42, y: 2.5, w: 0.78, h: 0.55, line: { color: 'A8B6C9', width: 1.3 } });
  slide.addShape(S.line, { x: 4.12, y: 3.7, w: 0.78, h: 1.4, line: { color: 'A8B6C9', width: 1.3 } });
  slide.addShape(S.line, { x: 8.42, y: 3.7, w: 0.78, h: 1.4, line: { color: 'A8B6C9', width: 1.3 } });
  addPill(slide, '本周边界：方法 / Benchmark / 输出协议 / 可落地路线', 4.25, 5.72, 4.84, C.amberPale, '8B6514', { h: 0.38, fontSize: 11 });
  addFooter(slide);
  addNotes(slide, '本周没有从“所有视频质量问题”泛泛展开，而是把总体任务拆成四个问题。后面的调研地图、论文对照和技术路线都围绕这四个问题组织。');
}

// Slide 4: Knowledge map
{
  const slide = pptx.addSlide();
  addHeader(slide, '本周建立的五类工作地图', '03 · LANDSCAPE', 4);
  addText(slide, '按“评测产出形态”归纳四类方法，并单列文本到音视频联合评测作为模态扩展', 0.72, 1.3, 11.9, 0.26, { fontSize: 11.5, color: C.muted });
  const lanes = [
    { color: C.teal, pale: C.tealPale, title: 'Agent / 可提示框架', works: 'Evaluation Agent · VideoGen-Eval · VQQA', tag: '规划与多轮评测' },
    { color: C.blue, pale: C.bluePale, title: 'MLLM-as-Judge / 奖励模型', works: 'VideoScore2 · VideoReward · VF-Eval', tag: '评分、偏好与反馈' },
    { color: C.orange, pale: C.orangePale, title: '缺陷检测 + 时空定位', works: 'Skyra / ViF-Bench · BrokenVideos · DVAR', tag: '时间 + bbox / mask' },
    { color: C.violet, pale: C.violetPale, title: '内在保真度 / 物理 / 人体', works: 'VBench-2.0 · VideoPhy-2 · GRADEO', tag: '深层生成正确性' },
    { color: C.green, pale: 'E2F7EB', title: '文本 → 音视频联合评测', works: 'AVGen-Bench · PhyAVBench · MTAVG-Bench', tag: '音视同步与一致性' },
  ];
  lanes.forEach((l, i) => {
    const y = 1.72 + i * 1.02;
    addCard(slide, 0.72, y, 11.9, 0.82, C.white, C.line);
    slide.addShape(S.rect, { x: 0.72, y, w: 0.11, h: 0.82, fill: { color: l.color }, line: { color: l.color } });
    addPill(slide, String(i + 1).padStart(2, '0'), 1.02, y + 0.24, 0.56, l.pale, l.color, { h: 0.3, fontSize: 9.5 });
    addText(slide, l.title, 1.82, y + 0.14, 3.45, 0.42, { fontSize: 16.2, color: C.ink, bold: true });
    addText(slide, l.works, 5.35, y + 0.16, 4.62, 0.38, { fontSize: 11.2, color: C.muted, bold: true });
    addPill(slide, l.tag, 10.23, y + 0.24, 2.02, l.pale, l.color, { h: 0.3, fontSize: 9.2 });
  });
  addFooter(slide, '来源：avgen_eval_survey.docx 表 1');
  addNotes(slide, '这里严格按照调研报告表 1 展示五类工作。前三类主要区分评测主体与输出形态，第四类按评测内容深化，第五类则是从纯视频向联合音视频的模态扩展。');
}

// Slide 5: Work matrix
{
  const slide = pptx.addSlide();
  addHeader(slide, '五类工作的产出形态与项目价值', '04 · CATEGORY COMPARISON', 5);
  const cols = [0.72, 3.12, 7.08, 8.25, 9.42, 10.59];
  const widths = [2.3, 3.86, 1.07, 1.07, 1.07, 2.03];
  const headers = ['类别', '代表性工作', '主要\n产出', '时空\n接地', '音视\n联合', '对本项目的价值'];
  headers.forEach((h, i) => {
    slide.addShape(S.rect, { x: cols[i], y: 1.55, w: widths[i], h: 0.62, fill: { color: i === 0 || i === 5 ? C.ink2 : 'E8EDF4' }, line: { color: C.bg } });
    addText(slide, h, cols[i] + 0.06, 1.58, widths[i] - 0.12, 0.54, { fontSize: 10.5, color: i === 0 || i === 5 ? C.white : C.text, bold: true, align: i === 0 || i === 1 || i === 5 ? 'left' : 'center' });
  });
  const rows = [
    ['Agent / 可提示框架', 'Evaluation Agent · VideoGen-Eval · VQQA', 'CoT', 0, 0, '规划与主执行器'],
    ['MLLM-as-Judge / 奖励模型', 'VideoScore2 · VideoReward · VF-Eval', 'S/P/QA', 0, 0, '多维判断与反馈'],
    ['缺陷检测 + 时空定位', 'Skyra · BrokenVideos · DVAR', 'G+CoT', 1, 0, '接地协议与数据'],
    ['内在保真度 / 物理 / 人体', 'VBench-2.0 · VideoPhy-2 · GRADEO', 'S/CoT', 0, 0, '人体、物理与常识'],
    ['文本 → 音视频联合评测', 'AVGen-Bench · PhyAVBench · MTAVG-Bench', 'S/QA', 0, 1, '音视同步与一致性'],
  ];
  rows.forEach((r, ri) => {
    const y = 2.17 + ri * 0.82;
    const fill = ri % 2 === 0 ? C.white : 'F0F3F7';
    widths.forEach((w, i) => slide.addShape(S.rect, { x: cols[i], y, w, h: 0.82, fill: { color: fill }, line: { color: C.bg } }));
    addText(slide, r[0], cols[0] + 0.16, y + 0.09, widths[0] - 0.28, 0.62, { fontSize: 10.4, color: C.ink, bold: true });
    addText(slide, r[1], cols[1] + 0.16, y + 0.1, widths[1] - 0.28, 0.58, { fontSize: 10.6, color: C.text, bold: true });
    addPill(slide, r[2], cols[2] + 0.16, y + 0.25, widths[2] - 0.32, 'E8EDF4', C.text, { h: 0.3, fontSize: 9.2 });
    [3, 4].forEach((i) => {
      if (r[i]) {
        slide.addShape(S.ellipse, { x: cols[i] + widths[i] / 2 - 0.12, y: y + 0.29, w: 0.24, h: 0.24, fill: { color: i === 3 ? C.orange : C.violet }, line: { color: i === 3 ? C.orange : C.violet } });
      } else {
        slide.addShape(S.line, { x: cols[i] + widths[i] / 2 - 0.11, y: y + 0.41, w: 0.22, h: 0, line: { color: 'BAC4D2', width: 1 } });
      }
    });
    const emphasis = ri === 2 ? C.orange : ri === 4 ? C.violet : C.tealDark;
    const pale = ri === 2 ? C.orangePale : ri === 4 ? C.violetPale : C.tealPale;
    addPill(slide, r[5], cols[5] + 0.13, y + 0.25, widths[5] - 0.26, pale, emphasis, { h: 0.3, fontSize: 8.9 });
  });
  addCard(slide, 0.72, 6.45, 11.9, 0.48, C.ink2, C.ink2);
  addText(slide, '项目组合', 0.95, 6.56, 0.88, 0.22, { fontSize: 10.5, color: C.teal, bold: true });
  addText(slide, 'Agent 组织评测流程；Judge 提供判断；Grounding 与专用工具补足定位、人体和音视维度。', 1.92, 6.49, 10.2, 0.32, { fontSize: 13.2, color: C.white, bold: true });
  addFooter(slide, 'S=标量分数 · P=成对偏好 · QA=诊断问答 · CoT=解释 · G=时空接地');
  addNotes(slide, '五类工作并非互斥，而是对应系统中的不同能力。当前项目需要把它们组合起来：Agent 负责组织流程，Judge 负责多维判断，Grounding 和专用工具解决空间、时间、人体及音视同步。');
}

// Slide 6: Grounding protocol
{
  const slide = pptx.addSlide();
  addHeader(slide, '最关键启发：缺陷需要被“接地”', '05 · GROUNDED DIAGNOSIS', 6);
  addFrameVisual(slide, 0.72, 1.55, 5.6, 4.55, '手部畸变', '[T=2.40s]', C.orange);
  // Replace the generic person-level box with a small-region hand box.
  slide.addShape(S.rect, {
    x: 1.19, y: 2.1, w: 1.35, h: 1.92,
    fill: { color: '23344F' }, line: { color: '23344F', width: 3.2 },
  });
  slide.addShape(S.ellipse, {
    x: 1.38, y: 2.19, w: 0.62, h: 0.82,
    fill: { color: '5B6D89' }, line: { color: '5B6D89' },
  });
  slide.addShape(S.roundRect, {
    x: 1.16, y: 2.9, w: 1.2, h: 0.92, rectRadius: 0.06,
    fill: { color: '53657F' }, line: { color: '53657F' },
  });
  slide.addShape(S.ellipse, {
    x: 2.45, y: 3.56, w: 0.26, h: 0.26,
    fill: { color: '5B6D89' }, line: { color: '5B6D89' },
  });
  slide.addShape(S.rect, {
    x: 2.35, y: 3.43, w: 0.48, h: 0.52,
    fill: { color: C.white, transparency: 100 }, line: { color: C.orange, width: 2.5 },
  });
  addText(slide, '采样帧 + 时间戳', 1.04, 6.28, 2.2, 0.26, { fontSize: 11, color: C.muted, bold: true });
  for (let i = 0; i < 5; i += 1) {
    slide.addShape(S.line, { x: 3.14 + i * 0.54, y: 6.42, w: 0.38, h: 0, line: { color: i === 2 ? C.orange : 'AAB6C7', width: i === 2 ? 3 : 1.2 } });
  }
  addArrow(slide, 6.42, 3.75, 7.05, 3.75, C.teal, 2.4);
  addCard(slide, 7.18, 1.55, 5.2, 4.55, C.white, C.line);
  addPill(slide, 'STRUCTURED OUTPUT', 7.5, 1.89, 1.82, C.ink2, C.white, { h: 0.32, fontSize: 9.5 });
  const fields = [
    { label: '类别', value: '人体与手部保真 / 手部畸变', color: C.orange },
    { label: '解释', value: '右手手指数量异常，轮廓发生融合', color: C.blue },
    { label: '时间', value: '[2.10s, 2.80s]', color: C.teal },
    { label: '空间', value: 'bbox = [0.18, 0.31, 0.34, 0.62]', color: C.violet },
  ];
  fields.forEach((f, i) => {
    const y = 2.54 + i * 0.72;
    addPill(slide, f.label, 7.52, y, 0.62, f.color, C.white, { h: 0.3, fontSize: 9.5 });
    addText(slide, f.value, 8.35, y - 0.02, 3.62, 0.34, { fontSize: 12.7, color: C.text, bold: i === 0 });
  });
  addCard(slide, 7.52, 5.52, 4.46, 0.36, C.orangePale, C.orangePale, true);
  addText(slide, '从“这段视频不好” → “什么问题、何时发生、在哪里”', 7.69, 5.57, 4.1, 0.2, { fontSize: 10.8, color: C.orange, bold: true, align: 'center' });
  addFooter(slide, 'Skyra / ViF-Bench；BrokenVideos');
  addNotes(slide, '这是与项目最直接相关的结论。输出不应只给一个分数或一句评价，而应强制每条缺陷携带类别、解释、时间区间和 bbox。这样结果才可审计，也能进入后续 Benchmark 和误差分析。');
}

// Slide 7: Taxonomy
{
  const slide = pptx.addSlide();
  addHeader(slide, '由调研收敛出的评测问题体系', '06 · TAXONOMY V0', 7);
  const dims = [
    { x: 0.72, y: 1.55, color: C.blue, pale: C.bluePale, no: 'D1', title: '文本 / 参考约束对齐', items: ['对象、数量与属性', '动作主体与关系', '身份、服装与参考图'] },
    { x: 6.73, y: 1.55, color: C.orange, pale: C.orangePale, no: 'D2', title: '人体与手部保真', items: ['手指数目与形态', '穿模、融合与断裂', '人体解剖异常'] },
    { x: 0.72, y: 4.18, color: C.teal, pale: C.tealPale, no: 'D3', title: '时序一致性', items: ['主体身份漂移', '背景 / 物体闪烁', '跨帧动作连续性'] },
    { x: 6.73, y: 4.18, color: C.violet, pale: C.violetPale, no: 'D4', title: '音视一致与同步', items: ['事件级起音同步', '唇音同步', '声音的物理合理性'] },
  ];
  dims.forEach((d) => {
    addCard(slide, d.x, d.y, 5.58, 2.15, C.white, C.line);
    addPill(slide, d.no, d.x + 0.28, d.y + 0.28, 0.68, d.color, C.white, { h: 0.33, fontSize: 10.5 });
    addText(slide, d.title, d.x + 1.16, d.y + 0.23, 3.98, 0.42, { fontSize: 18, color: C.ink, bold: true });
    d.items.forEach((item, i) => {
      slide.addShape(S.ellipse, { x: d.x + 0.38, y: d.y + 0.91 + i * 0.35, w: 0.12, h: 0.12, fill: { color: d.color }, line: { color: d.color } });
      addText(slide, item, d.x + 0.65, d.y + 0.82 + i * 0.35, 4.35, 0.28, { fontSize: 12.5, color: C.muted });
    });
  });
  addFooter(slide, '参考：T2V-CompBench · VBench-2.0 · Skyra · AVGen-Bench');
  addNotes(slide, '当前问题体系先收敛为四个一级维度。第二周会继续向 L2、L3 细化，并把每个子类的定义、正反例和 bbox 标注规则写成协议。');
}

// Slide 8: Agent loop
{
  const slide = pptx.addSlide();
  addHeader(slide, '拟搭建的 Agent Loop', '07 · SYSTEM ROUTE', 8);
  const nodes = [
    { x: 0.66, w: 1.62, no: '01', title: '输入', sub: 'prompt / 参考图\n生成音视频', color: C.blue, pale: C.bluePale },
    { x: 2.52, w: 1.68, no: '02', title: '采样', sub: '帧 + 时间戳\n音频片段', color: C.teal, pale: C.tealPale },
    { x: 4.45, w: 1.75, no: '03', title: '评测 Skill', sub: '维度规划\n结构化提示', color: C.violet, pale: C.violetPale },
    { x: 6.45, w: 1.8, no: '04', title: '工具校验', sub: '人体 / 定位\n音视同步', color: C.orange, pale: C.orangePale },
    { x: 8.5, w: 1.72, no: '05', title: '结构化缺陷', sub: '类别 / 时间\nbbox / 解释', color: C.teal, pale: C.tealPale },
    { x: 10.47, w: 2.18, no: '06', title: 'Benchmark', sub: 'IoU / F1 / 一致性\n误差分析', color: C.blue, pale: C.bluePale },
  ];
  nodes.forEach((n, i) => {
    addCard(slide, n.x, 2.2, n.w, 2.25, C.white, C.line);
    addPill(slide, n.no, n.x + 0.23, 2.46, 0.52, n.pale, n.color, { h: 0.29, fontSize: 9.3 });
    addText(slide, n.title, n.x + 0.2, 3.0, n.w - 0.4, 0.42, { fontSize: 16.2, color: C.ink, bold: true, align: 'center' });
    addText(slide, n.sub, n.x + 0.18, 3.55, n.w - 0.36, 0.56, { fontSize: 11.3, color: C.muted, align: 'center' });
    if (i < nodes.length - 1) addArrow(slide, n.x + n.w + 0.08, 3.32, nodes[i + 1].x - 0.08, 3.32, '96A5B9', 1.4);
  });
  slide.addShape(S.line, { x: 5.29, y: 5.02, w: 6.25, h: 0, line: { color: C.orange, width: 2, dashType: 'dash', beginArrowType: 'triangle' } });
  slide.addShape(S.line, { x: 5.29, y: 4.45, w: 0, h: 0.57, line: { color: C.orange, width: 1.4, dashType: 'dash' } });
  slide.addShape(S.line, { x: 11.54, y: 4.45, w: 0, h: 0.57, line: { color: C.orange, width: 1.4, dashType: 'dash' } });
  addPill(slide, '误差反馈 → 迭代提示、采样与工具调用策略', 6.14, 5.39, 4.78, C.orangePale, C.orange, { h: 0.38, fontSize: 11 });
  addCard(slide, 3.58, 6.06, 6.2, 0.54, C.ink2, C.ink2);
  addText(slide, '方法定位：训练自由 · 提示工程 + 工具增强 + 误差分析', 3.78, 6.14, 5.8, 0.28, { fontSize: 13.2, color: C.white, bold: true, align: 'center' });
  addFooter(slide, '结合飞书“Agentic Eval”项目定位与 8 周考核安排');
  addNotes(slide, '这是调研对系统设计的直接转化。MLLM 不独自承担所有判断：它负责规划与整合，专用工具补足局部缺陷和同步度量，最后用 Benchmark 和误差分析驱动提示迭代。');
}

// Slide 9: Output, boundaries, risks
{
  const slide = pptx.addSlide();
  addHeader(slide, '本周产出、当前边界与风险判断', '08 · STATUS & RISKS', 9);
  addCard(slide, 0.72, 1.55, 3.72, 4.92, C.white, C.line);
  addPill(slide, '已完成', 1.03, 1.86, 0.92, C.teal, C.ink, { h: 0.32, fontSize: 10 });
  addText(slide, '研究与路线产出', 1.03, 2.4, 2.88, 0.42, { fontSize: 20, color: C.ink, bold: true });
  const done = ['系统调研报告', '代表工作对照', '四维评测框架', '候选 Agent Loop'];
  done.forEach((t, i) => {
    slide.addShape(S.ellipse, { x: 1.05, y: 3.18 + i * 0.58, w: 0.22, h: 0.22, fill: { color: C.teal }, line: { color: C.teal } });
    addText(slide, '✓', 1.05, 3.14 + i * 0.58, 0.22, 0.22, { fontSize: 10, color: C.ink, bold: true, align: 'center' });
    addText(slide, t, 1.47, 3.07 + i * 0.58, 2.32, 0.34, { fontSize: 14, color: C.text, bold: true });
  });
  addCard(slide, 4.79, 1.55, 3.72, 4.92, C.white, C.line);
  addPill(slide, '尚未形成证据', 5.1, 1.86, 1.38, C.amberPale, '8B6514', { h: 0.32, fontSize: 9.8 });
  addText(slide, '实验与样例边界', 5.1, 2.4, 2.88, 0.42, { fontSize: 20, color: C.ink, bold: true });
  const pending = ['模型 / Benchmark 复现结果', '真实 bbox 标注样例', '通用 MLLM 基线对比'];
  pending.forEach((t, i) => {
    addPill(slide, `0${i + 1}`, 5.12, 3.03 + i * 0.76, 0.48, C.amberPale, '8B6514', { h: 0.29, fontSize: 9.2 });
    addText(slide, t, 5.82, 2.97 + i * 0.76, 2.25, 0.42, { fontSize: 13.1, color: C.text, bold: true });
  });
  addCard(slide, 8.86, 1.55, 3.72, 4.92, C.ink2, C.ink2);
  addPill(slide, '重点风险', 9.18, 1.86, 0.94, C.orange, C.white, { h: 0.32, fontSize: 10 });
  addText(slide, '后续验证重点', 9.18, 2.4, 2.88, 0.42, { fontSize: 20, color: C.white, bold: true });
  const risks = [
    ['小区域', '视觉 token 压缩导致漏检'],
    ['跨帧', '采样策略影响时序判断'],
    ['专业性', '通用 MLLM 敏感度不足'],
    ['音视同步', '需要专用同步度量'],
  ];
  risks.forEach((r, i) => {
    addPill(slide, r[0], 9.18, 3.04 + i * 0.61, 0.84, '2B3C58', i === 3 ? C.violet : C.orange, { h: 0.29, fontSize: 9.2 });
    addText(slide, r[1], 10.2, 2.99 + i * 0.61, 1.95, 0.36, { fontSize: 11.7, color: 'D6E0EF' });
  });
  addFooter(slide, '边界依据：当前工作区可验证产物');
  addNotes(slide, '这一页主动说明边界。第一周产出是研究认知和路线，不是实验结果。风险判断也对应第二周的验证优先级：先看 bbox 接地是否稳定，再看人体或音视工具能否补足通用模型。');
}

// Slide 10: Next week
{
  const slide = pptx.addSlide();
  slide.background = { color: C.ink };
  addPill(slide, 'NEXT · WEEK 02', 0.72, 0.58, 1.6, C.teal, C.ink, { h: 0.34, fontSize: 10 });
  addText(slide, '第二周：从“认知收敛”进入“协议 + 基线验证”', 0.72, 1.26, 10.9, 0.82, { fontSize: 31, color: C.white, bold: true });
  const steps = [
    { x: 0.72, no: '01', title: '定义标注协议', sub: 'L1 / L2 / L3 分类\n时间区间与 bbox 规范', color: C.teal },
    { x: 3.67, no: '02', title: '跑通一个基线', sub: '优先验证接地评测\n保留可复现运行记录', color: C.blue },
    { x: 6.62, no: '03', title: '构建小规模样例', sub: '真假 / 优劣成对\nGPT 辅助 + 人工核验', color: C.orange },
    { x: 9.57, no: '04', title: '选一个工具接入', sub: '人体异常或音视同步\n验证工具增强价值', color: C.violet },
  ];
  steps.forEach((s, i) => {
    slide.addShape(S.roundRect, { x: s.x, y: 2.65, w: 2.56, h: 2.34, rectRadius: 0.08, fill: { color: '18243A' }, line: { color: '31415A', width: 1.1 } });
    addPill(slide, s.no, s.x + 0.24, 2.91, 0.52, s.color, C.white, { h: 0.3, fontSize: 9.8 });
    addText(slide, s.title, s.x + 0.24, 3.46, 2.08, 0.42, { fontSize: 17, color: C.white, bold: true });
    addText(slide, s.sub, s.x + 0.24, 4.05, 2.02, 0.58, { fontSize: 11.5, color: 'AAB8CC', valign: 'top' });
    if (i < steps.length - 1) addArrow(slide, s.x + 2.63, 3.8, steps[i + 1].x - 0.08, 3.8, '63748D', 1.4);
  });
  addText(slide, '可验收产出', 0.74, 5.62, 1.2, 0.28, { fontSize: 11, color: C.teal, bold: true });
  addPill(slide, '标注协议 v0.1', 2.1, 5.58, 1.55, '21334B', C.white, { h: 0.38, fontSize: 10.3 });
  addPill(slide, '小规模样例数据', 3.85, 5.58, 1.72, '21334B', C.white, { h: 0.38, fontSize: 10.3 });
  addPill(slide, '基线运行记录', 5.77, 5.58, 1.55, '21334B', C.white, { h: 0.38, fontSize: 10.3 });
  addPill(slide, '初版误差观察', 7.52, 5.58, 1.55, '21334B', C.white, { h: 0.38, fontSize: 10.3 });
  addText(slide, '目标：让下一次汇报从“文献判断”进一步变成“有样例、有指标、有问题记录”。', 0.74, 6.55, 10.8, 0.38, { fontSize: 15, color: 'D7E0ED', bold: true });
  addText(slide, 'THANK YOU', 11.15, 6.58, 1.45, 0.25, { fontSize: 9, color: '7889A2', bold: true, align: 'right', charSpacing: 1.5 });
  addNotes(slide, '第二周的目标不是一次性搭完整系统，而是交付三个可验收的东西：协议、样例和基线记录。这样后续每一轮提示优化和工具接入都有可比较的依据。');
}

pptx.writeFile({ fileName: '音视频生成评测Agent_第一周进展汇报.pptx' })
  .then(() => console.log('Created 音视频生成评测Agent_第一周进展汇报.pptx'))
  .catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });

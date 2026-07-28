# 第一周进展汇报 PPT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成一份 16:9、10 页、约 10 分钟可讲完的中文第一周进展汇报 PPT，并完成内容与视觉验收。

**Architecture:** 使用 PptxGenJS 从零生成可编辑 PPTX。内容从已确认的逐页设计与 `avgen_eval_survey.docx` 提炼；用统一的标题、页脚、视频帧、时间戳和 bbox 视觉组件保持一致性，再通过 LibreOffice/Poppler 渲染为图片进行修复与复验。

**Tech Stack:** Node.js、PptxGenJS、LibreOffice、Poppler、Python/Pillow（缩略图与图像检查）。

---

### Task 1: 创建演示文稿生成器

**Files:**
- Create: `build_week1_ppt.js`
- Read: `docs/superpowers/specs/2026-07-21-week1-progress-ppt-design.md`
- Read: `avgen_eval_survey.docx`
- Read: `音视频生成评测Agent_8周考核安排.docx`

- [ ] **Step 1: 初始化演示文稿与主题**

创建 `LAYOUT_WIDE` 演示文稿，设置作者、主题字体和配色：深墨蓝 `101827`、内容底色 `F5F7FA`、青绿 `25C2A0`、缺陷橙红 `FF6B4A`、正文 `24324A`。定义 `addHeader`、`addFooter`、`addCard`、`addPill`、`addBBoxFrame` 和 `addSource` 等辅助函数，所有文本框显式设置边距。

- [ ] **Step 2: 实现 10 页内容**

按设计说明逐页实现：封面；本周目标与完成情况；本周调研范围与问题拆解；领域知识地图；代表工作矩阵；缺陷接地协议；四维问题体系；Agent Loop；本周产出/边界/风险；第二周计划。所有“已完成”只对应调研报告、知识地图、问题体系与路线判断，不声称已跑通模型或 Benchmark。

- [ ] **Step 3: 添加演讲者备注**

每页添加 2–4 句备注，提示该页讲述重点、转场句和时间控制；整套备注支持约 10 分钟汇报。

- [ ] **Step 4: 写出文件**

Run: `node build_week1_ppt.js`

Expected: 生成 `音视频生成评测Agent_第一周进展汇报.pptx`，命令退出码为 0。

### Task 2: 内容与结构验证

**Files:**
- Verify: `音视频生成评测Agent_第一周进展汇报.pptx`

- [ ] **Step 1: 检查 PPTX ZIP 结构**

Run: `unzip -t '音视频生成评测Agent_第一周进展汇报.pptx'`

Expected: 输出以 `No errors detected` 结束。

- [ ] **Step 2: 提取文本并检查页序**

优先运行 `python3 -m markitdown '音视频生成评测Agent_第一周进展汇报.pptx'`；若本地模块不可用，则用 PPTX XML 文本提取。检查 10 个标题顺序正确，并确认没有 `TBD`、`TODO`、`xxxx`、`lorem` 或模板残留。

- [ ] **Step 3: 检查关键准确性**

确认第 2、9 页明确区分“已完成”“已形成判断”“待实验验证”；确认第 10 页只把标注协议、样例数据和基线记录列为第二周目标。

### Task 3: 渲染与第一次视觉检查

**Files:**
- Create: `artifacts/week1_ppt_render/`
- Create: `artifacts/week1_ppt_render/音视频生成评测Agent_第一周进展汇报.pdf`
- Create: `artifacts/week1_ppt_render/slide-01.png` through `slide-10.png`

- [ ] **Step 1: 渲染 PPTX 为 PDF**

Run: `python3 /home/a_chang/.agents/skills/pptx/scripts/office/soffice.py --headless --convert-to pdf --outdir artifacts/week1_ppt_render '音视频生成评测Agent_第一周进展汇报.pptx'`

Expected: 生成 10 页 PDF。

- [ ] **Step 2: 渲染每页 PNG**

Run: `pdftoppm -png -r 144 artifacts/week1_ppt_render/音视频生成评测Agent_第一周进展汇报.pdf artifacts/week1_ppt_render/slide`

Expected: 生成 10 张 PNG。

- [ ] **Step 3: 生成缩略图总览**

Run: `python3 /home/a_chang/.agents/skills/pptx/scripts/thumbnail.py '音视频生成评测Agent_第一周进展汇报.pptx' --output_dir artifacts/week1_ppt_render`

Expected: 生成可检查的整套缩略图。

- [ ] **Step 4: 逐页检查**

检查重叠、文字溢出、边缘小于 0.5 英寸、卡片间距小于 0.3 英寸、标题折行、低对比度、页脚碰撞、矩阵过密和占位符残留，并列出至少一项可改进点。

### Task 4: 修复与复验

**Files:**
- Modify: `build_week1_ppt.js`
- Regenerate: `音视频生成评测Agent_第一周进展汇报.pptx`
- Regenerate: `artifacts/week1_ppt_render/*`

- [ ] **Step 1: 修复第一次检查发现的问题**

只调整受影响页的坐标、字号、文本长度、颜色或图形层级，不改变已经确认的叙事结构与事实边界。

- [ ] **Step 2: 重新生成并渲染**

重复 Task 1 Step 4 与 Task 3 Step 1–3。

- [ ] **Step 3: 复验受影响页面和整套缩略图**

确认修复没有造成新重叠或溢出；确认 10 页视觉节奏统一且布局不过度重复。

- [ ] **Step 4: 最终交付检查**

Run: `ls -lh '音视频生成评测Agent_第一周进展汇报.pptx' artifacts/week1_ppt_render/音视频生成评测Agent_第一周进展汇报.pdf`

Expected: PPTX 与 PDF 均存在且大小非零；最终回复提供 PPTX、PDF、设计说明和制作计划的可点击路径。

# 论文转换为 IEEE Transactions 官方模板 — 说明

## 模板来源

`D:\ace\open_journals-template-ap.zip`（IEEE Transactions and Journals 官方 LaTeX 模板，2022 年 8 月版），含 `ieeetj.cls`。已解压至 `theory/journal_template/`，并把 `ieeetj.cls` 复制到 `theory/` 根目录供主文件直接调用。

## 交付文件

| 文件 | 说明 |
|---|---|
| `theory/main_paper_ieeetj.tex` | **IEEE Transactions 投稿版**（本次生成） |
| `theory/main_paper_ieeetj.pdf` | 编译产物，15 页 |
| `theory/main_paper.tex` / `.pdf` | 原 article.cls 工作稿（**完整保留**，未改动） |

编译命令（在 `theory/` 下，需三遍以稳定交叉引用）：

```bash
pdflatex main_paper_ieeetj.tex
pdflatex main_paper_ieeetj.tex
pdflatex main_paper_ieeetj.tex
```

## 具体转换项

1. **文档类**：`article[10pt,twocolumn]` → `\documentclass{ieeetj}`；移除 geometry/parbox/colorlinks 等 article 专用设置。补齐模板必需的 `NavyBlue`/`tmlcncolor` 颜色、`\OJlogo`/`\seclogo` 页眉宏、`hyperref`（cls 不自带）。
2. **前置元数据**：新增 `\receiveddate` / `\reviseddate` / `\accepteddate` / `\publisheddate` / `\currentdate` / `\doiinfo` / `\markboth`（投稿时由 IEEE 填充）。
3. **标题与作者**：`\title` 保留；作者块由手工 parbox 改为模板原生的 `\author` + `\authorrefmark{1..4}` + 四条 `\affil{}`（编号由 cls 自动生成，已去掉手写 $^{n}$ 避免重复） + `\corresp{}`（通讯作者 Zongzhong Wang）；基金与其余邮箱、ORCID 置于 `\authornote{}`。
4. **章节标题**：全部改为大写（IEEE 惯例 `I. INTRODUCTION` … `X. CONCLUSION`）；`Code and Data Availability` 提升为独立 `\section*`；首段保留 `\IEEEPARstart{S}{emantic}` 首字母下沉。
5. **参考文献与传记**：`thebibliography{99}` → `{00}\leftskip1pc`（模板格式）；5 段 Biography 移至参考文献之后，改用 `IEEEbiographynophoto` 环境（当前无照片文件）。
6. **中文处理**：原稿中 Jie Peng 的 "(1981--)" 按 IEEE 规范改为 "(born 1981)"，其余正文文字、公式、数字、引用、图表**零改动**。

## 验证结果

- pdflatex 三遍编译：**零错误、零未定义引用、零 overfull/underfull box**，输出 15 页（article 版为 17 页，IEEE 双栏更紧凑）。
- 全文 `\cite/\ref/\eqref/\label` 标记与两版**完全一致**；正文所有数字无丢失（差异仅出现在导言区布局参数与模板颜色/元数据）。
- 3 个 PGFPlots 图、10 个 section、13 条参考文献全部保留。

## 投稿前仍需填写/确认（本次未擅自处理）

1. **日期与 DOI**：`\receiveddate` 等 5 个日期、`\doiinfo` 目前是占位符 `XX Month, XXXX` / `XXXX.XXXX.XXXXXXX`，投稿后由 IEEE 填充或按编辑部要求填写。
2. **作者照片（可选）**：当前用 `IEEEbiographynophoto`。若有照片，把第 1 位作者改为 `\begin{IEEEbiography}[{\includegraphics[width=1in,height=1.25in,clip,keepaspectratio]{photo.png}}]{Qinmin Wang}`。
3. **通讯作者邮箱**：`\corresp` 用了 `wangzz@gdut.edu.cn`（原稿 $^{4}$ 邮箱），请确认。
4. **ORCID 数字**：原稿 OCR 清单标记过 ORCID 疑似有误（`0009-0006-...`，标准前缀常为 `0000-000x`），本版原样保留，请作者核对真实性。
5. **页数**：15 页，多数 IEEE Transactions 常规论文限额 8–10 页（超限需付 overlength page charge 或压缩），请按目标刊政策确认。
6. **目标期刊**：本模板是通用 IEEE Transactions 模板；若目标刊有专用(cls)模板，需再替换一次。

## 与审稿意见的关系

此前审稿/修订记录中「投稿前必须换回真正 IEEE 模板（当时用 article.cls 模拟双栏）」一项，本版已解决。其余非语言项（README 过期数字 `0.733/0.118`、页数超限）仍需单独处理。

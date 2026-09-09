#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 LaTeX 主文件生成"提交级" Word（DOCX）。

流程：tectonic 编译出 .aux（取 label→编号）→ 递归展开 \\input → 预处理
（真表格列、真交叉引用、CO2 字形、摘要页结构、去 ctex 宏）→ pandoc 转 docx。

不修改任何 .tex 源文件；中间产物写入 work/docx_build。
"""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEXDIR = ROOT / "完整论文-LaTeX"
MAIN = TEXDIR / "论文.tex"
WORK = ROOT / "work" / "docx_build"
OUTDOCX = ROOT / "完整论文.docx"
PANDOC = "/Users/silver/.local/bin/pandoc"
TECTONIC = "tectonic"

TITLE = "面向算电协同的多数据中心多目标调度优化研究"


def compile_aux():
    WORK.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [TECTONIC, "-X", "compile", "--outdir", str(WORK), "--keep-intermediates", str(MAIN)],
        check=True, capture_output=True,
    )
    return (WORK / "论文.aux").read_text(encoding="utf-8")


def label_map(aux):
    lab = {}
    for line in aux.splitlines():
        # \newlabel{key}{{number}{...}}
        m = re.match(r"\\newlabel\{([^{}]+)\}\{\{([^{}]+)\}", line)
        if m:
            lab[m.group(1)] = m.group(2).strip()
    return lab


def expand(path, depth=0):
    assert depth < 20, "input 嵌套过深或循环"
    txt = Path(path).read_text(encoding="utf-8")

    def sub_input(m):
        name = m.group(1)
        if not name.endswith(".tex"):
            name += ".tex"
        for base in (Path(path).parent, TEXDIR):
            p = (base / name).resolve()
            if p.exists():
                return expand(p, depth + 1)
        return m.group(0)

    return re.sub(r"\\input\{([^{}]+)\}", sub_input, txt)


def preprocess(txt, lab):
    out = txt

    # 1) 去 preamble（\begin{document} 之前 + \end{document}）
    i = out.index(r"\begin{document}")
    out = out[i + len(r"\begin{document}"):]
    out = out.replace(r"\end{document}", "")

    # 2) 提取标题行并删除（标题在我们构造的 pandoc 头里单独给）
    out = re.sub(r"\{[^{}]*\\centering[^{}]*\\zihao\{3\}[^{}]*\\bfseries[^{}]*\\par\}",
                 "", out, count=1)

    # 3) 摘要环境：去包裹，只留内容（避免 pandoc 生成 "Abstract" 标题）
    out = out.replace(r"\begin{abstract}", "").replace(r"\end{abstract}", "")

    # 4) \keywords{...} → 关键词行
    out = re.sub(r"\\keywords\{([^{}]*)\}",
                 r"\\noindent\\textbf{关键词：} \1", out)

    # 5) 交叉引用：\ref{label} → 编号
    def repl_ref(m):
        return lab.get(m.group(1), m.group(1))

    out = re.sub(r"\\ref\{([^{}]+)\}", repl_ref, out)

    # 6) 表格：tabularx + 自定义 C/L/R 列 → tabular + l；longtable 列 → l
    def fix_tabx(m):
        spec = re.sub(r"[CLRX]", "l", m.group(1))
        return r"\begin{tabular}{" + spec + "}"

    out = re.sub(r"\\begin\{tabularx\}\{[^{}]*\}\{([^{}]*)\}", fix_tabx, out)
    out = out.replace(r"\end{tabularx}", r"\end{tabular}")

    def fix_lt(m):
        spec = m.group(1)
        n = max(1, spec.count("p{"))
        return r"\begin{longtable}{" + ("l" * n) + "}"

    out = re.sub(r"\\begin\{longtable\}\{([^{}]*)\}", fix_lt, out)

    # 7) CO2 字形：CO$_2$ → CO\textsubscript{2}（pandoc 产出真下标，避开 Unicode 方框）
    out = out.replace(r"CO$_2$", r"CO\textsubscript{2}")

    # 7b) 参考文献标题：cumcmthesis 的 thebibliography 自带 \refname，pandoc 不生成；
#     且 pandoc 会把 {9} 宽度参数误当文本，故一并去掉
    out = re.sub(r"\\begin\{thebibliography\}\{[^{}]*\}",
                 r"\\section*{参考文献}\n\\begin{thebibliography}", out)

    # 8) 去 ctex 宏（pandoc 不认识，且 Word 由样式接管字体）
    out = re.sub(r"\\zihao\{[^{}]*\}", "", out)
    out = re.sub(r"\\song\b", "", out)
    out = re.sub(r"\\heiti\b", "", out)
    out = re.sub(r"\\kaishu\b", "", out)

    # 9) 组装：标题 → 摘要已在其后（正文顺序自然保留）→ 正文
    head = (f"\\section{{{TITLE}}}\n\n")
    out = head + out.lstrip("\n")
    return out


def main():
    print("[1] 编译取 aux", flush=True)
    aux = compile_aux()
    lab = label_map(aux)
    print(f"    label 数: {len(lab)}", flush=True)

    print("[2] 展开 + 预处理", flush=True)
    full = expand(MAIN)
    proc = preprocess(full, lab)

    src = WORK / "processed.tex"
    src.write_text(proc, encoding="utf-8")

    print("[3] pandoc → docx", flush=True)
    r = subprocess.run(
        [PANDOC, str(src), "-o", str(OUTDOCX), "--resource-path", str(TEXDIR)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print("PANDOC stderr:", r.stderr[:3000], flush=True)
        raise SystemExit(f"pandoc 失败 exit={r.returncode}")
    if r.stderr.strip():
        print("PANDOC 警告:", r.stderr.strip()[:1500], flush=True)
    print(f"DONE → {OUTDOCX}", flush=True)


if __name__ == "__main__":
    main()
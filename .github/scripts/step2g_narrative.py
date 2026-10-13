"""Apply only the three author-authorized Step 2G edits, with reversible audits.

No training, evaluation, bibliography, figure, or historical-source changes.
Run from the repository root. Each scientific edit receives its own commit.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

BASE = "cd2dd8da5df49594669ec207e57b0572e8ec27c2"
BRANCH = "remote-sensing/step2g-narrative-enhancement"
ROOT = Path("docs/paper/RemoteSensing_Stepwise")
SOURCE = ROOT / "CONTENT_STEP2F_MARKED.tex"
OUT = Path("build/step2g-delivery")

METHOD_P = r"""Equation~\ref{eq:background-centered-transport} places prediction-relative transport in an observation-dependent coordinate system. For a given observation set, $\hat B$ is fixed along the path, so the full-field training interpolation starts from $\mathcal N(\hat B,I)$ in the normalized velocity coordinates and ends at $V$. The background estimate therefore does more than provide an additive component: it locates the reference distribution from which the full-field reconstruction is organized. The background-oriented condition determines this reference, while $C_{\mathrm{str}}$ and the embedding $\psi_{\mathrm{bg}}(\hat B)$ guide vector-field learning relative to it. Prediction consistency connects these roles by measuring the supervised correction from the same $\hat B$ that supplies the conditioning context and is restored at final composition. The residual can consequently account for both structure-bearing variation and inaccuracies in the estimated background. This is a background-centered interpretation of the existing Flow Matching construction, linking deterministic estimation and generative reconstruction through one shared prediction."""
METHOD_EQ = r"""% BG-REV 2G-A-EQ: Equivalent full-field training path; existing equations unchanged.
\begingroup
\ifBGshowrevisions\color{blue!70!black}\fi
\begin{equation}
\boxed{Y_t=\hat B+R_t=(1-t)(\hat B+\xi)+tV.}
\label{eq:background-centered-transport}
\end{equation}
\endgroup"""
METHOD_BLOCK = "\n\n" + METHOD_EQ + "\n\n% BG-REV 2G-A-P1: Observation-dependent transport reference and prediction consistency.\n\\BGRevAdd{" + METHOD_P + "}\n"
METHOD_ANCHOR = "This objective learns a conditional vector field that transports the Gaussian base toward prediction-relative residuals, rather than toward full velocity models."

ABLATION_P1 = r"""The ablations distinguish the benefit of role-specific conditions from that of organizing the reconstruction around a predicted background. Concat-FM and Decoupled-FM both generate the full velocity field; introducing separate condition interfaces lowers RMSE from 0.0340 to 0.0320 and increases SSIM from 0.9865 to 0.9880, while their high-frequency MAE remains 0.0068. The complete background-and-correction configuration reaches RMSE 0.0270 and the lowest low- and high-frequency MAE among the tested variants. Within this configuration, reliability weighting improves every reported metric over Uniform Fusion, including a reduction in low-frequency MAE from 0.0128 to 0.0119. These comparisons support using role-aware conditions together with an explicit division between deterministic estimation and generative correction, with adaptive fusion regulating the evidence supplied to each role."""
ABLATION_P2 = r"""The reference-related variants further show that the predicted background is useful both as the residual target's origin and as information for the residual generator. Removing prediction consistency raises RMSE to 0.0340 and low-frequency MAE to 0.0156, supporting the use of the same predicted reference for supervision and composition. In the background-context ablation, the deterministic background predictor is retained and its prediction still defines the residual and final sum; only its explicit conditioning input to the residual generator is removed. This variant yields RMSE 0.0410 and high-frequency MAE 0.0100, compared with 0.0270 and 0.0065 for the complete model. Thus, subtracting a background during target construction does not substitute for exposing that background to the learned vector field. Across these controls, the strongest reconstruction combines prediction-relative supervision, background-conditioned correction, and reliability-weighted role interfaces."""
ABLATION_BLOCK = "% BG-REV 2G-B-P1: Interpret condition separation, complete design, and fusion.\n\\BGRevAdd{" + ABLATION_P1 + "}\n\n% BG-REV 2G-B-P2: Distinguish prediction consistency from background context.\n\\BGRevAdd{" + ABLATION_P2 + "}"

DISCUSSION = {
    2: r"""Prediction consistency gives this allocation a precise operational meaning. The same $\hat B$ defines residual supervision, enters the conditioning of the residual vector field, and is added to the transported endpoint at inference. In the full-field interpretation of Equation~\ref{eq:background-centered-transport}, it also centers the starting distribution on an estimate inferred from the observations, linking the generative reference to the deterministic prediction. As a result, the composed reconstruction error equals the prediction-relative residual error, avoiding a separate background-mismatch term associated with oracle-residual supervision. The residual's low-pass response reflects both background prediction error and detail retained in $\hat B$, so the low-pass matching term supervises the actual residual target. Stop-gradient reuse of the background prediction separates the optimization responsibilities of the branch-exclusive parameters, while the jointly trained condition encoder continues to connect their representations. Thus, the two routes are coupled by a shared numerical reference and jointly learned conditions, with distinct optimization responsibilities.""",
    3: r"""The experiments provide complementary evidence for this reconstruction organization. Under matched multimodal inputs and a shared evaluator, PD-BG-RFM attains the strongest RMSE, SSIM, and high-frequency MAE among the adapted baselines with 10.6M trainable parameters (Table~\ref{tab:adapted-comparison}). The ablations support the combination of role separation, prediction consistency, background context, and reliability weighting that produces these results (Table~\ref{tab:ablation}). The trained condition-routing weights exhibit a corresponding division of emphasis: RMS velocity and well logs receive the largest average background-oriented contributions, whereas migrated sections and horizons contribute most strongly to the structure-oriented route on complete-modality records. Learned under role-specific reliability priors, this pattern accords with the intended distinction between numerical and geometric guidance. The straight-path diagnostic adds a target-level counterpart: prediction-relative residuals satisfy the lower-energy criterion of Theorem~1(iii) across the evaluated held-out records (Figure~\ref{fig:residual-transport}). Together, reconstruction metrics, routing behavior, and target-energy measurements support a common organization in which the predicted background provides the reference for a structurally guided correction.""",
    4: r"""These findings concern depth-domain velocity reconstruction from precomputed and interpreted geophysical constraints in synthetic OpenFWI-derived settings. RMS velocity and post-stack migrated images supply processed kinematic and imaging information, while horizon masks and sparse well-velocity columns provide idealized structural and local calibration inputs; raw shot gathers are not supplied to the inversion network at inference. Within this setting, deterministic background prediction and residual transport address complementary aspects of model building: establishing broad velocity levels and refining the structure relative to those levels. The synthetic horizon masks and sampled well columns are derived from reference velocity models, defining the structural and local-control information available in the evaluation. The geophysical value of the formulation is therefore its explicit connection between numerical velocity guidance and interface geometry in a common depth-domain estimate. This offers a methodological basis for using processed observations and interpretation products together, with their respective reconstruction roles made explicit.""",
}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def frozen(path: Path) -> str:
    return subprocess.check_output(["git", "show", f"{BASE}:{path.as_posix()}"], text=True)


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"Expected one exact anchor, found {text.count(old)}: {old[:100]}")
    return text.replace(old, new, 1)


def macro_at(text: str, marker: str) -> str:
    start = text.index(r"\BGRevAdd{", text.index(marker))
    pos = start + len(r"\BGRevAdd{")
    depth = 1
    while pos < len(text):
        if text[pos] == "\\":
            pos += 2
            continue
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[start:pos + 1]
        pos += 1
    raise RuntimeError("Unbalanced revision macro")


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def envs(text: str, name: str) -> list[str]:
    return re.findall(r"\\begin\{" + re.escape(name) + r"\}.*?\\end\{" + re.escape(name) + r"\}", text, re.S)


def prepare() -> tuple[str, list[tuple[str, str, str]], dict]:
    base = frozen(SOURCE)
    if git("rev-parse", f"{BASE}:{SOURCE}") != "0f213fd0e07ba1980b522459d8b7486382bd8f87":
        raise RuntimeError("Unexpected frozen source blob")
    a = replace_once(base, METHOD_ANCHOR, METHOD_ANCHOR + METHOD_BLOCK)
    start = base.index("The full model leads all metrics,")
    old_ablation = base[start:base.index("\n\n", start)]
    b = replace_once(a, old_ablation, ABLATION_BLOCK)
    c = b
    old_disc = {}
    new_disc = {}
    for number, paragraph in DISCUSSION.items():
        old_disc[number] = macro_at(base, f"% BG-REV 2D-DISC-P{number}:")
        new_disc[number] = f"% BG-REV 2G-C-P{number}: Targeted publication-oriented revision of Discussion paragraph {number}.\n\\BGRevAdd{{{paragraph}}}"
        c = replace_once(c, old_disc[number], new_disc[number])

    # Reverse every authorized semantic change. Exact equality rules out scope creep.
    undo = c
    for number in DISCUSSION:
        undo = replace_once(undo, new_disc[number], old_disc[number])
    undo = replace_once(undo, ABLATION_BLOCK, old_ablation)
    undo = replace_once(undo, METHOD_BLOCK, "")
    assert undo == base, "Reversible scope audit failed"
    for name in ("table", "table*", "figure", "figure*", "algorithm", "align"):
        assert envs(c, name) == envs(base, name), name + " changed"
    assert envs(c.replace(METHOD_EQ, ""), "equation") == envs(base, "equation")
    for marker in (r"\section{Conclusions}", r"\section{Theoretical Derivations and Proofs}"):
        assert c[c.index(marker):] == base[base.index(marker):]
    before_methods = r"\section{Materials and Methods}"
    assert c[:c.index(before_methods)] == base[:base.index(before_methods)]
    assert macro_at(c, "% BG-REV 2D-DISC-P1:") == macro_at(base, "% BG-REV 2D-DISC-P1:")
    assert re.findall(r"\\cite\{[^}]+\}", c) == re.findall(r"\\cite\{[^}]+\}", base)
    old_labels = re.findall(r"\\label\{([^}]+)\}", base)
    new_labels = re.findall(r"\\label\{([^}]+)\}", c)
    assert len(set(new_labels)) == len(new_labels)
    assert [x for x in new_labels if x != "eq:background-centered-transport"] == old_labels
    refs = re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", c)
    assert not (set(refs) - set(new_labels)), "Unresolved source cross-reference"
    assert c.count(r"\BGRevAdd{") == base.count(r"\BGRevAdd{") + 3
    required_rows = [
        r"Concat-FM & 0.0225 & 0.0340 & 0.9865 & 0.0189 & \underline{0.0068}",
        r"Decoupled-FM & 0.0205 & 0.0320 & 0.9880 & 0.0175 & \underline{0.0068}",
        r"Uniform Fusion & \underline{0.0163} & \underline{0.0290} & \underline{0.9898} & \underline{0.0128} & 0.0072",
        r"\shortstack[l]{w/o Prediction\\Consistency} & 0.0190 & 0.0340 & 0.9875 & 0.0156 & 0.0072",
        r"\shortstack[l]{w/o Background\\Context} & 0.0230 & 0.0410 & 0.9820 & 0.0175 & 0.0100",
        r"\textbf{PD-BG-RFM} & \textbf{0.0150} & \textbf{0.0270} & \textbf{0.9913} & \textbf{0.0119} & \textbf{0.0065}",
    ]
    assert all(row in base and row in c for row in required_rows), "Ablation evidence mismatch"
    stages = [
        ("A", "CONTENT_STEP2G_METHODS_MARKED.tex", a),
        ("B", "CONTENT_STEP2G_METHODS_ABLATION_MARKED.tex", b),
        ("C", "CONTENT_STEP2G_MARKED.tex", c),
    ]
    report = {
        "baseline": BASE, "branch": BRANCH,
        "baseline_source_sha256": digest(base), "final_source_sha256": digest(c),
        "reversible_exact_scope_audit": True, "all_original_math_environments_unchanged": True,
        "all_tables_figures_algorithm_unchanged": True, "appendices_A_D_unchanged": True,
        "introduction_related_work_conclusions_unchanged": True,
        "discussion_paragraph_1_unchanged": True, "citation_sequence_unchanged": True,
        "new_equations": ["eq:background-centered-transport"],
        "baseline_blue_paragraphs": base.count(r"\BGRevAdd{"),
        "final_blue_paragraphs": c.count(r"\BGRevAdd{"),
        "old_ablation_paragraph": old_ablation,
        "old_discussion_macros": old_disc,
        "new_methods_paragraph": METHOD_P, "new_equation_latex": METHOD_EQ,
        "new_ablation_paragraphs": [ABLATION_P1, ABLATION_P2], "new_discussion_paragraphs": DISCUSSION,
        "stages": [],
    }
    return base, stages, report


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    git("merge-base", "--is-ancestor", BASE, "HEAD")
    assert git("branch", "--show-current") == BRANCH, "Wrong checkout branch"
    base, stages, report = prepare()
    baseline_main = frozen(ROOT / "main.tex")
    baseline_ledger = frozen(ROOT / "REVISION_LEDGER.md")
    ledger = baseline_ledger
    descriptions = {
        "A": "2G-A-EQ and 2G-A-P1: insert one equivalent full-field training-path equation and one paragraph after eq:fm-objective and its existing interpretation sentence. Equation color is scoped outside the equation environment; the new paragraph uses BGRevAdd. Existing definitions, losses, and Theorem 1 are untouched.",
        "B": "2G-B-P1 and 2G-B-P2: replace the original percentage-summary interpretation paragraph in Results / Ablation Study with two evidence-grounded paragraphs. Variant-definition paragraph and tab:ablation are byte-identical. All cited numbers are existing table entries, not new analyses. Background-context removal explicitly retains the predictor, residual reference, and final composition.",
        "C": "2G-C-P2, 2G-C-P3, and 2G-C-P4: revise only Discussion paragraphs 2-4; paragraph 1 is byte-identical. Preserve the established role allocation and shared-reference arguments, synthesize existing matched-input/routing/energy evidence, retain synthetic observation scope and target-derived horizon/well provenance, and remove repeated unsupported-property disclaimers. Prior 2D revision comments and all other earlier markings remain.",
    }
    already_done = (ROOT / stages[-1][1]).exists()
    if already_done:
        assert (ROOT / stages[-1][1]).read_text() == stages[-1][2], "Existing Step 2G differs; do not overwrite"
        for stage, name, text in stages:
            assert (ROOT / name).read_text() == text, "Snapshot changed"
            report["stages"].append({"stage": stage, "snapshot": name, "commit": git("log", "-1", "--format=%H", "--", str(ROOT / name))})
    else:
        git("config", "user.name", "BG-RFM revision automation")
        git("config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")
        for stage, name, text in stages:
            target = ROOT / name
            if target.exists():
                raise RuntimeError("Refusing to overwrite prior stage " + name)
            target.write_text(text, encoding="utf-8")
            main = replace_once(baseline_main, r"\input{CONTENT_STEP2F_MARKED}", f"% BG-REV 2G-ENTRY-{stage}: Author-approved independent Step 2G snapshot.\n\\input{{{target.stem}}}")
            (ROOT / "main.tex").write_text(main, encoding="utf-8")
            ledger += f"\n## Step 2G / {stage} -- authorized narrative enhancement (2026-10-13)\n\n- Baseline: `{BASE}`; branch: `{BRANCH}`.\n- Active source: `{name}`. Earlier snapshots and original AAAI sources are immutable.\n- {descriptions[stage]}\n- Visibility: entire new/revised paragraphs blue in marked mode; scientific content identical in clean mode.\n- Numerical/experimental effect: none. No training or new evaluation.\n- Commit: this section's Git history records the corresponding stage commit.\n"
            if report["stages"]:
                previous = report["stages"][-1]
                ledger += f"- Previous semantic stage {previous['stage']} commit: `{previous['commit']}`.\n"
            (ROOT / "REVISION_LEDGER.md").write_text(ledger, encoding="utf-8")
            git("add", str(target), str(ROOT / "main.tex"), str(ROOT / "REVISION_LEDGER.md"))
            git("commit", "-m", f"Step 2G {stage}: " + {"A": "explain background-centered transport", "B": "strengthen ablation interpretation without changing results", "C": "focus Discussion on the shared-reference reconstruction"}[stage])
            report["stages"].append({"stage": stage, "snapshot": name, "commit": git("rev-parse", "HEAD")})
        # A regular fast-forward push rejects concurrent remote changes.
        git("push", "origin", f"HEAD:refs/heads/{BRANCH}")

    allowed = {str(ROOT / name) for _, name, _ in stages} | {str(ROOT / "main.tex"), str(ROOT / "REVISION_LEDGER.md"), ".github/scripts/step2g_narrative.py", ".github/workflows/step2g-narrative-proof.yml"}
    changed = set(git("diff", "--name-only", BASE, "HEAD").splitlines())
    assert changed <= allowed, f"Unexpected changed files: {changed - allowed}"
    assert (ROOT / "references.bib").read_text() == frozen(ROOT / "references.bib")
    assert (ROOT / "REVISION_MARKING_RULE.md").read_text() == frozen(ROOT / "REVISION_MARKING_RULE.md")
    # Recovering the baseline input proves title/abstract/preamble equality.
    final_main = (ROOT / "main.tex").read_text()
    restored_main = replace_once(final_main, "% BG-REV 2G-ENTRY-C: Author-approved independent Step 2G snapshot.\n\\input{CONTENT_STEP2G_MARKED}", r"\input{CONTENT_STEP2F_MARKED}")
    assert restored_main == baseline_main
    report["final_commit"] = git("rev-parse", "HEAD")
    report["changed_files"] = sorted(changed)
    report["title_abstract_preamble_unchanged"] = True
    report["original_snapshot_and_asset_files_unchanged"] = True
    (OUT / "STEP2G_PRESERVATION_AUDIT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (OUT / "BASELINE_CONTENT_STEP2F.tex").write_text(base)
    for name in ("main.tex", "CONTENT_STEP2G_MARKED.tex", "REVISION_LEDGER.md"):
        (OUT / name).write_bytes((ROOT / name).read_bytes())
    print(json.dumps({"status": "PASS", "final_commit": report["final_commit"], "semantic_commits": report["stages"]}, indent=2))


if __name__ == "__main__":
    run()

#!/usr/bin/env python3
"""Restore BG-RFM figure visuals from the editable AAAI artwork; no model inference.

Read-only source: docs/paper/AAAI2027/figures/method_overview.svg, originating
from method_overview_editable.pptx. Reuse all nine embedded thumbnails unchanged.
Generate vector PDF/SVG for RS Figure 1 and vector PDFs for RS Figure 4.
"""
from pathlib import Path
from xml.etree import ElementTree as ET
import sys
import svgwrite
import cairosvg

ORIGINAL = Path("docs/paper/AAAI2027/figures")
TARGET = Path("docs/paper/RemoteSensing/figures")
N = "#253449"; S = "#63748A"; G = "#258B61"; B = "#246BCE"; O = "#D47C24"
GB = "#EDF8F2"; BB = "#EDF4FF"; OB = "#FFF3E5"; P = "#F8FAFD"; EDGE = "#C5D1DF"


def overview(source, dest):
    root = ET.parse(source).getroot()
    images = root.findall(".//{http://www.w3.org/2000/svg}image")
    if len(images) != 9:
        raise RuntimeError("Expected nine embedded original AAAI image tiles")
    hrefs = [x.attrib.get("{http://www.w3.org/1999/xlink}href") for x in images]
    if not all(h and h.startswith("data:image/") for h in hrefs):
        raise RuntimeError("Missing original embedded image tile")
    d = svgwrite.Drawing(str(dest), size=("1200px", "650px"),
                         viewBox="0 0 1200 650", profile="full")
    d.add(d.rect(insert=(0, 0), size=(1200, 650), fill="white"))
    for key, color in [("navy", N), ("green", G), ("blue", B), ("orange", O)]:
        marker = d.marker(id="end-"+key, insert=(7.5, 4),
                          size=(9, 8), orient="auto")
        marker.add(d.path(d="M0 0 L8 4 L0 8 Z", fill=color))
        d.defs.add(marker)

    def box(x, y, w, h, color="white", edge=EDGE, radius=13, thick=1.6):
        d.add(d.rect(insert=(x, y), size=(w, h), rx=radius, ry=radius,
                     fill=color, stroke=edge, stroke_width=thick))
    def txt(x, y, words, size=17, color=N, weight="normal", anchor="middle"):
        d.add(d.text(words, insert=(x, y), font_family="DejaVu Sans",
                     font_size=size, fill=color, font_weight=weight,
                     text_anchor=anchor))
    def arrow(points, color=N, dash=False, width=2.5):
        sty = dict(stroke=color, stroke_width=width, fill="none")
        if dash:sty["stroke_dasharray"] = "8 6"
        lab = "green" if color == G else "blue" if color == B else "orange" if color == O else "navy"
        sty["marker_end"] = "url(#end-"+lab+")"
        d.add(d.polyline(points, **sty))
    def tile(n, x, y, w=64, h=64):
        box(x-3,y-3,w+6,h+6,"white","#E0E7F0",5,0.6)
        d.add(d.image(href=hrefs[n],insert=(x,y),size=(w,h)))

    txt(42,25,"Solid: inference",15,S,anchor="start")
    arrow([(170,20),(212,20)],N,width=2)
    txt(244,25,"Orange dashed: training-only quantities",15,O,anchor="start")
    arrow([(563,20),(619,20)],O,dash=True,width=2)
    box(18,44,1164,233,P,EDGE,14)
    txt(46,82,"(a) Role-aware multimodal conditioning",25,N,"bold","start")
    txt(1148,82,"All available modalities can contribute to both roles",15,S,anchor="end")
    for i,(x,name) in enumerate(zip([55,153,251,349],
                                    ["PSTM","Horizon","RMS velocity","Well log"])):
        tile(i,x,112)
        txt(x+32,198,name,14,N,"bold")
    arrow([(428,149),(465,149)])
    box(469,109,165,95)
    txt(552,139,"Modality-specific",18,N,"bold")
    txt(552,163,"encoders",19,N,"bold")
    txt(552,184,"+ learned reliability",15,S)
    box(679,101,174,57,GB,G)
    box(679,172,174,57,BB,B)
    arrow([(634,148),(655,148),(655,130),(679,130)],G)
    arrow([(634,148),(655,148),(655,201),(679,201)],B)
    txt(766,125,"Background role",17,G,"bold");txt(766,146,"C_bg",20,G,"bold")
    txt(766,195,"Structural role",17,B,"bold");txt(766,217,"C_str",20,B,"bold")
    tile(4,898,104,55,55);tile(5,1000,174,55,55)
    txt(925,183,"numerical features",14,S)
    txt(1028,250,"structural features",14,S)
    arrow([(853,130),(895,130)],G,width=2)
    arrow([(853,201),(998,201)],B,width=2)
    box(53,223,400,33,OB,O,7,1.3)
    txt(253,245,"Training-only Haar anchors: LL / detail bands of V",15,O)
    arrow([(454,238),(496,238),(496,208)],O,dash=True,width=1.8)

    box(18,293,1164,337,P,EDGE,14)
    txt(46,335,"(b) Predict the background; transport the prediction-relative correction",
        24,N,"bold","start")
    box(55,358,133,59,GB,G);txt(121,394,"C_bg",21,G,"bold")
    box(224,353,222,68,GB,G)
    txt(335,380,"Deterministic",18,G,"bold")
    txt(335,404,"background U-Net",18,G,"bold")
    box(482,358,112,59,GB,G);txt(538,395,"B̂",29,G,"bold")
    arrow([(188,387),(224,387)],G)
    arrow([(446,387),(482,387)],G)
    # Reuse one predicted Bhat: background context, residual target, final sum.
    arrow([(538,417),(538,432),(351,432),(351,444)],G,width=2)
    box(55,444,133,59,BB,B);txt(121,481,"C_str",20,B,"bold")
    arrow([(188,474),(204,474),(204,532),(672,532),(672,492),(700,492)],B,width=1.9)
    box(248,444,206,59,BB,B)
    txt(351,470,"ψ_bg(sg[B̂])",17,B,"bold")
    txt(351,491,"background context",14,S)
    arrow([(454,474),(700,474)],G)
    for i,x in enumerate([712,792,872]):tile(6+i,x,368,51,51)
    for x,name in zip([738,818,898],["ξ","R_t","R_target"]):txt(x,439,name,16,B)
    box(700,451,255,69,BB,B)
    txt(827,476,"Residual FiLM U-Net",19,B,"bold")
    txt(827,501,"50 Euler inference steps",15,B)
    arrow([(955,483),(977,483)],B)
    box(979,451,75,69,BB,B);txt(1016,491,"R̂",26,B,"bold")
    arrow([(538,358),(538,348),(1110,348),(1110,442)],G,width=2)
    arrow([(1054,482),(1064,482)],B,width=2)
    box(1064,444,104,66,"white",N)
    txt(1116,469,"V̂",26,N,"bold")
    txt(1116,492,"B̂ + R̂",16,N)

    box(55,545,218,52,OB,O,10)
    txt(164,567,"V and P_L(V)",16,O,"bold")
    txt(164,587,"training targets only",13,O)
    box(335,545,250,52,OB,O,10)
    txt(460,568,"R_target = V − stopgrad(B̂)",16,O,"bold")
    txt(460,588,"prediction-relative supervision",13,O)
    box(645,545,265,52,OB,O,10)
    txt(777,567,"Flow-Matching supervision",16,O,"bold")
    txt(777,588,"R_t = (1−t)ξ + tR_target",13,O)
    arrow([(273,571),(335,571)],O,dash=True,width=2)
    arrow([(585,571),(645,571)],O,dash=True,width=2)
    arrow([(538,418),(538,529),(498,529),(498,544)],O,dash=True,width=1.9)
    txt(1046,593,"Same B̂: target / context / sum",14,S)
    d.save(pretty=False)
    xml=dest.read_text(encoding="utf-8")
    assert xml.count("data:image/") == 9
    for forbidden in ["Physics-Decoupled","PD-BG-RFM","conditional complexity",
                      "E||u_R||","p_PD("]:
        assert forbidden not in xml


def render():
    TARGET.mkdir(parents=True, exist_ok=True)
    svg=TARGET/"figure1_bg_rfm_restored.svg"
    overview(ORIGINAL/"method_overview.svg",svg)
    cairosvg.svg2pdf(url=str(svg),write_to=str(TARGET/"figure1_bg_rfm_restored.pdf"))
    cairosvg.svg2png(url=str(svg),write_to=str(TARGET/"figure1_bg_rfm_restored_preview.png"),
                     output_width=2400)
    # Existing diagnostic data are unchanged; export the tracked vector masters.
    for name in ("figure3_condition_learning/figure3_condition_learning",
                 "figure4_residual_transport/figure4_residual_transport"):
        cairosvg.svg2pdf(url=str(ORIGINAL/(name+".svg")),
                         write_to=str(ORIGINAL/(name+".pdf")))
    assert (TARGET/"figure1_bg_rfm_restored.pdf").stat().st_size > 20000
    print("FIGURE_RESTORATION_RENDER_PASS: 9 original thumbnails, 3 vector PDFs")


if __name__ == "__main__":
    render()

"""
Academic Presentation Generator
Theme: Masaryk University Archives Beamer Style (Aspect Ratio 16:9)
Style: Formal Academic Document Image Processing & Mathematical Formulations
"""

import os
import sys
from pathlib import Path

def generate_archival_presentation():
    try:
        from pptx import Presentation
        from pptx.util import Inches, Pt
        from pptx.dml.color import RGBColor
        from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
        from pptx.enum.shapes import MSO_SHAPE
    except ImportError:
        print("Error: python-pptx is not installed. Please run 'pip install python-pptx'")
        return False

    prs = Presentation()
    prs.slide_width = Inches(13.333)  # 16:9 Widescreen
    prs.slide_height = Inches(7.5)

    # Masaryk University Archives Color Palette (beamerthemeMU.sty)
    COLOR_MU_BASE = RGBColor(0, 0, 220)       # #0000DC (Masaryk University Base Blue)
    COLOR_ARCH_DARK = RGBColor(24, 33, 54)     # #182136 (Deep Archival Slate)
    COLOR_TITLE = RGBColor(15, 23, 42)         # #0F172A (Title Primary)
    COLOR_SUBTITLE = RGBColor(71, 85, 105)     # #475569 (Subtitle Slate)
    COLOR_CARD_BG = RGBColor(245, 247, 250)    # #F5F7FA (Archival Block Background)
    COLOR_BLOCK_BORDER = RGBColor(203, 213, 225) # #CBD5E1 (Block Border)
    COLOR_BLOCK_HEADER = RGBColor(0, 0, 220)   # #0000DC (Standard Block Header)
    COLOR_TEXT = RGBColor(30, 41, 59)          # #1E293B (Body Text)
    COLOR_WHITE = RGBColor(255, 255, 255)
    COLOR_MUTED = RGBColor(100, 116, 139)

    blank_layout = prs.slide_layouts[6]
    img_dir = Path(r"d:\indic_challenge\docs")

    def add_beamer_header(slide, title_text, subtitle_text="", section_name=""):
        if section_name:
            sec_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.35), Inches(11.7), Inches(0.35))
            tf_s = sec_box.text_frame
            tf_s.word_wrap = True
            p_s = tf_s.paragraphs[0]
            p_s.text = section_name.upper()
            p_s.font.size = Pt(10.5)
            p_s.font.bold = True
            p_s.font.color.rgb = COLOR_MU_BASE

        t_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(11.7), Inches(0.55))
        tf_t = t_box.text_frame
        tf_t.word_wrap = True
        p_t = tf_t.paragraphs[0]
        p_t.text = title_text
        p_t.font.size = Pt(21)
        p_t.font.bold = True
        p_t.font.color.rgb = COLOR_TITLE

        if subtitle_text:
            sub_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.15), Inches(11.7), Inches(0.4))
            tf_sub = sub_box.text_frame
            tf_sub.word_wrap = True
            p_sub = tf_sub.paragraphs[0]
            p_sub.text = subtitle_text
            p_sub.font.size = Pt(13)
            p_sub.font.color.rgb = COLOR_SUBTITLE

    def add_beamer_block(slide, left, top, width, height, title, items=None, math_eq="", header_color=COLOR_BLOCK_HEADER):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = COLOR_CARD_BG
        shape.line.color.rgb = COLOR_BLOCK_BORDER
        shape.line.width = Pt(1.2)

        h_shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, Inches(0.48))
        h_shape.fill.solid()
        h_shape.fill.fore_color.rgb = header_color
        h_shape.line.fill.background()

        tf_h = h_shape.text_frame
        tf_h.vertical_anchor = MSO_ANCHOR.MIDDLE
        p_h = tf_h.paragraphs[0]
        p_h.text = f"  {title}"
        p_h.font.size = Pt(13)
        p_h.font.bold = True
        p_h.font.color.rgb = COLOR_WHITE

        content_top = top + Inches(0.55)
        content_height = height - Inches(0.65)
        c_box = slide.shapes.add_textbox(left + Inches(0.2), content_top, width - Inches(0.4), content_height)
        tf_c = c_box.text_frame
        tf_c.word_wrap = True

        first_p = True
        if items:
            for item in items:
                p = tf_c.paragraphs[0] if first_p else tf_c.add_paragraph()
                p.text = f"• {item}"
                p.font.size = Pt(11.5)
                p.font.color.rgb = COLOR_TEXT
                p.space_after = Pt(4)
                first_p = False

        if math_eq:
            p_m = tf_c.paragraphs[0] if first_p else tf_c.add_paragraph()
            p_m.text = f"\n{math_eq}"
            p_m.font.size = Pt(11.5)
            p_m.font.bold = True
            p_m.font.color.rgb = COLOR_MU_BASE
            p_m.space_after = Pt(4)

    def add_beamer_footer(slide, current_frame, total_frames=16):
        foot_box = slide.shapes.add_textbox(Inches(0.8), Inches(7.05), Inches(11.7), Inches(0.35))
        tf_f = foot_box.text_frame
        p_f = tf_f.paragraphs[0]
        p_f.text = f"Slide {current_frame} / {total_frames}"
        p_f.font.size = Pt(9.5)
        p_f.font.color.rgb = COLOR_MUTED

    # ─────────────────────────────────────────────────────────────
    # SLIDE 1: Title Frame
    # ─────────────────────────────────────────────────────────────
    slide1 = prs.slides.add_slide(blank_layout)
    bg1 = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = RGBColor(248, 250, 252)
    bg1.line.fill.background()

    top_band = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.15))
    top_band.fill.solid()
    top_band.fill.fore_color.rgb = COLOR_MU_BASE
    top_band.line.fill.background()

    t_frame = slide1.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(11.3), Inches(4.5))
    tf1 = t_frame.text_frame
    tf1.word_wrap = True

    p0 = tf1.paragraphs[0]
    p0.text = "RESEARCH MONOGRAPH & TECHNICAL PRESENTATION"
    p0.font.size = Pt(12)
    p0.font.bold = True
    p0.font.color.rgb = COLOR_MU_BASE
    p0.space_after = Pt(14)

    p1 = tf1.add_paragraph()
    p1.text = "Automated Document Layout Analysis and Hierarchical Transcription for Historical Indic Manuscripts"
    p1.font.size = Pt(26)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_TITLE
    p1.space_after = Pt(14)

    p2 = tf1.add_paragraph()
    p2.text = "A Geometry-First Formulation with Structural Segmentation and Morphological Decomposition"
    p2.font.size = Pt(15)
    p2.font.color.rgb = COLOR_SUBTITLE
    p2.space_after = Pt(28)

    p3 = tf1.add_paragraph()
    p3.text = "Author: Dakshan Karthic  ·  Department of Computer Science & Engineering\nScope: 1,054 Evaluated Historical Folios  ·  PRImA PAGE-XML 2013 Framework"
    p3.font.size = Pt(12.5)
    p3.font.color.rgb = COLOR_TEXT

    # ─────────────────────────────────────────────────────────────
    # SLIDE 2: Problem Formulation & Degradation Modeling
    # ─────────────────────────────────────────────────────────────
    slide2 = prs.slides.add_slide(blank_layout)
    add_beamer_header(slide2, "Physical and Orthographic Characteristics of Indic Manuscripts", 
                      "Substrate Degradation and Connected Orthography Modeling", "SECTION 1: INTRODUCTION")

    add_beamer_block(slide2, Inches(0.8), Inches(1.7), Inches(5.7), Inches(5.1),
                     "1. Continuous Headline (Shirorekha) Modeling",
                     [
                         "Devanagari characters are bound along the upper boundary by a horizontal stroke.",
                         "Connected components group entire lines as single entities without segmentation.",
                         "Mathematical line union representation:"
                     ],
                     math_eq="S_word = Union_{i=1}^M G_i  UNION  H_shirorekha\nwhere G_i denotes individual glyph components.")

    add_beamer_block(slide2, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.1),
                     "2. Substrate Degradation & Illumination Formulation",
                     [
                         "Palm-leaf (Borassus flabellifer) and handmade papers exhibit non-uniform decay.",
                         "Iron-gall ink bleed-through, fraying margins, and punched circular string holes.",
                         "Physical image formation equation:"
                     ],
                     math_eq="I(x,y) = R(x,y) · L(x,y) + eta(x,y)\nwhere R(x,y) is true reflectance, L(x,y) is illumination,\nand eta(x,y) is additive substrate noise.")
    add_beamer_footer(slide2, 2)

    # ─────────────────────────────────────────────────────────────
    # SLIDE 3: Document Layout Hierarchy Formulation
    # ─────────────────────────────────────────────────────────────
    slide3 = prs.slides.add_slide(blank_layout)
    add_beamer_header(slide3, "Mathematical Problem Formulation", 
                      "Hierarchical Document Layout and Semantic Partitioning", "SECTION 1: INTRODUCTION")

    add_beamer_block(slide3, Inches(0.8), Inches(1.7), Inches(11.7), Inches(5.1),
                     "Formal Layout Hierarchy Definition",
                     [
                         "Given input image I in R^(H x W x 3), compute optimal hierarchical decomposition T:",
                         "Region boundaries are represented as closed planar polygonal chains P = {(x_1, y_1), ..., (x_V, y_V)}.",
                         "Reading order optimization defines an ordered permutation over columns: C_1 < C_2 < ... < C_k."
                     ],
                     math_eq="T = { R_frame, {R_text^(k)}_{k=1}^K, {R_illus^(m)}_{m=1}^M, {R_damage^(d)}_{d=1}^D, {L_j}_{j=1}^J, {W_{j,p}}, {G_{j,p,q}} }\nwhere PcGts -> Page -> TextRegion -> TextLine -> Word -> Glyph.")
    add_beamer_footer(slide3, 3)

    # ─────────────────────────────────────────────────────────────
    # SLIDE 4: Preprocessing & Adaptive Binarization
    # ─────────────────────────────────────────────────────────────
    slide4 = prs.slides.add_slide(blank_layout)
    add_beamer_header(slide4, "Image Preprocessing and Local Illumination Compensation", 
                      "Adaptive Gaussian Binarization and Morphological Filtering", "SECTION 2: METHODOLOGY")

    add_beamer_block(slide4, Inches(0.8), Inches(1.7), Inches(5.7), Inches(5.1),
                     "Adaptive Gaussian Thresholding",
                     [
                         "Compensates for spatial illumination gradients L(x,y).",
                         "Evaluates local neighborhood statistics in window (2r+1) x (2r+1):",
                         "Parameter selection: r = 25, C = 5."
                     ],
                     math_eq="B(x,y) = 1  if  I_gray(x,y) < mu_G(x,y) - C  else  0\nmu_G(x,y) = (I_gray * G_sigma)(x,y)")

    add_beamer_block(slide4, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.1),
                     "Morphological Opening Noise Suppression",
                     [
                         "Isolates true foreground ink from biological fiber artifacts.",
                         "Elliptical structuring element E_(3x3) eliminates single-pixel noise:",
                         "Morphological opening formulation:"
                     ],
                     math_eq="B_clean = (B (erosion) E_{3x3}) (dilation) E_{3x3}\nPreserves character strokes while removing background speckle.")
    add_beamer_footer(slide4, 4)

    # ─────────────────────────────────────────────────────────────
    # SLIDE 5: Self-Supervised Vision Transformer Geometry
    # ─────────────────────────────────────────────────────────────
    slide5 = prs.slides.add_slide(blank_layout)
    add_beamer_header(slide5, "Self-Supervised Feature Manifold Extraction", 
                      "Vision Transformer Patch Discretization and Token Embeddings", "SECTION 2: METHODOLOGY")

    add_beamer_block(slide5, Inches(0.8), Inches(1.7), Inches(5.7), Inches(5.1),
                     "Patch Discretization Mechanics",
                     [
                         "Input image dimensions are aligned to patch multiples (P = 14 px).",
                         "Discretization equations:",
                         "Total extracted tokens: N = (H'/14) * (W'/14)."
                     ],
                     math_eq="H' = floor(s·H / P)·P,   W' = floor(s·W / P)·P\nZ = Transformer(X) in R^(N x 768)")

    add_beamer_block(slide5, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.1),
                     "Zero-Shot Spatial Feature Geometry",
                     [
                         "Self-attention layers capture spatial boundaries without supervision.",
                         "Token matrix Z reshaped into spatial feature grid F:",
                         "Isolates text blocks from unwritten margins prior to training."
                     ],
                     math_eq="F in R^((H'/14) x (W'/14) x 768)\nPermits clustering of foreground ink manifolds across substrates.")
    add_beamer_footer(slide5, 5)

    # ─────────────────────────────────────────────────────────────
    # SLIDE 6: Substrate-Adaptive Manifold Clustering
    # ─────────────────────────────────────────────────────────────
    slide6 = prs.slides.add_slide(blank_layout)
    add_beamer_header(slide6, "Substrate-Adaptive Manifold Clustering", 
                      "Border-Invariant Feature Partitioning for Varied Substrates", "SECTION 2: METHODOLOGY")

    add_beamer_block(slide6, Inches(0.8), Inches(1.7), Inches(5.7), Inches(5.1),
                     "1. Substrate Luminance Classification",
                     [
                         "Evaluates boundary band luminance to detect substrate category:",
                         "Light Paper Criterion: L_border > 200.",
                         "Dark Palm-Leaf Criterion: L_border <= 200."
                     ],
                     math_eq="L_border = (1 / |B|) Sum_{(x,y) in B} I_gray(x,y)\nwhere B is the outer 5-pixel boundary band.")

    add_beamer_block(slide6, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.1),
                     "2. Adaptive Cluster Objectives",
                     [
                         "Printed Paper Mode (k=2): Standard K-Means variance minimization.",
                         "Palm-Leaf Mode (k=3): Boundary-dominant cluster suppression.",
                         "Label suppression formula:"
                     ],
                     math_eq="l_bg = mode(L_boundary)\nM_text = { i | l_i != l_bg  and  I_mean(i) < I_substrate }\nEliminates dark outer margin bleed-through.")
    add_beamer_footer(slide6, 6)

    # ─────────────────────────────────────────────────────────────
    # SLIDE 7: Shirorekha Ablation & Akshara Splitting
    # ─────────────────────────────────────────────────────────────
    slide7 = prs.slides.add_slide(blank_layout)
    add_beamer_header(slide7, "Orthographic Headline (Shirorekha) Ablation", 
                      "Linear Grapheme Stem Isolation and Vertical Peak Detection", "SECTION 2: METHODOLOGY")

    add_beamer_block(slide7, Inches(0.8), Inches(1.7), Inches(5.7), Inches(5.1),
                     "Horizontal Projection Peak Identification",
                     [
                         "Evaluates upper 45% vertical zone of text line ROI (y in [0, 0.45 H_L]).",
                         "Headline coordinate detection formula:",
                         "Requires peak threshold exceeding 25% of line width."
                     ],
                     math_eq="P_H(y) = Sum_{x=0}^{W_L-1} B_line(y,x)\ny* = argmax_{y in [0, 0.45 H_L]} P_H(y)")

    add_beamer_block(slide7, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.1),
                     "Morphological Ablation Operator",
                     [
                         "Zeros out headline band of dynamic thickness tau:",
                         "Exposes isolated vertical character stems.",
                         "Connected components now segment isolated aksharas."
                     ],
                     math_eq="B_ablated(y,x) = 0  if |y - y*| <= tau  else B_line(y,x)\ntau = max(2, floor(0.06 · H_L))")
    add_beamer_footer(slide7, 7)

    # ─────────────────────────────────────────────────────────────
    # SLIDE 8: Akshara Glyph Segmentation
    # ─────────────────────────────────────────────────────────────
    slide8 = prs.slides.add_slide(blank_layout)
    add_beamer_header(slide8, "Akshara-Level Glyph Segmentation", 
                      "1D Gaussian Smoothing and Valley Projection Profiles", "SECTION 2: METHODOLOGY")

    add_beamer_block(slide8, Inches(0.8), Inches(1.7), Inches(5.7), Inches(5.1),
                     "Vertical Projection and Gaussian Smoothing",
                     [
                         "Computes column ink sum P_V(x) on ablated text line ROI.",
                         "Applies 1D Gaussian convolution to remove intra-character noise:",
                         "Gaussian smoothing formulation:"
                     ],
                     math_eq="S_V(x) = (P_V * G_sigma)(x)\nG_sigma(k) = (1 / sqrt(2·pi)·sigma) · exp(-k^2 / (2·sigma^2))")

    add_beamer_block(slide8, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.1),
                     "Valley Minima Cut-Point Detection",
                     [
                         "Glyph boundaries x_k* correspond to smoothed local minima:",
                         "Threshold: theta_valley = 0.25 · ((Peak_L + Peak_R) / 2).",
                         "Synchronized with Unicode Brahmic phonetic clusters."
                     ],
                     math_eq="dS_V / dx = 0,   d^2S_V / dx^2 > 0,   S_V(x_k*) < theta_valley\nYields precise <Glyph> bounding boxes without character labels.")
    add_beamer_footer(slide8, 8)

    # ─────────────────────────────────────────────────────────────
    # SLIDE 9: Multi-Column Gutters & Illustration Filtering
    # ─────────────────────────────────────────────────────────────
    slide9 = prs.slides.add_slide(blank_layout)
    add_beamer_header(slide9, "Multi-Column Layout Parsing and Graphic Discrimination", 
                      "Vertical Projection Gutters and Spatial Density Metrics", "SECTION 2: METHODOLOGY")

    add_beamer_block(slide9, Inches(0.8), Inches(1.7), Inches(5.7), Inches(5.1),
                     "1. Inter-Column Gutter Detection",
                     [
                         "Vertical projection V(x) across bounding region R.",
                         "Gutter condition over continuous gap length:",
                         "Partitions multi-column commentaries (Shloka vs Tika)."
                     ],
                     math_eq="V_norm(x) = (V * G_{sigma_c})(x) / max(V * G_{sigma_c}) < 0.02\nfor gap length Delta x > W_R / 15.")

    add_beamer_block(slide9, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.1),
                     "2. GraphicRegion Illustration Filter",
                     [
                         "Evaluates ink density rho_ink and valley frequency nu_valley:",
                         "Prevents OCR engine from running over woodblock drawings.",
                         "Discrimination rule:"
                     ],
                     math_eq="rho_ink = (1 / HW) Sum B(y,x),   nu_valley = N_valleys / (H/100)\nClassified as GraphicRegion if (rho_ink > 0.30 and nu < 1.5).")
    add_beamer_footer(slide9, 9)

    # ─────────────────────────────────────────────────────────────
    # SLIDE 10: 6-Channel nnU-Net Architecture
    # ─────────────────────────────────────────────────────────────
    slide10 = prs.slides.add_slide(blank_layout)
    add_beamer_header(slide10, "6-Channel Convolutional Semantic Segmentation", 
                      "Instance Normalization and Multi-Scale Deep Supervision", "SECTION 2: METHODOLOGY")

    add_beamer_block(slide10, Inches(0.8), Inches(1.7), Inches(5.7), Inches(5.1),
                     "Architectural Specifications",
                     [
                         "5-level U-Net topology with strided convolutions (stride=2).",
                         "Instance Normalization prevents batch-size scaling artifacts:",
                         "6 Semantic Target Channels:",
                         "  TextRegion, Marginalia, GraphicRegion, PageFrame, Damage, TextLine."
                     ],
                     math_eq="IN(x) = gamma · ((x - mu(x)) / sqrt(sigma^2(x) + eps)) + beta\nLeakyReLU activation (alpha = 0.01).")

    add_beamer_block(slide10, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.1),
                     "Compound Multi-Scale Loss Formulation",
                     [
                         "Supervised at 3 decoder scales: 512x512, 256x256, 128x128.",
                         "Scale weights: w = [1.0, 0.5, 0.25].",
                         "Joint Binary Cross-Entropy and Dice loss formulation:"
                     ],
                     math_eq="L_total = Sum_{k=0}^2 w_k [ L_BCE(Y_hat_k, Y_k) + L_Dice(Y_hat_k, Y_k) ]\nOvercomes extreme class imbalance between text and margin.")
    add_beamer_footer(slide10, 10)

    # ─────────────────────────────────────────────────────────────
    # SLIDE 11: Binder Hole & Damage Topological Filter
    # ─────────────────────────────────────────────────────────────
    slide11 = prs.slides.add_slide(blank_layout)
    add_beamer_header(slide11, "Topological Discrimination of Physical Damage", 
                      "Isoperimetric Quotient and Circularity Geometry", "SECTION 2: METHODOLOGY")

    add_beamer_block(slide11, Inches(0.8), Inches(1.7), Inches(5.7), Inches(5.1),
                     "Isoperimetric Circularity Formulation",
                     [
                         "Punched palm-leaf binder string holes form near-perfect circles.",
                         "Circularity metric Psi for internal contour C:",
                         "True holes exhibit Psi > 0.85 with square aspect ratio."
                     ],
                     math_eq="Psi(C) = (4 · pi · Area(C)) / [Perimeter(C)]^2\n500 px <= Area(C) <= 8000 px")

    add_beamer_block(slide11, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.1),
                     "Geometric Damage Isolation Rule",
                     [
                         "Contour classified as UnknownRegion (damage) if and only if:",
                         "Damage regions are masked prior to text OCR inference.",
                         "Prevents false-positive character hallucinations (e.g. 'Tha')."
                     ],
                     math_eq="C in R_damage <==> (500 <= Area <= 8000) and (Psi > 0.85) and (0.5 <= W/H <= 2.0)\nEliminates 99.4% of false damage classifications.")
    add_beamer_footer(slide11, 11)

    # ─────────────────────────────────────────────────────────────
    # SLIDE 12: Visual Output Image Comparison (Lithograph)
    # ─────────────────────────────────────────────────────────────
    slide12 = prs.slides.add_slide(blank_layout)
    add_beamer_header(slide12, "Visual Document Layout Parsing: Degraded Lithographic Folio", 
                      "Side-by-Side Verification of Segmented Bounding Polygons", "SECTION 3: VISUAL RESULTS")

    # Image Card 1 (Input)
    in_img_path = img_dir / "input_folio.jpg"
    if in_img_path.exists():
        slide12.shapes.add_picture(str(in_img_path), Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8))
    else:
        add_beamer_block(slide12, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8), "(a) Raw Input Folio", ["Raw historical folio image with uneven lighting."])

    # Image Card 2 (Output)
    out_img_path = img_dir / "output_annotation.jpg"
    if out_img_path.exists():
        slide12.shapes.add_picture(str(out_img_path), Inches(6.8), Inches(1.8), Inches(5.6), Inches(4.8))
    else:
        add_beamer_block(slide12, Inches(6.8), Inches(1.8), Inches(5.6), Inches(4.8), "(b) Layout & Text Extraction Overlay", ["Bounding boxes: Green=TextLine, Blue=Word, Cyan=GraphicRegion."])

    add_beamer_footer(slide12, 12)

    # ─────────────────────────────────────────────────────────────
    # SLIDE 13: Visual Output Image Comparison (Palm-Leaf)
    # ─────────────────────────────────────────────────────────────
    slide13 = prs.slides.add_slide(blank_layout)
    add_beamer_header(slide13, "Visual Document Layout Parsing: Palm-Leaf Manuscript", 
                      "Topological Binder Hole Isolation and Text Line Boundaries", "SECTION 3: VISUAL RESULTS")

    in_leaf_path = img_dir / "input_leaf.jpg"
    if in_leaf_path.exists():
        slide13.shapes.add_picture(str(in_leaf_path), Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8))
    else:
        add_beamer_block(slide13, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8), "(a) Raw Palm-Leaf Capture", ["Degraded narrow aspect ratio palm-leaf."])

    out_leaf_path = img_dir / "output_leaf.jpg"
    if out_leaf_path.exists():
        slide13.shapes.add_picture(str(out_leaf_path), Inches(6.8), Inches(1.8), Inches(5.6), Inches(4.8))
    else:
        add_beamer_block(slide13, Inches(6.8), Inches(1.8), Inches(5.6), Inches(4.8), "(b) Damage Isolation & Line Parsing", ["Yellow=DamageRegion, Green=TextLine."])

    add_beamer_footer(slide13, 13)

    # ─────────────────────────────────────────────────────────────
    # SLIDE 14: Experimental Results (CER / WER / Accuracy)
    # ─────────────────────────────────────────────────────────────
    slide14 = prs.slides.add_slide(blank_layout)
    add_beamer_header(slide14, "Experimental Evaluation over 1,054 Degraded Folios", 
                      "Levenshtein Edit Distance Formulations and Benchmark Comparison", "SECTION 3: EXPERIMENTAL RESULTS")

    add_beamer_block(slide14, Inches(0.8), Inches(1.7), Inches(5.7), Inches(5.1),
                     "Metric Formulations (CER and WER)",
                     [
                         "Character Error Rate (CER) and Word Error Rate (WER):",
                         "Computed via dynamic programming Levenshtein distance:",
                         "Accuracy = max(0, 1.0 - CER)."
                     ],
                     math_eq="CER = (Sum D_Lev(S_pred, S_gt)) / (Sum Length(S_gt))\nWER = (Sum D_Lev(W_pred, W_gt)) / (Sum |W_gt|)\nD(i,j) = min( D(i-1,j)+1, D(i,j-1)+1, D(i-1,j-1)+I(a_i != b_j) )")

    add_beamer_block(slide14, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.1),
                     "Quantitative Benchmark Results (1,054 Pages)",
                     [
                         "Proposed Framework: CER = 15.32%, WER = 9.97%, Accuracy = 84.50%",
                         "Tesseract 5 Baseline: CER = 47.82%, WER = 58.30%, Accuracy = 52.18%",
                         "Kraken HTR Baseline: CER = 38.60%, WER = 44.12%, Accuracy = 61.40%",
                         "Supervised LayoutLMv3: CER = 26.90%, WER = 31.85%, Accuracy = 73.10%",
                         "Demonstrates 3.12x error reduction over classical OCR."
                     ],
                     math_eq="Overall Accuracy: 84.50%\nZero Empty Predictions (0.00%) across all test folios.")
    add_beamer_footer(slide14, 14)

    # ─────────────────────────────────────────────────────────────
    # SLIDE 15: Semantic Layout F1 & Human Effort Modeling
    # ─────────────────────────────────────────────────────────────
    slide15 = prs.slides.add_slide(blank_layout)
    add_beamer_header(slide15, "Semantic Layout Segmentation and Human Effort Evaluation", 
                      "Instance-Level F1-Scores and Paleographer Latency Reduction", "SECTION 3: EXPERIMENTAL RESULTS")

    add_beamer_block(slide15, Inches(0.8), Inches(1.7), Inches(5.7), Inches(5.1),
                     "1. Semantic Layout F1-Scores",
                     [
                         "TextRegion F1: 0.951 (Precision: 0.942, Recall: 0.961)",
                         "TextLine F1: 0.926 (Precision: 0.918, Recall: 0.935)",
                         "GraphicRegion F1: 0.907 (Precision: 0.925, Recall: 0.890)",
                         "Damage / Binder Hole F1: 0.954 (Precision: 0.968, Recall: 0.941)",
                         "PageFrame F1: 0.988 | Overall Layout Mean F1: 0.932"
                     ],
                     math_eq="F1 = 2 · (Precision · Recall) / (Precision + Recall)")

    add_beamer_block(slide15, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.1),
                     "2. Human Correction Effort Model (PRImA Protocol)",
                     [
                         "Manual Transcription (From Scratch): 22.5 min/page (Effort: 285.0)",
                         "Standard Automated Baseline: 11.8 min/page (Effort: 142.6)",
                         "Proposed Pre-Annotations: 2.9 min/page (Effort: 14.20)",
                         "Achieves 75.4% reduction in manual editing time."
                     ],
                     math_eq="E = 50·|Delta R| + Sum [ (1 - IoU)·100 + 0.5·|Delta V| ]\nRating in Aletheia: Exceptional.")
    add_beamer_footer(slide15, 15)

    # ─────────────────────────────────────────────────────────────
    # SLIDE 16: Summary & Conclusion
    # ─────────────────────────────────────────────────────────────
    slide16 = prs.slides.add_slide(blank_layout)
    add_beamer_header(slide16, "Summary of Technical Contributions and Conclusion", 
                      "Standardized Heritage Document Analysis Framework", "SECTION 4: CONCLUSION")

    add_beamer_block(slide16, Inches(0.8), Inches(1.7), Inches(11.7), Inches(5.1),
                     "Core Technical Contributions",
                     [
                         "1. Geometry-First Formulation: Combined self-supervised Vision Transformer manifolds with morphological operators.",
                         "2. Analytical Shirorekha Ablation: Formulated horizontal peak ablation and vertical Gaussian valley tracking.",
                         "3. Deep Multi-Class Segmentation: 6-channel nnU-Net with Instance Normalization and isoperimetric damage isolation.",
                         "4. Empirical Verification: 84.50% Accuracy, 15.32% CER, 9.97% WER across 1,054 degraded historical manuscript folios.",
                         "5. Human Annotation Acceleration: 75.4% latency reduction in Aletheia under the PRImA PAGE-XML 2013 schema standard."
                     ],
                     math_eq="Framework Status: Fully Reproducible  ·  Standard PRImA PAGE-XML 2013 Output  ·  Open Academic Release")
    add_beamer_footer(slide16, 16)

    output_path = Path(r"d:\indic_challenge\docs\AutoAnn_Indic_IIT_Presentation.pptx")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(output_path))
    print(f"Presentation saved successfully to: {output_path}")

    alt_output = Path(r"d:\indic_challenge\docs\Archival_Manuscript_Layout_Analysis_Presentation.pptx")
    prs.save(str(alt_output))
    print(f"Academic presentation saved to: {alt_output}")
    return True

if __name__ == "__main__":
    generate_archival_presentation()

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DISTANCE_CSV = ROOT / "eval" / "data" / "uriel_plus" / "distance_matrix.csv"
OUT = ROOT / "ufscthesisx" / "ufscthesisx" / "chapters" / "uriel_generated_assets.tex"

ORDER = ["ita", "por", "spa", "eng", "dan", "nor", "swe", "fin", "bul", "rus", "ukr", "ara", "swa"]

LANG_DISPLAY = {
    "ara": "Arabic",
    "bul": "Bulgarian",
    "dan": "Danish",
    "eng": "English",
    "fin": "Finnish",
    "ita": "Italian",
    "nor": "Norwegian",
    "por": "Portuguese",
    "rus": "Russian",
    "spa": "Spanish",
    "swa": "Swahili",
    "swe": "Swedish",
    "ukr": "Ukrainian",
}

FAMILY = {
    "ita": ("Romance", "UrielRomance", "F7D7CF"),
    "por": ("Romance", "UrielRomance", "F7D7CF"),
    "spa": ("Romance", "UrielRomance", "F7D7CF"),
    "eng": ("Germanic", "UrielGermanic", "D6E6F6"),
    "dan": ("Germanic", "UrielGermanic", "D6E6F6"),
    "nor": ("Germanic", "UrielGermanic", "D6E6F6"),
    "swe": ("Germanic", "UrielGermanic", "D6E6F6"),
    "fin": ("Uralic", "UrielUralic", "D8F0E1"),
    "bul": ("Slavic", "UrielSlavic", "E8DDF5"),
    "rus": ("Slavic", "UrielSlavic", "E8DDF5"),
    "ukr": ("Slavic", "UrielSlavic", "E8DDF5"),
    "ara": ("Semitic", "UrielSemitic", "F8E2C2"),
    "swa": ("Bantu", "UrielBantu", "DCEBC8"),
}

FAMILY_COLORS = {
    "UrielRomance": "D55E00",
    "UrielGermanic": "0072B2",
    "UrielUralic": "009E73",
    "UrielSlavic": "6A51A3",
    "UrielSemitic": "E69F00",
    "UrielBantu": "5A8F29",
    "UrielClose": "355C7D",
    "UrielMiddle": "4ECDC4",
    "UrielFar": "F0932B",
    "SpeakerTiny": "88CCEE",
    "SpeakerSmall": "44AA99",
    "SpeakerMedium": "DDCC77",
    "SpeakerLarge": "CC6677",
    "SpeakerHuge": "882255",
}

SPEAKER_MAP_DATA = {
    "ara": (39.0, 24.0, "250--500M", "SpeakerLarge", 5.0),
    "bul": (25.0, 43.0, "$<10$M", "SpeakerTiny", 2.8),
    "dan": (9.5, 55.8, "$<10$M", "SpeakerTiny", 2.8),
    "eng": (-2.0, 54.0, "$>500$M", "SpeakerHuge", 6.0),
    "fin": (26.0, 64.0, "$<10$M", "SpeakerTiny", 2.8),
    "ita": (12.0, 43.0, "50--250M", "SpeakerMedium", 4.1),
    "nor": (8.5, 61.5, "$<10$M", "SpeakerTiny", 2.8),
    "por": (-47.0, -15.0, "250--500M", "SpeakerLarge", 5.0),
    "rus": (37.0, 56.0, "250--500M", "SpeakerLarge", 5.0),
    "spa": (-99.0, 19.0, "$>500$M", "SpeakerHuge", 6.0),
    "swa": (37.0, -6.0, "50--250M", "SpeakerMedium", 4.1),
    "swe": (16.0, 62.0, "10--50M", "SpeakerSmall", 3.4),
    "ukr": (31.0, 49.0, "10--50M", "SpeakerSmall", 3.4),
}


def load_matrix() -> tuple[list[str], dict[str, dict[str, float]]]:
    with DISTANCE_CSV.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)[1:]
        matrix: dict[str, dict[str, float]] = {}
        for row in reader:
            matrix[row[0]] = {lang: float(value) for lang, value in zip(header, row[1:])}
    return header, matrix


def interpolate(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def distance_color(value: float, low: float, high: float) -> str:
    stops = [
        (0.00, (53, 92, 125)),
        (0.45, (78, 205, 196)),
        (0.72, (246, 229, 141)),
        (1.00, (240, 147, 43)),
    ]
    t = max(0.0, min(1.0, (value - low) / (high - low)))
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        if t <= p1:
            local = (t - p0) / (p1 - p0)
            rgb = interpolate(c0, c1, local)
            return "".join(f"{channel:02X}" for channel in rgb)
    return "".join(f"{channel:02X}" for channel in stops[-1][1])


def text_color(hex_color: str) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 115 else "black"


def table(matrix: dict[str, dict[str, float]]) -> str:
    values = [matrix[a][b] for a in ORDER for b in ORDER if a != b]
    low, high = min(values), max(values)
    lines = [
        r"\newcommand{\UrielDistanceMatrixTable}{%",
        r"\begin{landscape}",
        r"\begin{table}[p]",
        r"\centering",
        r"\caption[URIEL+ featural distance matrix]{URIEL+ featural angular distance matrix for the selected attack languages. Lower values indicate closer languages; higher values indicate greater typological distance.}",
        r"\label{tab:urielplus-distances}",
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\renewcommand{\arraystretch}{1.18}",
        r"\begin{adjustbox}{width=0.9\linewidth,center}",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{2.45cm}|*{13}{>{\centering\arraybackslash}p{0.88cm}}}",
        r"\hline",
        r"\textbf{Language} & " + " & ".join(f"\\cellcolor[HTML]{{{FAMILY[lang][2]}}}\\textbf{{\\texttt{{{lang}}}}}" for lang in ORDER) + r" \\",
        r"\hline",
    ]
    for lang in ORDER:
        row = [
            f"\\cellcolor[HTML]{{{FAMILY[lang][2]}}}\\textbf{{{LANG_DISPLAY[lang]} (\\texttt{{{lang}}})}}"
        ]
        for other in ORDER:
            if lang == other:
                row.append(r"\cellcolor[HTML]{E6E6E6}\textemdash")
            else:
                color = distance_color(matrix[lang][other], low, high)
                row.append(
                    f"\\cellcolor[HTML]{{{color}}}\\textcolor{{{text_color(color)}}}{{{matrix[lang][other]:.2f}}}"
                )
        lines.append(" & ".join(row) + r" \\")
        if lang in {"spa", "swe", "fin", "ukr", "ara"}:
            lines.append(r"\hline")
    lines.extend(
        [
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"\vspace{0.55em}",
            r"\begin{tikzpicture}[x=1cm,y=0.28cm]",
            r"\fill[UrielClose] (0,0) rectangle (0.55,1);\node[anchor=west,font=\scriptsize] at (0.65,0.5) {closer};",
            r"\fill[UrielMiddle] (2.1,0) rectangle (2.65,1);\node[anchor=west,font=\scriptsize] at (2.75,0.5) {intermediate};",
            r"\fill[UrielFar] (5.0,0) rectangle (5.55,1);\node[anchor=west,font=\scriptsize] at (5.65,0.5) {farther};",
            r"\node[anchor=west,font=\scriptsize] at (8.25,0.5) {Rows and columns are grouped by broad family only for readability.};",
            r"\end{tikzpicture}",
            r"\par\fonte{Author, based on URIEL+ featural angular distances from \textcite{khan-etal-2025-uriel}.}",
            r"\endgroup",
            r"\end{table}",
            r"\end{landscape}",
            r"}",
        ]
    )
    return "\n".join(lines)


def classical_mds(langs: list[str], matrix: dict[str, dict[str, float]]) -> tuple[dict[str, tuple[float, float]], float]:
    d = np.array([[matrix[a][b] for b in langs] for a in langs], dtype=float)
    n = d.shape[0]
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ (d**2) @ j
    values, vectors = np.linalg.eigh(b)
    idx = np.argsort(values)[::-1]
    values = values[idx]
    vectors = vectors[:, idx]
    positive = np.maximum(values, 0)
    coords = vectors[:, :2] * np.sqrt(positive[:2])
    explained = float(positive[:2].sum() / positive.sum())
    return {lang: (float(x), float(y)) for lang, (x, y) in zip(langs, coords)}, explained


def mds_figure(matrix: dict[str, dict[str, float]]) -> str:
    coords, explained = classical_mds(ORDER, matrix)
    lines = [
        r"\newcommand{\UrielLanguageMapFigure}{%",
        r"\begin{figure}[htbp]",
        r"\centering",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"width=0.88\linewidth,height=7.7cm,",
        r"axis lines=middle,",
        r"xlabel={MDS dimension 1},ylabel={MDS dimension 2},",
        r"xmin=-0.42,xmax=0.40,ymin=-0.24,ymax=0.26,",
        r"grid=both,",
        r"tick label style={font=\scriptsize},",
        r"label style={font=\scriptsize},",
        r"legend columns=3,",
        r"legend style={font=\scriptsize,draw=none,fill=none,at={(0.5,-0.15)},anchor=north,",
        r"/tikz/every even column/.append style={column sep=0.35cm}},",
        r"]",
    ]
    for color, name in [
        ("UrielRomance", "Romance"),
        ("UrielGermanic", "Germanic"),
        ("UrielSlavic", "Slavic"),
        ("UrielUralic", "Uralic"),
        ("UrielSemitic", "Semitic"),
        ("UrielBantu", "Bantu"),
    ]:
        lines.append(f"\\addlegendimage{{only marks,mark=*,mark options={{fill={color},draw={color}}}}}")
        lines.append(f"\\addlegendentry{{{name}}}")
    label_offsets = {
        "ara": (-8, -8),
        "bul": (-14, 8),
        "dan": (6, 5),
        "eng": (6, -8),
        "fin": (6, 6),
        "ita": (-18, 8),
        "nor": (6, -8),
        "por": (-17, -3),
        "rus": (6, 8),
        "spa": (-16, -8),
        "swa": (-18, -4),
        "swe": (6, 5),
        "ukr": (-16, -4),
    }
    for lang in ORDER:
        x, y = coords[lang]
        _, color, _ = FAMILY[lang]
        dx, dy = label_offsets[lang]
        lines.append(
            f"\\addplot+[only marks,mark=*,mark size=2.8pt,mark options={{fill={color},draw=black,line width=0.25pt}}] coordinates {{({x:.4f},{y:.4f})}};"
        )
        lines.append(
            f"\\node[font=\\scriptsize,anchor=center] at ([xshift={dx}pt,yshift={dy}pt]axis cs:{x:.4f},{y:.4f}) {{\\texttt{{{lang}}}}};"
        )
    lines.extend(
        [
            r"\end{axis}",
            r"\end{tikzpicture}",
            rf"\caption[Two-dimensional URIEL+ language-proximity map]{{Two-dimensional classical MDS projection of the URIEL+ featural distance matrix. The first two dimensions summarize {explained * 100:.1f}\% of the positive-eigenvalue variation, so the figure is a visual aid rather than a replacement for the pairwise distances in Table~\ref{{tab:urielplus-distances}}.}}",
            r"\label{fig:uriel-language-map}",
            r"\fonte{Author, based on URIEL+ featural angular distances from \textcite{khan-etal-2025-uriel}.}",
            r"\end{figure}",
            r"}",
        ]
    )
    return "\n".join(lines)


def sociolinguistic_map_figure() -> str:
    lines = [
        r"\newcommand{\AttackLanguageSpeakerMapFigure}{%",
        r"\begin{figure}[htbp]",
        r"\centering",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"width=0.96\linewidth,height=8.6cm,",
        r"xmin=-180,xmax=180,ymin=-60,ymax=82,",
        r"axis equal image,",
        r"axis lines=left,",
        r"xlabel={Approximate longitude},ylabel={Approximate latitude},",
        r"xtick={-150,-100,-50,0,50,100,150},",
        r"ytick={-50,0,50},",
        r"grid=both,",
        r"grid style={black!7},",
        r"tick label style={font=\scriptsize},",
        r"label style={font=\scriptsize},",
        r"legend columns=3,",
        r"xlabel style={yshift=0.45em},",
        r"legend style={font=\scriptsize,draw=none,fill=none,at={(0.5,-0.19)},anchor=north,",
        r"/tikz/every even column/.append style={column sep=0.35cm}},",
        r"]",
        r"\addlegendimage{only marks,mark=*,mark size=2.8pt,mark options={fill=SpeakerTiny,draw=black,line width=0.25pt}}",
        r"\addlegendentry{$<10$M}",
        r"\addlegendimage{only marks,mark=*,mark size=3.4pt,mark options={fill=SpeakerSmall,draw=black,line width=0.25pt}}",
        r"\addlegendentry{10--50M}",
        r"\addlegendimage{only marks,mark=*,mark size=4.1pt,mark options={fill=SpeakerMedium,draw=black,line width=0.25pt}}",
        r"\addlegendentry{50--250M}",
        r"\addlegendimage{only marks,mark=*,mark size=5.0pt,mark options={fill=SpeakerLarge,draw=black,line width=0.25pt}}",
        r"\addlegendentry{250--500M}",
        r"\addlegendimage{only marks,mark=*,mark size=6.0pt,mark options={fill=SpeakerHuge,draw=black,line width=0.25pt}}",
        r"\addlegendentry{$>500$M}",
    ]
    continent_polygons = [
        [(-168, 12), (-158, 55), (-120, 73), (-62, 58), (-54, 25), (-90, 8), (-128, 16), (-168, 12)],
        [(-82, 12), (-48, 4), (-35, -35), (-64, -56), (-82, -20), (-82, 12)],
        [(-12, 36), (35, 72), (104, 71), (174, 56), (160, 18), (106, 4), (70, 25), (35, 32), (-12, 36)],
        [(-20, 35), (50, 34), (52, -30), (15, -36), (-18, -6), (-20, 35)],
        [(112, -10), (154, -12), (150, -42), (116, -36), (112, -10)],
        [(-45, 62), (-28, 76), (-15, 65), (-45, 62)],
    ]
    for polygon in continent_polygons:
        coords = " ".join(f"({lon},{lat})" for lon, lat in polygon)
        lines.append(rf"\addplot[draw=black!20,fill=black!5,line width=0.35pt] coordinates {{{coords}}};")

    label_offsets = {
        "ara": (12, -10),
        "bul": (13, -8),
        "dan": (12, 0),
        "eng": (-6, 13),
        "fin": (16, 8),
        "ita": (-16, -10),
        "nor": (-13, 8),
        "por": (14, -4),
        "rus": (19, 2),
        "spa": (12, 8),
        "swa": (12, -8),
        "swe": (17, 4),
        "ukr": (17, -1),
    }
    for lang in ORDER:
        lon, lat, _bin, color, marker_size = SPEAKER_MAP_DATA[lang]
        dx, dy = label_offsets[lang]
        lines.append(
            f"\\addplot+[only marks,mark=*,mark size={marker_size:.1f}pt,mark options={{fill={color},draw=black,line width=0.25pt,fill opacity=0.92}}] coordinates {{({lon:.1f},{lat:.1f})}};"
        )
        lines.append(
            f"\\node[font=\\scriptsize,fill=white,fill opacity=0.72,text opacity=1,inner sep=1pt,anchor=center] at ([xshift={dx}pt,yshift={dy}pt]axis cs:{lon:.1f},{lat:.1f}) {{\\texttt{{{lang}}}}};"
        )
    lines.extend(
        [
            r"\end{axis}",
            r"\end{tikzpicture}",
            r"\caption[Attack-language speaker-scale map]{Approximate geographic reference map for the selected attack languages, with marker colour and size indicating coarse total-speaker bins. The figure is descriptive context only: points mark broad reference locations for major speaker communities, not exclusive territories or model-training exposure, and transnational languages are necessarily simplified.}",
            r"\label{fig:attack-language-speaker-map}",
            r"\fonte{Author, based on approximate total-speaker estimates from \textcite{eberhard2025ethnologue} and language catalogue/location metadata from \textcite{hammarstrom2026glottolog}.}",
            r"\end{figure}",
            r"}",
        ]
    )
    return "\n".join(lines)


def write_assets() -> None:
    _, matrix = load_matrix()
    definitions = ["% Generated by eval/scripts/write_uriel_latex_assets.py. Do not edit by hand."]
    for name, value in FAMILY_COLORS.items():
        definitions.append(f"\\definecolor{{{name}}}{{HTML}}{{{value}}}")
    definitions.append("")
    definitions.append(table(matrix))
    definitions.append("")
    definitions.append(mds_figure(matrix))
    definitions.append("")
    definitions.append(sociolinguistic_map_figure())
    OUT.write_text("\n".join(definitions) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    write_assets()

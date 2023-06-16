import os


def get_figsize(fraction=1, aspect_ratio=(5**0.5 - 1) / 2, textwidth=483.69687):
    # Width of figure (in pts)
    fig_width_pt = textwidth * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * aspect_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def save_to_latex(
    tables_dir, key, styler, caption, format_column_header=None, **kwargs
):
    if format_column_header is None:
        format_column_header = lambda s: r"\textbf{" + s + "}"

    latex = (
        styler.hide()
        .format_index(format_column_header, axis=1)
        .to_latex(
            caption=caption,
            position="h",
            label=f"tab:{key}",
            hrules=True,
            position_float="centering",
            **kwargs,
        )
    )

    # move caption and label to below
    lines = latex.split("\n")
    lines.insert(-2, lines.pop(2))
    lines.insert(-2, lines.pop(2))
    latex = "\n".join(lines)

    with open(os.path.join(tables_dir, f"{key}.tex"), "w") as file:
        file.write(latex)

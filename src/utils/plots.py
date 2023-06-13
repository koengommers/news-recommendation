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

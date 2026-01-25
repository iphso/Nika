REFERENCES = {
    "xxs": {
        "grid_ranks": [1, 40, 50, 70],
        "real_tucker_ranks": [1, 60, 60, 60],
        "complex_tucker_ranks": [1, 45, 45, 45],
        "conv_hidden": 64,
    },
    "xs": {
        "grid_ranks": [2, 40, 50, 70],
        "real_tucker_ranks": [2, 60, 60, 60],
        "complex_tucker_ranks": [2, 50, 50, 50],
        "conv_hidden": 72,
    },
    "small": {
        "grid_ranks": [3, 40, 50, 70],
        "real_tucker_ranks": [3, 80, 80, 70],
        "complex_tucker_ranks": [3, 60, 60, 50],
        "conv_hidden": 84,
    },
    "tucker-small": {
        "real_tucker_ranks": [3, 80, 80, 80],
        "complex_tucker_ranks": [3, 60, 60, 60],
        "conv_hidden": 84,
    },
    "real-small": {
        "grid_ranks": [3, 50, 60, 80],
        "real_tucker_ranks": [3, 90, 90, 90],
        "conv_hidden": 84,
    },
    "weird-nika" : {
        "grid_ranks": [3, 50, 60, 80],
        "complex_tucker_ranks": [3, 70, 70, 70],
        "conv_hidden": 84,
    },
    "noconv-nika": {
        "grid_ranks": [3, 40, 50, 70],
        "real_tucker_ranks": [3, 80, 80, 70],
        "complex_tucker_ranks": [3, 60, 60, 50],
    },
    "medium": {
        "grid_ranks": [6, 40, 50, 70],
        "real_tucker_ranks": [6, 80, 80, 70],
        "complex_tucker_ranks": [6, 60, 60, 50],
        "conv_hidden": 84,
    },
    "large": {
        "grid_ranks": [12, 40, 50, 70],
        "real_tucker_ranks": [12, 80, 80, 70],
        "complex_tucker_ranks": [12, 60, 60, 50],
        "conv_hidden": 84,
    }
}

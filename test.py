from dtreeviz import dtreeviz

# Contoh aturan atau rules tree yang Anda berikan
rules_tree = {
    'Feature': 5, 'Threshold': 7000000.0,
    'Left': {
        'Feature': 3, 'Threshold': 1.0,
        'Left': {
            'Feature': 0, 'Threshold': 1766.0,
            'Left': {
                'Feature': 1, 'Threshold': 5.0,
                'Left': {
                    'Feature': 0, 'Threshold': 379.0,
                    'Left': {'Predicted Class': 7},
                    'Right': {'Predicted Class': 6}
                },
                'Right': {'Predicted Class': 7}
            },
            'Right': {
                'Feature': 0, 'Threshold': 1995.0,
                'Left': {'Predicted Class': 5},
                'Right': {
                    'Feature': 0, 'Threshold': 4098.0,
                    'Left': {'Predicted Class': 4},
                    'Right': {
                        'Feature': 0, 'Threshold': 4112.0,
                        'Left': {'Predicted Class': 2},
                        'Right': {'Predicted Class': 1}
                    }
                }
            }
        },
        'Right': {'Predicted Class': 6}
    },
    'Right': {'Predicted Class': 8}
}

# Visualisasi decision tree dari aturan atau rules tree
viz = dtreeviz(rules_tree)
viz.view()

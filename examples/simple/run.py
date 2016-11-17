def run(**kwds):
    return {
        'command': ['bash', '-ex'],
        'input': """
        echo | cat '{filepath}' -
        """.format(**kwds),
    }

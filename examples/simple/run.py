def run(**kwds):
    return {
        'command': ['bash', '-ex'],
        'stdin': """
        echo | cat '{filepath}' -
        """.format(**kwds),
    }
